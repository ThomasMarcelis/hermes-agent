"""Foreground memory calls must never overtake a queued session rebind."""

import contextvars
import json
import threading

import pytest

import agent.memory_manager as manager_module
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


_PROFILE = contextvars.ContextVar("memory-fence-profile", default="A")


class _BankProvider(MemoryProvider):
    def __init__(self, name="external"):
        self._name = name
        self.bank = ("A", "session-A")
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_switch = False

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.bank = (_PROFILE.get(), session_id)

    def get_tool_schemas(self):
        return [{"name": f"{self.name}_recall", "parameters": {"type": "object"}}]

    def _block(self):
        self.entered.set()
        assert self.release.wait(timeout=10)
        self.calls.append(("write", self.bank, _PROFILE.get()))

    def sync_turn(self, *args, **kwargs):
        self._block()

    def on_session_end(self, messages):
        self._block()

    def on_session_switch(self, new_session_id, **kwargs):
        if self.fail_switch:
            raise RuntimeError("session rebind failed")
        self.bank = (_PROFILE.get(), new_session_id)

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append(("tool", self.bank, _PROFILE.get()))
        return json.dumps({"bank": self.bank})

    def on_turn_start(self, *args, **kwargs):
        self.calls.append(("turn", self.bank, _PROFILE.get()))

    def prefetch(self, query, **kwargs):
        self.calls.append(("prefetch", self.bank, _PROFILE.get()))
        return ":".join(self.bank)


@pytest.mark.parametrize("boundary", ["switch", "new_session"])
def test_pending_rebind_withholds_foreground_calls_then_recovers(boundary, monkeypatch):
    """A queued old write/extraction cannot expose A's bank to B; A→B→A stays scoped."""
    monkeypatch.setattr(manager_module, "_SESSION_SWITCH_TIMEOUT_S", 0.0, raising=False)
    manager = MemoryManager()
    provider = _BankProvider()
    manager.add_provider(provider)
    token = None
    try:
        if boundary == "switch":
            manager.sync_all("old user", "old answer", session_id="session-A")
            assert provider.entered.wait(timeout=5)
            token = _PROFILE.set("B")
            manager.queue_session_switch("session-B")
        else:
            manager.commit_session_boundary_async([], new_session_id="session-B")
            assert provider.entered.wait(timeout=5)
            token = _PROFILE.set("B")
        result = json.loads(manager.handle_tool_call("external_recall", {}))
        manager.on_turn_start(1, "new question")
        context = manager.prefetch_all("new question", session_id="session-B")
        assert "error" in result
        assert context == ""
        assert provider.calls == []

        provider.release.set()
        assert manager.flush_pending(timeout=5)
        # A boundary commit executes extraction and switch under one admission scope.
        expected_profile = "B" if boundary == "switch" else "A"
        assert provider.calls == [("write", ("A", "session-A"), "A")]
        assert json.loads(manager.handle_tool_call("external_recall", {}))["bank"] == [expected_profile, "session-B"]
        manager.on_turn_start(1, "new question")
        assert manager.prefetch_all("new question") == f"{expected_profile}:session-B"

        _PROFILE.reset(token)
        token = None
        manager.queue_session_switch("session-A")
        assert manager.flush_pending(timeout=5)
        assert json.loads(manager.handle_tool_call("external_recall", {}))["bank"] == ["A", "session-A"]
    finally:
        if token is not None:
            _PROFILE.reset(token)
        provider.release.set()
        manager.shutdown_all()


def test_failed_rebind_withholds_only_failed_provider_until_retry():
    manager = MemoryManager()
    builtin = _BankProvider("builtin")
    external = _BankProvider()
    manager.add_provider(builtin)
    manager.add_provider(external)
    external.fail_switch = True
    try:
        manager.queue_session_switch("session-B")
        assert manager.flush_pending(timeout=5)
        result = json.loads(manager.handle_tool_call("external_recall", {}))
        manager.on_turn_start(1, "new question")
        context = manager.prefetch_all("new question")
        assert "error" in result
        assert context == "A:session-B"
        assert external.calls == []
        assert [call[0] for call in builtin.calls] == ["turn", "prefetch"]

        external.fail_switch = False
        manager.queue_session_switch("session-B")
        assert manager.flush_pending(timeout=5)
        assert json.loads(manager.handle_tool_call("external_recall", {}))["bank"] == ["A", "session-B"]
    finally:
        manager.shutdown_all()


def test_worker_reentry_does_not_wait_for_its_own_queued_rebind(monkeypatch):
    """A provider callback cannot wait on a switch queued behind itself."""
    monkeypatch.setattr(manager_module, "_SESSION_SWITCH_TIMEOUT_S", 30.0, raising=False)
    manager = MemoryManager()
    reentered = threading.Event()
    results = []

    class _ReentrantProvider(_BankProvider):
        def sync_turn(self, *args, **kwargs):
            self.entered.set()
            assert self.release.wait(timeout=10)
            results.append(json.loads(manager.handle_tool_call("external_recall", {})))
            reentered.set()

    provider = _ReentrantProvider()
    manager.add_provider(provider)
    try:
        manager.sync_all("old user", "old answer")
        assert provider.entered.wait(timeout=5)
        manager.queue_session_switch("session-B")
        provider.release.set()
        assert reentered.wait(timeout=5)
        assert "error" in results[0]
        assert manager.flush_pending(timeout=5)
        assert json.loads(manager.handle_tool_call("external_recall", {}))["bank"] == ["A", "session-B"]
    finally:
        provider.release.set()
        manager.shutdown_all()


def test_inline_executor_fallback_keeps_foreground_wait_bounded(monkeypatch):
    """The slow inline fallback holds admission serialization, not the foreground waiter."""
    import tools.daemon_pool

    def unavailable_executor(*args, **kwargs):
        raise RuntimeError("cannot start worker")

    monkeypatch.setattr(tools.daemon_pool, "DaemonThreadPoolExecutor", unavailable_executor)
    monkeypatch.setattr(manager_module, "_SESSION_SWITCH_TIMEOUT_S", 0.0)

    class _SlowSwitchProvider(_BankProvider):
        def on_session_switch(self, new_session_id, **kwargs):
            self.entered.set()
            assert self.release.wait(timeout=10)
            super().on_session_switch(new_session_id, **kwargs)

    manager = MemoryManager()
    provider = _SlowSwitchProvider()
    manager.add_provider(provider)
    switch = threading.Thread(target=lambda: manager.queue_session_switch("session-B"))
    switch.start()
    try:
        assert provider.entered.wait(timeout=5)
        assert "error" in json.loads(manager.handle_tool_call("external_recall", {}))
        assert provider.calls == []
    finally:
        provider.release.set()
        switch.join(timeout=5)
        manager.shutdown_all()
    assert not switch.is_alive()
    assert json.loads(manager.handle_tool_call("external_recall", {}))["bank"] == ["A", "session-B"]
