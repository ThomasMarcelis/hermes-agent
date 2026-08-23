"""Tests for the on_session_switch hook and session_id propagation.

Covers #6672: memory providers must be notified when AIAgent.session_id
rotates mid-process (via /resume, /branch, /reset, /new, or context
compression). Without the notification, providers that cache per-session
state in initialize() (Hindsight, and any plugin that stores session_id
for scoped writes) keep writing into the old session's record.
"""


import contextvars
import threading

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records every lifecycle call for assertion."""

    def __init__(self, name="rec"):
        self._name = name
        self.switch_calls: list[dict] = []
        self.sync_calls: list[dict] = []
        self.queue_calls: list[dict] = []
        self.initialize_calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:  # pragma: no cover - unused
        return True

    def initialize(self, session_id, **kwargs):
        self.initialize_calls.append({"session_id": session_id, **kwargs})

    def get_tool_schemas(self):
        return []

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self.sync_calls.append(
            {"user": user_content, "asst": assistant_content, "session_id": session_id}
        )

    def queue_prefetch(self, query, *, session_id=""):
        self.queue_calls.append({"query": query, "session_id": session_id})

    def on_session_switch(
        self,
        new_session_id,
        *,
        parent_session_id="",
        reset=False,
        **kwargs,
    ):
        self.switch_calls.append(
            {
                "new": new_session_id,
                "parent": parent_session_id,
                "reset": reset,
                "extra": kwargs,
            }
        )


# ---------------------------------------------------------------------------
# MemoryProvider ABC — default on_session_switch is a no-op
# ---------------------------------------------------------------------------


class _MinimalProvider(MemoryProvider):
    """Provider that does NOT override on_session_switch — ABC default must no-op."""

    @property
    def name(self) -> str:
        return "minimal"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):  # pragma: no cover - unused
        pass

    def get_tool_schemas(self):
        return []




# ---------------------------------------------------------------------------
# MemoryManager.on_session_switch — fan-out
# ---------------------------------------------------------------------------


def test_manager_fans_out_to_all_providers():
    mm = MemoryManager()
    # Only one external provider is allowed; use the builtin slot for p1.
    p1 = _RecordingProvider(name="builtin")
    p2 = _RecordingProvider(name="hindsight")
    mm.add_provider(p1)
    mm.add_provider(p2)

    mm.on_session_switch("new-sid", parent_session_id="old-sid", reset=False, reason="resume")

    assert len(p1.switch_calls) == 1
    assert len(p2.switch_calls) == 1
    for call in (p1.switch_calls[0], p2.switch_calls[0]):
        assert call["new"] == "new-sid"
        assert call["parent"] == "old-sid"
        assert call["reset"] is False
        assert call["extra"] == {"reason": "resume"}




def test_manager_isolates_provider_failures():
    """A provider that raises must not block other providers."""

    class _Broken(_RecordingProvider):
        def on_session_switch(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    mm = MemoryManager()
    # MemoryManager rejects a second external provider, so pair broken
    # (builtin slot) with a good external one.
    broken = _Broken(name="builtin")
    good = _RecordingProvider(name="good")
    mm.add_provider(broken)
    mm.add_provider(good)

    # Must not raise — exceptions in one provider are swallowed + logged
    mm.on_session_switch("new-sid", parent_session_id="old-sid")
    assert len(good.switch_calls) == 1
    assert good.switch_calls[0]["new"] == "new-sid"


def test_queued_switch_runs_after_write_with_caller_contexts():
    """Writes and ownership changes share one FIFO and retain admission context."""
    scope = contextvars.ContextVar("memory-switch-scope", default="missing")
    calls = []

    class _ContextProvider(_RecordingProvider):
        def sync_turn(self, user_content, assistant_content, *, session_id=""):
            calls.append(("write", session_id, scope.get()))

        def on_session_switch(self, new_session_id, **kwargs):  # type: ignore[override]
            calls.append(("switch", new_session_id, scope.get(), dict(kwargs)))

    mm = MemoryManager()
    mm.add_provider(_ContextProvider())

    token = scope.set("write-context")
    try:
        mm.sync_all("old user", "old assistant", session_id="old-sid")
    finally:
        scope.reset(token)
    token = scope.set("switch-context")
    try:
        mm.queue_session_switch(
            "new-sid",
            parent_session_id="old-sid",
            reason="resume",
        )
    finally:
        scope.reset(token)

    assert mm.flush_pending(timeout=2.0)
    assert calls == [
        ("write", "old-sid", "write-context"),
        (
            "switch",
            "new-sid",
            "switch-context",
            {"parent_session_id": "old-sid", "reset": False, "reason": "resume"},
        ),
    ]


def test_queued_switch_only_forwards_rewound_when_true():
    mm = MemoryManager()
    provider = _RecordingProvider()
    mm.add_provider(provider)

    mm.queue_session_switch("next", rewound=False)
    mm.queue_session_switch("next", rewound=True)

    assert mm.flush_pending(timeout=2.0)
    assert provider.switch_calls[0]["extra"] == {}
    assert provider.switch_calls[1]["extra"] == {"rewound": True}


def test_flush_pending_shares_one_deadline_with_provider_drain():
    started = threading.Event()
    release = threading.Event()

    class _DrainingProvider(_RecordingProvider):
        def __init__(self):
            super().__init__()
            self.drain_timeouts = []

        def sync_turn(self, *args, **kwargs):  # type: ignore[override]
            started.set()
            assert release.wait(timeout=1.0)

        def drain_pending(self, *, timeout=None):
            self.drain_timeouts.append(timeout)
            return True

    mm = MemoryManager()
    provider = _DrainingProvider()
    mm.add_provider(provider)
    mm.sync_all("u", "a")
    assert started.wait(timeout=1.0)
    threading.Timer(0.08, release.set).start()

    assert mm.flush_pending(timeout=0.3)
    assert len(provider.drain_timeouts) == 1
    assert 0.0 <= provider.drain_timeouts[0] < 0.27


@pytest.mark.parametrize("outcome", [False, RuntimeError("drain failed")])
def test_flush_pending_reports_provider_drain_failure(outcome):
    class _DrainingProvider(_RecordingProvider):
        def drain_pending(self, *, timeout=None):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    mm = MemoryManager()
    mm.add_provider(_DrainingProvider())

    assert mm.flush_pending(timeout=0.2) is False




# ---------------------------------------------------------------------------
# MemoryManager.sync_all / queue_prefetch_all — session_id propagation
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Hindsight reference implementation — state-flush semantics
# ---------------------------------------------------------------------------


def _make_hindsight_provider():
    """Build a bare HindsightMemoryProvider that skips network setup.

    We instantiate without importing optional deps at class-level by
    bypassing __init__ and seeding the attributes on_session_switch
    reads/writes. This keeps the test hermetic.
    """
    import threading
    hindsight_mod = pytest.importorskip("plugins.memory.hindsight")
    provider = object.__new__(hindsight_mod.HindsightMemoryProvider)
    provider._session_id = "old-sid"
    provider._parent_session_id = ""
    provider._document_id = "old-sid-20260101_000000_000000"
    provider._session_turns = ["turn-1", "turn-2"]
    provider._turn_counter = 2
    provider._turn_index = 2
    # Attrs read by _build_metadata / _build_retain_kwargs when the
    # buffer-flush path on session switch fires. Empty strings keep the
    # metadata minimal but well-formed.
    provider._retain_source = ""
    provider._platform = ""
    provider._user_id = ""
    provider._user_name = ""
    provider._chat_id = ""
    provider._chat_name = ""
    provider._chat_type = ""
    provider._thread_id = ""
    provider._agent_identity = ""
    provider._agent_workspace = ""
    provider._retain_tags = []
    provider._retain_context = "test-context"
    provider._retain_async = False
    provider._bank_id = "test-bank"
    provider._bank_id_template = ""
    provider._bank_id_fallback = "test-bank"
    provider._observation_scopes = None
    provider._observation_scope_exclude_tag_prefixes = []
    # Prefetch state the switch path drains/clears.
    provider._prefetch_thread = None
    provider._prefetch_lock = threading.Lock()
    provider._prefetch_result = ""
    provider._prefetch_count = 0
    provider._prefetch_generation = 0
    provider._prefetch_request_token = 0
    provider._prefetch_result_generation = -1
    provider._prefetch_result_session_id = ""
    provider._prefetch_result_request_token = -1
    # Sync thread tracking (legacy alias at the writer).
    provider._sync_thread = None
    # Writer queue infra the flush-on-switch path enqueues onto. We stub
    # _ensure_writer / _register_atexit so no real thread is spawned;
    # tests exercising flush delivery live in
    # tests/plugins/memory/test_hindsight_provider.py where the full
    # writer-queue wiring is in place.
    import queue as _queue
    provider._retain_queue = _queue.Queue()
    provider._shutting_down = threading.Event()
    provider._delivery_lock = threading.RLock()
    provider._active_delivery_state = None
    provider._delivery_states = []
    provider._last_retained_turn_count = 0
    provider._queued_retained_turn_count = 0
    provider._atexit_registered = True
    provider._ensure_writer = lambda: None
    provider._register_atexit = lambda: None
    # Mode + API state used by _resolve_retain_target; stub the resolver
    # so tests don't actually probe the API. Real probe behavior is
    # exercised by tests in tests/plugins/memory/test_hindsight_provider.py.
    provider._mode = "cloud"
    provider._api_url = ""
    provider._api_key = ""
    provider._client = None
    provider._resolve_retain_target = lambda fb: (fb, None)
    # Stub the network-touching helper so any enqueued flush closure is
    # a no-op if ever drained in a unit test.
    provider._run_hindsight_operation = lambda _op: None
    return provider


def test_hindsight_on_session_switch_updates_session_id_and_mints_fresh_doc():
    provider = _make_hindsight_provider()
    old_doc = provider._document_id

    provider.on_session_switch(
        "new-sid", parent_session_id="old-sid", reset=False, reason="resume"
    )

    assert provider._session_id == "new-sid"
    assert provider._parent_session_id == "old-sid"
    # Document id MUST be fresh — else next retain overwrites old session doc
    assert provider._document_id != old_doc
    assert provider._document_id.startswith("new-sid-")


def test_hindsight_on_session_switch_clears_turn_buffers():
    """Accumulated _session_turns must not leak into the next session.

    Hindsight batches turns under a single _document_id. If the buffer
    isn't cleared on switch, the next retain under the new _document_id
    flushes turns that belong to the previous session.
    """
    provider = _make_hindsight_provider()
    provider.on_session_switch("new-sid", parent_session_id="old-sid")
    assert provider._session_turns == []
    assert provider._turn_counter == 0
    assert provider._turn_index == 0



