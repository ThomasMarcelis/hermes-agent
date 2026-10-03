"""Native session messages: real discovery, profile ownership, lifetime and admission."""
import os
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
from hermes_cli import session_messages as routes
from hermes_cli.cli_session_messages import (
    close_cli_session_route, register_cli_session_route, unwrap_cli_session_message,
)
from hermes_cli.plugins import get_plugin_manager
from hermes_constants import get_hermes_home
from tools.terminal_scope import terminal_env


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    homes = [tmp_path / ".hermes", tmp_path / ".hermes" / "profiles" / "other"]
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    previous_multiplex = is_multiplex_active()
    loaded = []
    for index, home in enumerate(homes):
        plugin = home / "plugins" / "session-notify-test"
        plugin.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"terminal:\n  cwd: {home}\nplugins:\n  enabled: [session-notify-test]\n"
            "  entries:\n    session-notify-test:\n      allow_gateway_injection: true\n")
        (home / ".env").write_text(f"SESSION_TEST_SECRET=profile-{index}\n")
        (plugin / "plugin.yaml").write_text("name: session-notify-test\nversion: 0.1.0\ndescription: test\n")
        (plugin / "__init__.py").write_text(
            "from hermes_constants import get_hermes_home\n"
            "context = None\nclosed = []\n"
            "def register(ctx):\n"
            "    global context\n    context = ctx\n"
            "    ctx.register_hook('on_session_message_route_closed', on_close)\n"
            "def on_close(*, route_id):\n"
            "    closed.append((route_id, str(get_hermes_home())))\n")
        with routes.profile_message_scope(home):
            manager = get_plugin_manager()
            manager.discover_and_load()
            loaded.append(manager._plugins["session-notify-test"].module)
    set_multiplex_active(True)
    yield list(zip(homes, loaded))
    for home, _ in zip(homes, loaded):
        for route in list(routes._OWNERS.values()):
            if route.home == str(home):
                routes.close_session_message_route(home, route.owner)
    set_multiplex_active(previous_multiplex)


def test_real_plugin_context_routes_callbacks_and_permissions_in_owner_profile(profiles):
    observed = []
    env_before = dict(os.environ)
    for home, module in profiles:
        def receive(*, session_id, content, message_id, busy_mode):
            observed.append((str(get_hermes_home()), get_secret("SESSION_TEST_SECRET"), terminal_env("TERMINAL_CWD"), content))
            return {"accepted": True, "status": "queued"}
        with routes.profile_message_scope(home):
            routes.register_session_message_route(home, "same-id", receive)
    # Invoke A's context while B is active (and vice versa), as background bridge callbacks do.
    for index in (0, 1, 0):
        home, module = profiles[index]
        with routes.profile_message_scope(profiles[1 - index][0]):
            assert module.context.inject_session_message("same-id", f"message {index}", message_id=f"{index}-{len(observed)}") == {
                "accepted": True, "status": "queued"}
    assert [item[:3] for item in observed] == [
        (str(profiles[index][0]), f"profile-{index}", str(profiles[index][0])) for index in (0, 1, 0)]
    assert dict(os.environ) == env_before
    denied_home, denied = profiles[0]
    (denied_home / "config.yaml").write_text("plugins:\n  entries:\n    session-notify-test:\n      allow_gateway_injection: false\n")
    with routes.profile_message_scope(profiles[1][0]):
        assert denied.context.inject_session_message("same-id", "blocked", message_id="denied")["status"] == "denied"
    assert len(observed) == 3


def test_compression_alias_duplicates_and_stale_owner_cleanup(profiles):
    home, module = profiles[0]
    calls = []
    def receive(**message):
        calls.append(message)
        return {"accepted": True, "status": "started"}
    owner = routes.register_session_message_route(home, "parent", receive)
    assert routes.alias_session_route(home, "parent", "child")
    assert module.context.session_message_route("parent") == {"route_id": owner, "session_id": "child"}
    assert module.context.inject_session_message("parent", "hello", message_id="same")["status"] == "started"
    assert module.context.inject_session_message("child", "hello", message_id="same")["status"] == "duplicate"
    assert len(calls) == 1 and calls[0]["session_id"] == "child"
    with routes.profile_message_scope(profiles[1][0]):
        assert routes.close_session_message_route(home, owner)
    assert module.closed == [(owner, str(home))]
    assert module.context.inject_session_message("parent", "late", message_id="late")["status"] == "stale_session"
    replacement = routes.register_session_message_route(home, "child", receive)
    assert not routes.close_session_message_route(home, owner)
    assert module.context.session_message_route("child")["route_id"] == replacement
    # An old inbox callback may already have read its session ID when the same
    # durable session is reopened. Its owner token must fence that delayed reply.
    assert module.context.inject_session_message(
        "child", "old inbox", message_id="replacement-race", expected_route_id=owner,
    ) == {"accepted": False, "status": "stale_session"}
    assert len(calls) == 1
    assert module.context.inject_session_message(
        "child", "current inbox", message_id="replacement-race", expected_route_id=replacement,
    )["status"] == "started"


def test_failed_delivery_can_retry_and_inflight_id_is_not_claimed_delivered(profiles):
    home, module = profiles[0]
    entered, release = threading.Event(), threading.Event()
    def blocked(**message):
        entered.set()
        assert release.wait(5)
        return {"accepted": False, "status": "offline"}
    owner = routes.register_session_message_route(home, "session", blocked)
    results = []
    worker = threading.Thread(target=lambda: results.append(module.context.inject_session_message("session", "reply", message_id="id")))
    worker.start()
    assert entered.wait(5)
    assert not module.context.inject_session_message("session", "reply", message_id="id")["accepted"]
    release.set()
    worker.join(5)
    assert not results[0]["accepted"]
    routes.register_session_message_route(home, "session", lambda **kw: {"accepted": True, "status": "queued"}, owner=owner)
    assert module.context.inject_session_message("session", "reply", message_id="id")["status"] == "queued"


def test_classic_cli_queues_literal_messages_and_fences_conversation_replacement(profiles):
    home, module = profiles[0]
    cli = SimpleNamespace(session_id="cli-one", _pending_input=Queue(), _agent_running=False, _should_exit=False, agent=None)
    with routes.profile_message_scope(home):
        register_cli_session_route(cli)
    assert module.context.inject_session_message("cli-one", "/new", message_id="queued")["status"] == "queued"
    pending = cli._pending_input.get_nowait()
    assert unwrap_cli_session_message(cli, pending) == ("/new", True)
    old_owner = cli._plugin_message_owner
    cli.session_id = "cli-two"
    with routes.profile_message_scope(home):
        register_cli_session_route(cli)
    assert unwrap_cli_session_message(cli, pending) == (None, True)
    assert module.context.inject_session_message("cli-one", "late", message_id="late")["status"] == "stale_session"
    assert cli._plugin_message_owner != old_owner
    cli._agent_running = True
    cli.agent = SimpleNamespace(steer_session_message=lambda text: True)
    assert module.context.inject_session_message("cli-two", "busy", message_id="busy")["status"] == "steered"
    cli.agent.steer_session_message = lambda text: False
    assert module.context.inject_session_message("cli-two", "finalized", message_id="finalized")["status"] == "queued"
    close_cli_session_route(cli)
    assert module.context.session_message_route("cli-two") is None


@pytest.mark.parametrize("quiet", [False, True])
def test_one_shot_consumes_every_accepted_reply_before_retiring_route(profiles, monkeypatch, quiet):
    import cli as facade
    from agent.interrupt_control import InterruptControlMixin
    from hermes_cli.cli_session_messages import SessionMessageInput
    from hermes_cli import quiet_single_query
    home, module = profiles[0]
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(facade, "_should_seed_interactive", lambda *a: False)
    monkeypatch.setattr(facade, "_collect_query_images", lambda query, image: (query, []))
    monkeypatch.setattr(facade, "_collect_kanban_task_images", lambda images: [])
    monkeypatch.setattr(facade, "_route_single_query_images", lambda cli, query, effective, *args: effective)
    monkeypatch.setattr(facade, "_configure_quiet_agent", lambda agent: None)
    monkeypatch.setattr(facade, "_finalize_single_query", lambda cli: None)
    monkeypatch.setattr(quiet_single_query, "continue_quiet_notify_completions", lambda *args, **kw: None)
    seen = []

    class Agent(InterruptControlMixin):
        session_id = "oneshot"
        _pending_steer = None
        _pending_steer_lock = threading.Lock()
        _session_message_steer_active = False

        def run_conversation(self, *, user_message, conversation_history, **kwargs):
            seen.append(user_message)
            result = {"final_response": "ok", "completed": True, "messages": [
                *conversation_history, {"role": "user", "content": user_message}, {"role": "assistant", "content": "ok"}]}
            if len(seen) == 1:
                self._session_message_steer_active = True
                assert module.context.inject_session_message("oneshot", "steered peer", message_id="steered")["status"] == "steered"
                self._session_message_steer_active = False
                result["pending_steer"] = self._drain_pending_steer()
                assert module.context.inject_session_message("oneshot", "queued peer", message_id="queued")["status"] == "queued"
            return result

    agent = Agent()
    runtime = SimpleNamespace(
        session_id="oneshot", agent=agent, conversation_history=[], model="test",
        _pending_input=Queue(), _agent_running=False, _should_exit=False,
        _active_agent_route_signature="route", _claim_active_session=lambda *a, **kw: True,
        _ensure_runtime_credentials=lambda: True, _resolve_turn_agent_config=lambda query: {
            "signature": "route", "model": "test", "runtime": {}, "request_overrides": {}},
        _init_agent=lambda **kw: True, _show_security_advisories=lambda: None,
        _print_exit_summary=lambda **kw: None, console=SimpleNamespace(print=lambda *a: None),
    )

    def chat(text, **kwargs):
        result = agent.run_conversation(user_message=text, conversation_history=runtime.conversation_history)
        runtime.conversation_history = result["messages"]
        runtime._last_turn_result = result
        if result.get("pending_steer"):
            runtime._pending_input.put(SessionMessageInput(runtime._plugin_message_owner, result["pending_steer"]))
        return result["final_response"]
    runtime.chat = chat
    with routes.profile_message_scope(home), pytest.raises(SystemExit) as exc:
        facade._run_single_query_mode(runtime, "original", None, quiet, True)
    assert exc.value.code == 0
    assert len(seen) == 3
    assert any("steered peer" in text for text in seen)
    assert "queued peer" in seen
    assert runtime._pending_input.empty()
    assert module.context.session_message_route("oneshot") is None
    assert not module.context.inject_session_message("oneshot", "after exit", message_id="closed")["accepted"]
