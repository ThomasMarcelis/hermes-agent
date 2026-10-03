"""Real plugin registry → gateway admission, including profile and conversation boundaries."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_session_messages import (
    alias_session_message_route,
    close_session_message_routes,
    ensure_session_message_route,
    session_message_event_is_current,
)
from gateway.session import SessionSource, SessionStore
from hermes_cli.session_messages import (
    get_session_message_route,
    inject_session_message,
    profile_message_scope,
)
from hermes_constants import get_hermes_home


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.admitted = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("No network calls in admission tests")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    def _start_session_processing(self, event, session_key):
        # Exercise BasePlatformAdapter.handle_message's actual routing/admission gates,
        # then stop at the model-turn boundary.
        self.admitted.append((event, get_hermes_home()))
        return True


def _make_runner(home):
    home.mkdir()
    with profile_message_scope(home):
        store = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
        entry = store.get_or_create_session(source)
        adapter = Adapter()
        adapter.set_message_handler(AsyncMock())
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.session_store = store
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._profile_adapters = {}
        runner._running = True
        runner._draining = False
        runner._running_agents = {}
        runner._queued_events = {}
        runner._is_user_authorized = lambda source, **kwargs: True
        runner._profile_scope_for_source = lambda source: nullcontext()
        ensure_session_message_route(runner, source, entry.session_key, entry.session_id)
    return runner, adapter, entry


async def _inject(home, session_id, message_id="msg1", **kwargs):
    return await asyncio.to_thread(
        inject_session_message, home, session_id, kwargs.pop("content", "peer reply"),
        message_id=message_id, **kwargs,
    )


@pytest.mark.asyncio
async def test_idle_exact_session_receipt_and_dedup(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    try:
        receipt = await _inject(home, entry.session_id, content="/approve always")
        assert receipt == {"accepted": True, "status": "started"}
        event, owning_home = adapter.admitted[0]
        assert owning_home == home
        assert event.source.chat_id == entry.origin.chat_id
        assert event.internal is True and event.allow_gateway_control is False
        assert event.get_command() is None
        assert event.metadata["gateway_session_id"] == entry.session_id
        assert await _inject(home, entry.session_id) == {"accepted": True, "status": "duplicate"}
        assert len(adapter.admitted) == 1
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_busy_safe_steer_and_fallback_preserve_human_queue(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    key = entry.session_key
    agent = SimpleNamespace(steer_session_message=MagicMock(return_value=True))
    runner._running_agents[key] = agent
    human = MessageEvent(text="human follow-up", message_type=MessageType.TEXT, source=entry.origin)
    adapter._pending_messages[key] = human
    try:
        assert (await _inject(home, entry.session_id))["status"] == "steered"
        agent.steer_session_message.assert_called_once_with("peer reply")
        agent.steer_session_message.return_value = False
        assert (await _inject(home, entry.session_id, "msg2"))["status"] == "queued"
        assert adapter._pending_messages[key] is human
        queued = runner._overflow_queue(key)
        assert len(queued) == 1 and queued[0].text == "peer reply"
        assert queued[0].allow_gateway_control is False
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_final_steer_survives_alongside_human_followup(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    key = entry.session_key
    human = MessageEvent(text="human follow-up", message_type=MessageType.TEXT, source=entry.origin)
    adapter._pending_messages[key] = human
    try:
        result = {"final_response": "done", "pending_steer": "late peer reply"}
        next_event, text = await runner._run_agent_drain_pending(result, adapter, entry.origin, key)
        assert next_event is human and text == "human follow-up"
        peer = adapter._pending_messages[key]
        assert peer.text == "late peer reply" and peer.internal
        assert session_message_event_is_current(runner, peer, entry)
        assert "pending_steer" not in result
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_compression_and_cache_eviction_keep_route_reset_retires_it(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    original = entry.session_id
    key = entry.session_key
    try:
        owner = get_session_message_route(home, original)["route_id"]
        runner._agent_cache = {key: {}}
        runner._evict_cached_agent(key)
        assert get_session_message_route(home, original)["route_id"] == owner
        entry.session_id = "compressed-child"
        alias_session_message_route(runner, key, entry.session_id)
        with profile_message_scope(home):
            ensure_session_message_route(runner, entry.origin, key, entry.session_id)
        assert get_session_message_route(home, original) == {
            "route_id": owner, "session_id": entry.session_id,
        }
        assert (await _inject(home, original))["accepted"] is True
        delivered = adapter.admitted[-1][0]
        assert delivered.metadata["gateway_session_id"] == entry.session_id
        runner._clear_conversation_scope(key, reason="reset")
        assert get_session_message_route(home, original) is None
        assert (await _inject(home, original, "msg2"))["accepted"] is False
        assert session_message_event_is_current(runner, delivered, entry) is False
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_queued_message_follows_compression_but_never_replacement(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    runner._running_agents[entry.session_key] = SimpleNamespace()
    try:
        await _inject(home, entry.session_id)
        event = adapter._pending_messages[entry.session_key]
        entry.session_id = "child"
        alias_session_message_route(runner, entry.session_key, entry.session_id)
        assert session_message_event_is_current(runner, event, entry)
        assert event.metadata["gateway_session_id"] == "child"
        entry.session_id = "unrelated-conversation"
        assert not session_message_event_is_current(runner, event, entry)
        assert (await _inject(home, "child", "msg2"))["accepted"] is False
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_two_real_homes_a_b_a_never_cross_deliver(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    lanes = [_make_runner(home) for home in homes]
    try:
        for index in (0, 1, 0):
            runner, adapter, entry = lanes[index]
            receipt = await _inject(homes[index], entry.session_id, f"msg-{index}-{len(adapter.admitted)}")
            assert receipt["accepted"] is True
            assert adapter.admitted[-1][1] == homes[index]
        assert [len(lane[1].admitted) for lane in lanes] == [2, 1]
        # Some gateway session ids are timestamp-derived and can coincide across
        # homes. Their private route owners must still differ.
        assert get_session_message_route(homes[0], lanes[0][2].session_id)["route_id"] != (
            get_session_message_route(homes[1], lanes[1][2].session_id)["route_id"])
        lanes[0][2].session_id = "a-only-continuation"
        alias_session_message_route(lanes[0][0], lanes[0][2].session_key, lanes[0][2].session_id)
        assert (await _inject(homes[1], "a-only-continuation", "wrong-profile"))["accepted"] is False
    finally:
        for runner, _, _ in lanes:
            close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_gateway_rejects_queue_overflow_disconnected_and_denied(tmp_path):
    home = tmp_path / "profile"
    runner, adapter, entry = _make_runner(home)
    runner._running_agents[entry.session_key] = SimpleNamespace()
    try:
        for n in range(runner._BUSY_QUEUE_MAX_PENDING):
            assert (await _inject(home, entry.session_id, f"msg{n}"))["accepted"] is True
        assert (await _inject(home, entry.session_id, "overflow"))["accepted"] is False
        assert runner._queue_depth(entry.session_key, adapter=adapter) == runner._BUSY_QUEUE_MAX_PENDING
        runner._is_user_authorized = lambda source, **kwargs: False
        assert (await _inject(home, entry.session_id, "denied"))["status"] == "denied"
        runner._is_user_authorized = lambda source, **kwargs: True
        runner.adapters.clear()
        assert (await _inject(home, entry.session_id, "disconnected"))["accepted"] is False
    finally:
        close_session_message_routes(runner)


@pytest.mark.asyncio
async def test_one_multiplex_gateway_keeps_runtime_and_transport_homes(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.session_identity import resolve_identity

    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    runner, primary, first = _make_runner(home)
    runner.config.multiplex_profiles = True
    runner.session_store.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    del runner._profile_scope_for_source  # exercise the real runtime profile scope
    secondary_home = home / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    (home / ".env").write_text("SESSION_ROUTE_TEST_SECRET=primary\n")
    (secondary_home / ".env").write_text("SESSION_ROUTE_TEST_SECRET=secondary\n")
    secondary = Adapter()
    secondary.gateway_runner = runner
    secondary.set_owner_profile("secondary")
    secondary.set_message_handler(AsyncMock())
    runner._profile_adapters = {"secondary": {Platform.TELEGRAM: secondary}}
    source = secondary.build_source(chat_id="43", chat_type="dm", user_id="43")
    resolve_identity(source, runner=runner, adapter=secondary, transport_profile="secondary")
    with runner._profile_scope_for_source(source):
        second = runner.session_store.get_or_create_session(source)
        ensure_session_message_route(runner, source, second.session_key, second.session_id)
    observed = []

    def authorized(source, **kwargs):
        observed.append((get_hermes_home(), secret_scope.get_secret("SESSION_ROUTE_TEST_SECRET")))
        return True

    runner._is_user_authorized = authorized
    try:
        for target_home, entry in ((home, first), (secondary_home, second), (home, first)):
            assert (await _inject(target_home, entry.session_id, f"msg{len(observed)}"))["accepted"]
        assert observed == [(home, "primary"), (secondary_home, "secondary"), (home, "primary")]
        assert [scope for _, scope in primary.admitted] == [home, home]
        assert [scope for _, scope in secondary.admitted] == [secondary_home]
    finally:
        close_session_message_routes(runner)
