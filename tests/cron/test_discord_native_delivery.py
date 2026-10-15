"""Discover Discord, deliver through real cron orchestration, then continue its session.

Only transport I/O and the sync-to-async bridge are faked; configuration, plugin
loading, handoff creation, delivery targeting and SQLite history are production.
"""
import asyncio
from concurrent.futures import Future
from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import platform_registry
from gateway.session import SessionSource, SessionStore
from hermes_cli.plugins import discover_plugins


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(yaml.safe_dump({
        "cron": {"wrap_response": False},
        "platforms": {"discord": {"enabled": True, "token": "test-token"}},
        "discord": {"thread_free_response_channels": ["123"]},
    }))
    discover_plugins(force=True)
    entry = platform_registry.get("discord")
    assert entry is not None
    config = load_gateway_config()
    adapter = entry.adapter_factory(config.platforms[Platform.DISCORD])
    assert adapter._discord_thread_free_response_channels() == {"123"}
    store = SessionStore(profile / "sessions", config)
    adapter._session_store = store
    thread = SimpleNamespace(id=9001, send=AsyncMock(return_value=SimpleNamespace(id=9002)))
    seed = SimpleNamespace(id=7001, create_thread=AsyncMock(return_value=thread), delete=AsyncMock())
    parent = SimpleNamespace(send=AsyncMock(return_value=seed), create_thread=AsyncMock())
    adapter._client = SimpleNamespace(
        get_channel=lambda cid: {123: parent, 9001: thread}.get(cid),
        user=SimpleNamespace(id=999),
    )

    def run_now(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", run_now)
    monkeypatch.setattr("cron.scheduler._hermes_now", lambda: datetime(2026, 10, 15, 9, 30, tzinfo=UTC))
    yield SimpleNamespace(adapter=adapter, parent=parent, seed=seed, thread=thread, store=store,
                          loop=SimpleNamespace(is_running=lambda: True))
    store.close_all_db_handles()


def job(**options):
    return {"id": "daily", "name": "Daily brief", "deliver": "origin",
            "origin": {"platform": "discord", "chat_id": "123"}, **options}


@pytest.mark.parametrize("legacy", [False, True])
def test_native_continuation_has_visible_starter_and_replyable_history(delivery, legacy):
    options = {"attach_to_session": True}
    if legacy:
        options["delivery_options"] = {"discord": {
            "thread_per_run": True, "thread_name_template": "{job_name} {date} {time}",
            "thread_auto_archive_duration": 60,
        }}
    result = _deliver_result(job(**options), "A complete daily brief.",
                             adapters={Platform.DISCORD: delivery.adapter}, loop=delivery.loop)
    assert result is None
    delivery.parent.send.assert_awaited_once()
    delivery.parent.create_thread.assert_not_awaited()
    delivery.seed.create_thread.assert_awaited_once()
    kwargs = delivery.seed.create_thread.await_args.kwargs
    assert kwargs["auto_archive_duration"] == (60 if legacy else 1440)
    assert kwargs["name"] == ("Daily brief 2026-10-15 09:30" if legacy else "Hermes — Daily brief")
    assert "Daily brief" in delivery.parent.send.await_args.args[0]
    delivery.thread.send.assert_awaited_once()
    reply = SessionSource(platform=Platform.DISCORD, chat_id="9001", chat_type="thread",
                          thread_id="9001", user_id="42")
    session = delivery.store.get_or_create_session(reply)
    history = delivery.store.load_transcript(session.session_id)
    assert any(m["role"] == "user" and "A complete daily brief." in m["content"] for m in history)


def test_legacy_thread_option_does_not_enable_history_retention(delivery):
    result = _deliver_result(job(discord_thread_per_run=True, attach_to_session=False), "No retained copy.",
                             adapters={Platform.DISCORD: delivery.adapter}, loop=delivery.loop)
    assert result is None
    delivery.seed.create_thread.assert_awaited_once()
    delivery.thread.send.assert_awaited_once()
    reply = SessionSource(platform=Platform.DISCORD, chat_id="9001", chat_type="thread",
                          thread_id="9001", user_id="42")
    session = delivery.store.get_or_create_session(reply)
    assert delivery.store.load_transcript(session.session_id) == []


def test_native_thread_failure_delivers_flat_without_unanchored_thread(delivery):
    delivery.seed.create_thread.side_effect = RuntimeError("thread permission denied")
    result = _deliver_result(job(attach_to_session=True), "Fallback brief.",
                             adapters={Platform.DISCORD: delivery.adapter}, loop=delivery.loop)
    assert result is None
    assert delivery.parent.send.await_count == 2
    assert "Fallback brief." in delivery.parent.send.await_args.kwargs["content"]
    delivery.seed.delete.assert_awaited_once()
    delivery.parent.create_thread.assert_not_awaited()
    delivery.thread.send.assert_not_awaited()


def test_standalone_cron_uses_discovered_sender_and_reuses_thread_across_chunks(delivery, monkeypatch):
    def response(payload, status=200):
        result = MagicMock(status=status)
        result.json = AsyncMock(return_value=payload)
        result.__aenter__ = AsyncMock(return_value=result)
        result.__aexit__ = AsyncMock(return_value=None)
        return result

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post.side_effect = [response({"id": "7001"}), response({"id": "9001"}, 201),
                                *[response({"id": str(mid)}) for mid in range(9002, 9005)]]
    monkeypatch.setattr("aiohttp.ClientSession", lambda **_kw: session)
    monkeypatch.setattr("gateway.channel_directory.lookup_channel_type", lambda *_a: "channel")
    result = _deliver_result(job(discord_thread_per_run=True,
                                 discord_thread_title_template="{job_name} {date}",
                                 discord_thread_auto_archive_duration=60), "A" * 4500)
    assert result is None
    seed, thread, *chunks = session.post.call_args_list
    assert seed.args[0].endswith("/channels/123/messages")
    assert seed.kwargs["json"]["content"] == "🧵 **Daily brief 2026-10-15**"
    assert thread.args[0].endswith("/channels/123/messages/7001/threads")
    assert thread.kwargs["json"] == {"name": "Daily brief 2026-10-15", "auto_archive_duration": 60}
    assert len(chunks) == 3
    assert all(call.args[0].endswith("/channels/9001/messages") for call in chunks)
    contents = [call.kwargs["json"]["content"] for call in chunks]
    for index, content in enumerate(contents, 1):
        assert len(content) <= 2000
        assert content.endswith(f" ({index}/3)")
    assert "".join(content.removesuffix(f" ({index}/3)")
                   for index, content in enumerate(contents, 1)) == "A" * 4500
