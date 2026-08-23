"""Focused regression tests for generic webhook delivery safeguards."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import _INSECURE_NO_AUTH
from tests.gateway.test_webhook_adapter import _create_app, _make_adapter


class TestWebhookIdempotencyNamespace:
    @pytest.mark.asyncio
    async def test_provider_id_is_namespaced_by_route(self):
        adapter = _make_adapter(routes={
            "first": {"secret": _INSECURE_NO_AUTH, "prompt": "first"},
            "second": {"secret": _INSECURE_NO_AUTH, "prompt": "second"},
        })
        adapter.handle_message = AsyncMock()
        headers = {"X-GitHub-Delivery": "provider-reused-id"}

        async with TestClient(TestServer(_create_app(adapter))) as client:
            first = await client.post("/webhooks/first", json={"a": 1}, headers=headers)
            second = await client.post("/webhooks/second", json={"a": 1}, headers=headers)

        assert first.status == 202
        assert second.status == 202
        assert adapter.handle_message.await_count == 2

    def test_provider_id_is_namespaced_by_profile(self):
        adapter = _make_adapter()
        now = time.time()
        default_key = adapter._delivery_claim_key("default", "route", "provider-id")
        worker_key = adapter._delivery_claim_key("worker", "route", "provider-id")
        assert adapter._record_delivery_id(default_key, now) is True
        assert adapter._record_delivery_id(worker_key, now) is True

    @pytest.mark.asyncio
    async def test_failed_direct_delivery_releases_idempotency_claim(self):
        adapter = _make_adapter(routes={
            "direct": {
                "secret": _INSECURE_NO_AUTH,
                "deliver_only": True,
                "deliver": "discord",
                "deliver_extra": {"chat_id": "target-channel"},
            }
        })
        target = MagicMock()
        target.send = AsyncMock(return_value=SendResult(False, error="down"))
        runner = MagicMock()
        runner._active_profile_name.return_value = "default"
        runner.adapters = {Platform.DISCORD: target}
        runner._profile_adapters = {}
        adapter.gateway_runner = runner  # type: ignore[assignment]
        headers = {"X-GitHub-Delivery": "retryable-delivery"}

        async with TestClient(TestServer(_create_app(adapter))) as client:
            first = await client.post("/webhooks/direct", json={"a": 1}, headers=headers)
            retry = await client.post("/webhooks/direct", json={"a": 1}, headers=headers)

        assert first.status == 502
        assert retry.status == 502
        assert target.send.await_count == 2


@pytest.mark.asyncio
async def test_cross_platform_delivery_is_strictly_profile_scoped():
    adapter = _make_adapter()
    default_target = MagicMock()
    default_target.send = AsyncMock(return_value=SendResult(True))
    worker_target = MagicMock()
    worker_target.send = AsyncMock(return_value=SendResult(True))
    runner = MagicMock()
    runner._active_profile_name.return_value = "default"
    runner.adapters = {Platform.DISCORD: default_target}
    runner._profile_adapters = {"worker": {Platform.DISCORD: worker_target}}
    adapter.gateway_runner = runner  # type: ignore[assignment]

    result = await adapter._deliver_cross_platform(
        "discord",
        "profile-owned delivery",
        {
            "deliver_extra": {"chat_id": "worker-channel"},
            "profile": "worker",
            "route": "route-a",
            "delivery_id": "delivery-a",
        },
    )
    assert result.success is True
    worker_target.send.assert_awaited_once()
    default_target.send.assert_not_awaited()

    runner._profile_adapters["worker"].clear()
    result = await adapter._deliver_cross_platform(
        "discord",
        "must not leak",
        {"deliver_extra": {"chat_id": "worker-channel"}, "profile": "worker"},
    )
    assert result.success is False
    default_target.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("on_missing_cursor", ["raise", "fallback"])
async def test_webhook_delivery_sends_exactly_one_final_response(on_missing_cursor):
    from gateway.config import StreamingConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    adapter = _make_adapter()
    target = MagicMock()
    target.send = AsyncMock(return_value=SendResult(True))
    delivery_runner = MagicMock()
    delivery_runner._active_profile_name.return_value = "default"
    delivery_runner.adapters = {Platform.DISCORD: target}
    delivery_runner._profile_adapters = {}
    adapter.gateway_runner = delivery_runner  # type: ignore[assignment]
    chat_id = "webhook:generic:delivery-1"
    adapter._delivery_info[chat_id] = {
        "deliver": "discord",
        "deliver_extra": {"chat_id": "final-channel"},
        "profile": "default",
    }

    runner = object.__new__(GatewayRunner)
    with pytest.raises(RuntimeError, match="final-only"):
        runner._build_stream_consumer_config(
            SessionSource(platform=Platform.WEBHOOK, chat_id=chat_id),
            StreamingConfig(enabled=True),
            adapter,
            on_missing_cursor=on_missing_cursor,
        )

    result = await adapter.send(chat_id, "First partial. Final tail.")
    assert result.success is True
    target.send.assert_awaited_once_with(
        "final-channel", "First partial. Final tail.", metadata=None
    )


@pytest.mark.asyncio
async def test_final_only_adapter_disables_all_progress_and_status_wiring(monkeypatch):
    import asyncio
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    adapter = _make_adapter()
    adapter.supports_status_text = True
    adapter.send = AsyncMock(return_value=SendResult(True))
    source = SessionSource(platform=Platform.WEBHOOK, chat_id="webhook:route:run")
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _: adapter
    runner._resolve_turn_toolsets = lambda *_: ([], [])
    runner.hooks = MagicMock()
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "display": {"tool_progress": "all", "thinking_progress": True,
                    "interim_assistant_messages": True, "live_status": "full",
                    "long_running_notifications": True},
    })
    display = runner._run_agent_display_settings(source)
    assert not display.tool_progress_enabled
    assert not display.interim_assistant_messages_enabled
    assert not display._thinking_enabled
    assert not display.needs_progress_queue
    assert display._live_status_adapter is None

    context = TurnContext(source=source, _run_still_current=lambda: True)
    turn = TurnRunner(runner, context)
    runner._run_agent_bind_turn_wiring(context, turn, source, None, False)
    assert context._status_adapter is None
    context._status_callback_sync("compression", "Compacting context")
    await asyncio.wait_for(
        runner._run_agent_notify_long_running(display, context, [None]), timeout=2,
    )
    await asyncio.sleep(0)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_profile_route_keeps_delivery_and_retry_in_owning_profile(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    isolated_home = tmp_path / "home"
    default_home = isolated_home / ".hermes"
    worker_home = default_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: isolated_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    adapter = _make_adapter(routes={"worker-events": {
        "profile": "worker", "secret": _INSECURE_NO_AUTH, "deliver_only": True,
        "deliver": "discord", "deliver_extra": {"chat_id": "worker-channel"},
        "prompt": "Worker event",
    }})
    default_target = SimpleNamespace(send=AsyncMock(return_value=SendResult(True)))
    worker_target = SimpleNamespace(send=AsyncMock(side_effect=[RuntimeError("offline"), SendResult(True)]))
    adapter.gateway_runner = SimpleNamespace(
        _active_profile_name=lambda: "default", adapters={Platform.DISCORD: default_target},
        _profile_adapters={"worker": {Platform.DISCORD: worker_target}},
        config=SimpleNamespace(multiplex_profiles=True, multiplex_profile_allowlist=["worker"]),
    )
    app = _create_app(adapter)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    headers = {"X-Request-ID": "retry-event"}
    async with TestClient(TestServer(app)) as client:
        wrong = await client.post("/webhooks/worker-events", json={}, headers=headers)
        assert wrong.status == 404
        failed = await client.post("/p/worker/webhooks/worker-events", json={}, headers=headers)
        assert failed.status == 502
        retry = await client.post("/p/worker/webhooks/worker-events", json={}, headers=headers)
        assert (await retry.json())["status"] == "delivered"
        duplicate = await client.post("/p/worker/webhooks/worker-events", json={}, headers=headers)
        assert (await duplicate.json())["status"] == "duplicate"
    assert worker_target.send.await_count == 2
    assert all(call.args[0] == "worker-channel" for call in worker_target.send.await_args_list)
    default_target.send.assert_not_awaited()
