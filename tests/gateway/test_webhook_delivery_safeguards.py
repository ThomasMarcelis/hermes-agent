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
