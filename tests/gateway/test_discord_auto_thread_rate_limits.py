"""Regression tests for Discord auto-thread rate-limit isolation.

A long-running slash-command reconciliation must not change the shared
HTTP client's rate-limit policy used by live message traffic. Auto-thread
creation must wait/retry a real 429 on the direct path and must never leave
or duplicate a success-looking fallback seed when no thread was created.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
import plugins.platforms.discord.adapter as discord_platform


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Too many requests. Retry in {retry_after:.2f} seconds.")


@pytest.fixture
def adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))


def _thread(thread_id: int = 555):
    return SimpleNamespace(id=thread_id, name="thread")


def _message(*, direct_side_effect, channel):
    return SimpleNamespace(
        content="OO-ABO flies to IAD",
        author=SimpleNamespace(display_name="Tom"),
        channel=channel,
        create_thread=AsyncMock(side_effect=direct_side_effect),
    )


@pytest.mark.asyncio
async def test_slash_sync_does_not_mutate_shared_http_ratelimit_policy(adapter):
    """Maintenance sync must not poison concurrent live Discord requests."""
    http = SimpleNamespace(max_ratelimit_timeout=None)
    adapter._client = SimpleNamespace(
        application_id=123,
        user=SimpleNamespace(id=123),
        http=http,
    )
    adapter._get_discord_command_sync_policy = lambda: "safe"
    adapter._desired_command_sync_fingerprint = lambda: "fingerprint"
    adapter._command_sync_skip_reason = lambda _app, _fp: None
    adapter._record_command_sync_attempt = lambda _app, _fp: None
    adapter._record_command_sync_success = lambda _app, _fp, _summary: None

    async def safe_sync():
        # This assertion runs while the sync coroutine is active, which is the
        # exact concurrency window that broke live auto-thread creation.
        assert http.max_ratelimit_timeout is None
        return {
            "total": 1,
            "unchanged": 1,
            "updated": 0,
            "recreated": 0,
            "created": 0,
            "deleted": 0,
        }

    adapter._safe_sync_slash_commands = safe_sync

    await adapter._run_post_connect_initialization()

    assert http.max_ratelimit_timeout is None


@pytest.mark.asyncio
async def test_auto_thread_waits_exact_retry_after_and_retries_direct_only(adapter, monkeypatch):
    """A Discord 429 is a wait signal, not a reason to branch into a seed."""
    thread = _thread()
    channel = SimpleNamespace(send=AsyncMock())
    message = _message(
        direct_side_effect=[RateLimited(161.81), thread],
        channel=channel,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    result = await adapter._auto_create_thread(message)

    assert result is thread
    assert thread._hermes_auto_thread_initial_name == adapter._derive_auto_thread_name(message.content)
    sleep.assert_awaited_once_with(161.81)
    assert message.create_thread.await_count == 2
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_thread_second_rate_limit_fails_without_seed(adapter, monkeypatch):
    """The fallback shares the bucket, so a persistent 429 must not post it."""
    channel = SimpleNamespace(send=AsyncMock())
    message = _message(
        direct_side_effect=[RateLimited(4.0), RateLimited(3.0)],
        channel=channel,
    )
    monkeypatch.setattr(discord_platform.asyncio, "sleep", AsyncMock())

    result = await adapter._auto_create_thread(message)

    assert result is None
    assert message.create_thread.await_count == 2
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_thread_uses_at_most_one_fallback_seed(adapter, monkeypatch):
    """Two failed direct attempts may create one fallback seed, never two."""
    thread = _thread()
    seed = SimpleNamespace(
        create_thread=AsyncMock(return_value=thread),
        delete=AsyncMock(),
    )
    channel = SimpleNamespace(send=AsyncMock(return_value=seed))
    message = _message(
        direct_side_effect=[RuntimeError("connect failed"), RuntimeError("connect failed")],
        channel=channel,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    result = await adapter._auto_create_thread(message)

    assert result is thread
    assert thread._hermes_auto_thread_initial_name == adapter._derive_auto_thread_name(message.content)
    sleep.assert_awaited_once_with(0.75)
    channel.send.assert_awaited_once()
    seed.create_thread.assert_awaited_once()
    seed.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_thread_deletes_single_orphan_seed_on_fallback_failure(adapter, monkeypatch):
    """No success-looking seed may survive when fallback creation fails."""
    seed = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("fallback failed")),
        delete=AsyncMock(),
    )
    channel = SimpleNamespace(send=AsyncMock(return_value=seed))
    message = _message(
        direct_side_effect=[RuntimeError("connect failed"), RuntimeError("connect failed")],
        channel=channel,
    )
    monkeypatch.setattr(discord_platform.asyncio, "sleep", AsyncMock())

    result = await adapter._auto_create_thread(message)

    assert result is None
    channel.send.assert_awaited_once()
    seed.create_thread.assert_awaited_once()
    seed.delete.assert_awaited_once()
