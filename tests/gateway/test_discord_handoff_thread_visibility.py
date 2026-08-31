"""Regression tests for visible Discord handoff threads.

A directly-created, unanchored Discord thread may accept the cron brief while
producing no starter message in the parent channel. The user then never sees the
thread. Handoffs must be created from a visible parent seed message instead.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter_with_parent(parent):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=parent),
        fetch_channel=AsyncMock(),
    )
    return adapter


@pytest.mark.asyncio
async def test_handoff_thread_is_anchored_to_visible_parent_message():
    thread = SimpleNamespace(id=9001)
    seed_message = SimpleNamespace(
        create_thread=AsyncMock(return_value=thread),
    )
    parent = SimpleNamespace(
        send=AsyncMock(return_value=seed_message),
        create_thread=AsyncMock(),
    )
    adapter = _adapter_with_parent(parent)

    thread_id = await adapter.create_handoff_thread(
        "123",
        "Hermes — JD evening executive closeout",
        auto_archive_duration=1440,
    )

    assert thread_id == "9001"
    parent.send.assert_awaited_once_with(
        "🧵 **Hermes — JD evening executive closeout**"
    )
    seed_message.create_thread.assert_awaited_once_with(
        name="Hermes — JD evening executive closeout",
        auto_archive_duration=1440,
        reason="Hermes session handoff",
    )
    parent.create_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_handoff_does_not_fall_back_to_invisible_direct_thread(caplog):
    parent = SimpleNamespace(
        send=AsyncMock(side_effect=RuntimeError("cannot post seed")),
        create_thread=AsyncMock(return_value=SimpleNamespace(id=9002)),
    )
    adapter = _adapter_with_parent(parent)

    with caplog.at_level("WARNING"):
        thread_id = await adapter.create_handoff_thread("123", "Daily brief")

    assert thread_id is None
    parent.create_thread.assert_not_awaited()
    assert "visible seed/thread creation failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_handoff_thread_failure_cleans_up_parent_seed(cleanup_fails):
    seed = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("thread rejected")),
        delete=AsyncMock(side_effect=RuntimeError("cleanup rejected") if cleanup_fails else None),
    )
    parent = SimpleNamespace(send=AsyncMock(return_value=seed), create_thread=AsyncMock())
    adapter = _adapter_with_parent(parent)

    assert await adapter.create_handoff_thread("123", "Daily brief") is None
    seed.delete.assert_awaited_once()
    parent.create_thread.assert_not_awaited()
