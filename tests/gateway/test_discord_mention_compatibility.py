"""Native free-response inheritance cannot replace the parent's legacy exemption."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway import test_discord_free_response as free_response_cases
from tests.gateway.test_discord_free_response import (
    FakeTextChannel, FakeThread, make_message,
)

adapter = free_response_cases.adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["free_response_channels", "thread_free_response_channels"])
@pytest.mark.parametrize("thread_requires_mention", [False, True])
@pytest.mark.parametrize("mentioned", [False, True])
async def test_existing_thread_keeps_legacy_mention_boundary(
    adapter, setting, thread_requires_mention, mentioned,
):
    adapter.config.extra.update({
        "require_mention": True, "thread_require_mention": thread_requires_mention,
        setting: ["789"], "free_response_auto_thread": True,
    })
    parent = FakeTextChannel(channel_id=789)
    thread = FakeThread(channel_id=9001, parent=parent)
    adapter._threads.mark("9001")
    message = make_message(
        channel=thread, content="hello",
        mentions=[adapter._client.user] if mentioned else [],
    )
    await adapter._handle_message(message)
    admitted = setting == "free_response_channels" or not thread_requires_mention or mentioned
    assert adapter.handle_message.await_count == int(admitted)
    if admitted:
        assert adapter.handle_message.await_args.args[0].source.thread_id == "9001"


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["free_response_channels", "thread_free_response_channels"])
@pytest.mark.parametrize("no_thread", [False, True])
async def test_native_and_legacy_parent_auto_thread_respect_exclusions(adapter, setting, no_thread):
    adapter.config.extra.update({
        "require_mention": True, "auto_thread": True, "free_response_auto_thread": True,
        setting: ["789"], "no_thread_channels": ["789"] if no_thread else [],
    })
    adapter._auto_create_thread = AsyncMock(return_value=SimpleNamespace(id=9001, name="task"))
    await adapter._handle_message(make_message(channel=FakeTextChannel(channel_id=789), content="new task"))
    adapter.handle_message.assert_awaited_once()
    assert adapter._auto_create_thread.await_count == int(not no_thread)
    event = adapter.handle_message.await_args.args[0]
    assert event.source.thread_id == (None if no_thread else "9001")
