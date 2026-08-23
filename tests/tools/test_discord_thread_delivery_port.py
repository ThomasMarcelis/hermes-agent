"""Standalone Discord thread-per-run and send_message thread mirroring tests."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from plugins.platforms.discord.adapter import _standalone_send
from tests.tools.test_send_message_tool import (
    _patch_discord_sender,
    _run_async_immediately,
)
from tools.send_message_tool import _send_to_platform, send_message_tool


def test_send_message_created_thread_is_mirrored_to_thread_session():
    config = SimpleNamespace(
        platforms={Platform.DISCORD: SimpleNamespace(enabled=True, token="***", extra={})},
        get_home_channel=lambda _platform: None,
    )
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.send_message_tool._send_to_platform",
             new=AsyncMock(return_value={"success": True, "thread_id": "thread-1"}),
         ), \
         patch("gateway.mirror.mirror_to_session", return_value=True) as mirror:
        result = json.loads(send_message_tool({
            "action": "send", "target": "discord:123456789", "message": "done",
        }))

    assert result["mirrored"] is True
    assert mirror.call_args.args[:2] == ("discord", "thread-1")
    assert mirror.call_args.kwargs["thread_id"] == "thread-1"


def test_requested_name_creates_thread_then_sends_starter_content():
    thread_response = MagicMock()
    thread_response.status = 201
    thread_response.json = AsyncMock(return_value={"id": "thread-123"})
    thread_response.__aenter__ = AsyncMock(return_value=thread_response)
    thread_response.__aexit__ = AsyncMock(return_value=None)
    message_response = MagicMock()
    message_response.status = 200
    message_response.json = AsyncMock(return_value={"id": "message-456"})
    message_response.__aenter__ = AsyncMock(return_value=message_response)
    message_response.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(side_effect=[thread_response, message_response])

    with patch("aiohttp.ClientSession", return_value=session), patch(
        "gateway.channel_directory.lookup_channel_type", return_value="channel"
    ):
        result = asyncio.run(_standalone_send(
            SimpleNamespace(token="tok", extra={}),
            "ch1",
            "Cron output",
            thread_name="Daily report",
            thread_auto_archive_duration=60,
        ))

    assert result["thread_id"] == "thread-123"
    first, second = session.post.call_args_list
    assert first.args[0].endswith("/channels/ch1/threads")
    assert first.kwargs["json"] == {
        "name": "Daily report", "type": 11, "auto_archive_duration": 60,
    }
    assert second.args[0].endswith("/channels/thread-123/messages")


def test_thread_per_run_is_created_once_for_chunked_delivery():
    sender = AsyncMock(side_effect=[
        {"success": True, "message_id": "1", "thread_id": "thread-1"},
        {"success": True, "message_id": "2", "thread_id": "thread-1"},
    ])
    with _patch_discord_sender(sender):
        result = asyncio.run(_send_to_platform(
            Platform.DISCORD,
            SimpleNamespace(enabled=True, token="tok", extra={}),
            "parent",
            "A" * 2500,
            discord_thread_name="Daily report",
            discord_thread_auto_archive_duration=60,
        ))

    assert result["success"] is True
    first, second = sender.await_args_list
    assert first.kwargs["thread_id"] is None
    assert first.kwargs["thread_name"] == "Daily report"
    assert first.kwargs["thread_auto_archive_duration"] == 60
    assert second.kwargs["thread_id"] == "thread-1"
    assert "thread_name" not in second.kwargs
