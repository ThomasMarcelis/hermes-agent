"""Regression tests for Storyteller/Discord auto-TTS narration."""

from __future__ import annotations

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _event(chat_id: str = "123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=Platform.DISCORD,
    )
    event = MessageEvent(text="tell me what happens", message_type=MessageType.TEXT, source=source)
    event.message_id = "incoming-1"
    return event


def test_streamed_voice_reply_uses_full_text_and_preserves_elevenlabs_cues(tmp_path):
    """Gateway-owned streamed TTS must not pre-truncate Storyteller narration."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._reply_anchor_for_event = MagicMock(return_value="incoming-1")
    runner._thread_metadata_for_source = MagicMock(return_value=None)

    event = _event()
    mock_adapter = MagicMock()
    mock_adapter.send_voice = AsyncMock()
    runner.adapters[Platform.DISCORD] = mock_adapter

    tail_marker = "STREAMED-FINAL-MARKER"
    response = "[sighs] " + ("wind over black water " * 230) + tail_marker
    assert len(response) > 4000

    captured_tts_texts: list[str] = []
    tts_path = tmp_path / "streamed-full-narration.mp3"
    tts_path.write_bytes(b"mp3")

    def fake_tts(*, text: str, output_path: Optional[str] = None) -> str:
        captured_tts_texts.append(text)
        return json.dumps({"success": True, "file_path": str(tts_path)})

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts), \
         patch("tools.tts_tool._should_preserve_audio_tags_for_tts", return_value=True), \
         patch("tools.tts_tool._should_preserve_voice_markup_for_tts", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("os.unlink"), \
         patch("os.makedirs"):
        asyncio.run(runner._send_voice_reply(event, response))

    mock_adapter.send_voice.assert_awaited_once()
    assert len(captured_tts_texts) == 1
    spoken_text = captured_tts_texts[0]
    assert spoken_text.startswith("[sighs]")
    assert tail_marker in spoken_text
