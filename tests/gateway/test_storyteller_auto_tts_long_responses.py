"""Regression tests for Storyteller/Discord auto-TTS narration.

These cover the dedicated narration profile use case where a long assistant
reply may be split into multiple Discord messages, but should still produce one
coherent TTS generation from the full pre-split response text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
)


class _DiscordLikeNarrationAdapter(BasePlatformAdapter):
    """Small adapter that mimics Discord text splitting for focused tests."""

    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, response: str):
        super().__init__(PlatformConfig(enabled=True, extra={}), Platform.DISCORD)
        self._message_handler = AsyncMock(return_value=response)
        self.sent_text_chunks: list[str] = []
        self.played_audio_paths: list[str] = []
        self._auto_tts_default = True
        self._auto_tts_default_mode = "all"

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        self.sent_text_chunks.extend(
            self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        )
        return SendResult(success=True, message_id=f"msg-{len(self.sent_text_chunks)}")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Storyteller", "type": "channel"}

    async def _keep_typing(self, *args, **kwargs) -> None:
        await asyncio.Event().wait()

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        self.played_audio_paths.append(audio_path)
        return SendResult(success=True, message_id="audio-1")


def _event(chat_id: str = "123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=Platform.DISCORD,
    )
    event = MessageEvent(text="tell me what happens", message_type=MessageType.TEXT, source=source)
    event.message_id = "incoming-1"
    return event


@pytest.mark.asyncio
async def test_long_discord_split_response_gets_one_full_cued_audio_file(tmp_path, monkeypatch):
    """Long text split into Discord chunks should still be spoken once, in full.

    The important regression checks are:
    - TTS gets the pre-split response once, not one call per Discord text chunk.
    - ElevenLabs-style bracket cues survive the TTS preparation step.
    - The TTS call is not hard-truncated at four thousand chars by the gateway;
      provider/model limits are enforced inside ``text_to_speech_tool``.
    """
    tail_marker = "FINAL-SENTENCE-MARKER"
    response = "[whispers] " + ("stone corridor " * 360) + tail_marker
    assert len(response) > 4000

    adapter = _DiscordLikeNarrationAdapter(response)
    captured_tts_texts: list[str] = []
    audio_path = tmp_path / "storyteller-full-narration.mp3"
    audio_path.write_bytes(b"mp3")

    def fake_tts(*, text: str, output_path: Optional[str] = None) -> str:
        captured_tts_texts.append(text)
        return json.dumps({"success": True, "file_path": str(audio_path)})

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    monkeypatch.setattr("os.remove", lambda _path: None)

    await adapter._process_message_background(_event(), "discord:123:user1")

    assert len(adapter.sent_text_chunks) > 1
    assert len(adapter.played_audio_paths) == 1
    assert len(captured_tts_texts) == 1
    spoken_text = captured_tts_texts[0]
    assert spoken_text.startswith("[whispers]")
    assert tail_marker in spoken_text


@pytest.mark.asyncio
async def test_streamed_voice_reply_uses_full_text_and_preserves_elevenlabs_cues(tmp_path):
    """The runner-owned streamed TTS path must not pre-truncate or strip cues."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}

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
         patch("os.path.isfile", return_value=True), \
         patch("os.unlink"), \
         patch("os.makedirs"):
        await runner._send_voice_reply(event, response)

    mock_adapter.send_voice.assert_called_once()
    assert len(captured_tts_texts) == 1
    spoken_text = captured_tts_texts[0]
    assert spoken_text.startswith("[sighs]")
    assert tail_marker in spoken_text
