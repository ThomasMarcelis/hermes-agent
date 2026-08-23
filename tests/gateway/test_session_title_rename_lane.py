"""Which title stage is allowed to spend a platform rename.

Titling is two-stage: a derived slice of the user's own words lands inline, and
the model's version replaces it a moment later. A local sidebar wants both. A
Discord thread or a Telegram topic wants only the second — renaming twice lands
on the same name at twice the cost, and Discord allows two channel renames per
ten minutes, so the throwaway can be the one that survives.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, TurnRunner
from gateway.session import SessionSource


def _attach(lane):
    """Attach the title callback for *lane* and return (callback, renames)."""
    renames: list = []
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="chan-1")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda src: lane == "telegram",
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_telegram_topic_title_rename=(
            lambda src, sid, title: renames.append(title)
        ),
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append(title)
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=source)
    )
    return agent._on_session_title, renames


@pytest.mark.parametrize("lane", ["telegram", "discord"])
def test_the_rename_waits_for_the_model_title(lane):
    callback, renames = _attach(lane)

    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


@pytest.mark.asyncio
async def test_native_discord_auto_thread_rename_uses_native_signature(
    monkeypatch,
):
    renames = []

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    async def rename_thread(
        thread_id, name, *, only_if_current_name=None
    ):
        renames.append((thread_id, name, only_if_current_name))
        return True

    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        Platform.DISCORD: types.SimpleNamespace(rename_thread=rename_thread)
    }
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="999",
        chat_type="thread",
        user_id="user-1",
        thread_id="999",
        parent_chat_id="100",
        auto_thread_created=True,
        auto_thread_initial_name="Hermes",
    )

    await runner._rename_discord_auto_thread_for_session_title(
        source, "sess-1", "Voice note follow-up"
    )

    assert renames == [("999", "Voice note follow-up", "Hermes")]
