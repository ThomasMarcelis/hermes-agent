"""Finalization, SQLite/live history and rewind share the same memory identities."""

from __future__ import annotations

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MEMORY_LIFECYCLE_API_VERSION, MemoryProvider
from agent.session_persistence import SessionPersistenceMixin
from agent.turn_finalizer import finalize_turn
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


class _RecordingProvider(MemoryProvider):
    name = "recording"

    def __init__(self):
        self.synced = []
        self.rewound = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self.synced.append((assistant_content, messages[-1].get("message_uid")))

    def on_session_switch(self, new_session_id, **kwargs):
        self.rewound.append((new_session_id, kwargs["rewound_message_uids"]))


class _LiveAgent(SessionPersistenceMixin, FakeAgent):
    """Real persistence and memory sync; the fixture supplies unrelated finalizer setup."""

    _sync_external_memory_for_turn = AIAgent._sync_external_memory_for_turn

    def __init__(self, db, provider):
        super().__init__()
        self._session_db = db
        self._session_db_created = True
        self._last_flushed_db_idx = 0
        self.save_trajectories = False
        self._memory_manager = MemoryManager()
        self._memory_manager.add_provider(provider)


def _finish(agent, history, user, outcome):
    before = list(history)
    history.append({"role": "user", "content": user})
    response = "" if outcome == "empty" else f"answer to {user}"
    if outcome == "interrupted":
        history.append({"role": "assistant", "content": response})
    result = finalize_turn(
        agent, final_response=response, api_call_count=1,
        interrupted=outcome == "interrupted", failed=False,
        messages=history, conversation_history=before,
        effective_task_id="task", turn_id=f"turn-{user}",
        user_message=user, original_user_message=user,
        _should_review_memory=False, _turn_exit_reason="text_response(final)",
    )
    assert not result.get("cleanup_errors"), result.get("cleanup_errors")
    assert agent._memory_manager.flush_pending(timeout=5)
    return result["messages"]


def _rewind(agent, history, surface, action):
    if surface == "cli":
        from hermes_cli.cli_session_mixin import CLISessionMixin

        cli = CLISessionMixin.__new__(CLISessionMixin)
        cli.agent, cli.session_id = agent, agent.session_id
        cli._session_db, cli.conversation_history = agent._session_db, history
        if action == "retry":
            assert cli.retry_last() == "second"
        else:
            assert cli.undo_last(prefill=False) == 1
        return cli.conversation_history

    from agent.context_compressor import user_originated_turn_view
    from tui_gateway.server import _rewind_active_session_history

    db = agent._session_db
    session = {"agent": agent, "history": history,
               "session_key": agent.session_id if db is not None else ""}
    if db is not None:
        from pathlib import Path

        session["profile_home"] = str(Path(db.db_path).parent)
    ordinal = sum(user_originated_turn_view(m) is not None for m in history) - 1
    prefix, live_view, _ = _rewind_active_session_history(
        session, ordinal, require_retryable=action == "retry")
    if action == "retry":
        assert live_view["content"] == "second"
    return prefix


@pytest.mark.parametrize("storage", ["memory", "sqlite", "resumed"])
@pytest.mark.parametrize("surface", ["cli", "tui"])
@pytest.mark.parametrize("action", ["undo", "retry"])
@pytest.mark.parametrize("outcome", ["completed", "interrupted", "empty"])
def test_live_finalization_and_rewind_preserve_retained_identity(
    tmp_path, monkeypatch, storage, surface, action, outcome,
):
    """Skipping the second turn must not revoke the first; real completed turns remain addressable."""
    assert MEMORY_LIFECYCLE_API_VERSION >= 1
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db") if storage != "memory" else None
    if db is not None:
        db.create_session("sess-test", source="cli")
    provider = _RecordingProvider()
    agent = _LiveAgent(db, provider)
    try:
        history = _finish(agent, [], "first", "completed")
        first_uid = history[-1].get("message_uid")
        assert first_uid, "completed live assistant needs an identity without SQLite"
        assert provider.synced == [("answer to first", first_uid)]
        agent._persist_session(history)
        assert history[-1]["message_uid"] == first_uid

        if storage == "resumed":
            agent._memory_manager.shutdown_all()
            history = db.get_resume_conversations(agent.session_id)[0]
            assert history[-1]["message_uid"] == first_uid
            provider = _RecordingProvider()
            agent = _LiveAgent(db, provider)

        sync_count = len(provider.synced)
        history = _finish(agent, history, "second", outcome)
        completed = outcome == "completed"
        second_uid = history[-1].get("message_uid") if completed else None
        assert len(provider.synced) == sync_count + int(completed)
        if completed:
            assert second_uid and second_uid != first_uid
            assert provider.synced[-1] == ("answer to second", second_uid)
        if db is not None:
            stored = db.get_messages_as_conversation(agent.session_id)
            assert [m.get("message_uid") for m in stored] == [m.get("message_uid") for m in history]

        history = _rewind(agent, history, surface, action)
        assert agent._memory_manager.flush_pending(timeout=5)
        assert len(provider.rewound) == 1
        owner, removed = provider.rewound[-1]
        assert owner == agent.session_id
        assert first_uid not in removed
        if completed:
            assert second_uid in removed
        if storage == "memory" and not completed:
            assert removed == (), "known empty identity sets are not turn counts"
        assert history[-1]["message_uid"] == first_uid

        if action == "retry":
            history = _finish(agent, history, "second", "completed")
            retried_uid = history[-1]["message_uid"]
            assert retried_uid not in (first_uid, second_uid)
            history = _rewind(agent, history, surface, "undo")
            assert agent._memory_manager.flush_pending(timeout=5)
            assert retried_uid in provider.rewound[-1][1]
            assert first_uid not in provider.rewound[-1][1]

        assert _rewind(agent, history, surface, "undo") == []
        assert agent._memory_manager.flush_pending(timeout=5)
        assert first_uid in provider.rewound[-1][1]
    finally:
        agent._memory_manager.shutdown_all()
        if db is not None:
            db.close()
