"""External steering cannot land behind the final drain; early exits preserve accepted text."""
import threading

from agent.interrupt_control import InterruptControlMixin


class Agent(InterruptControlMixin):
    def __init__(self):
        self._pending_steer_lock = threading.Lock()
        self._pending_steer = None


def test_external_steer_admission_is_closed_without_a_turn():
    agent = Agent()
    assert not agent.steer_session_message("early")
    agent._session_message_steer_active = True
    assert agent.steer_session_message("during")
    with agent._pending_steer_lock:
        agent._session_message_steer_active = False
    assert "during" in agent._drain_pending_steer()
    assert not agent.steer_session_message("too late")
    assert agent._drain_pending_steer() is None


def test_early_turn_result_hands_back_accepted_external_steer(monkeypatch):
    from agent import conversation_loop
    agent = Agent()
    def run(agent, *args, **kwargs):
        assert agent.steer_session_message("not consumed")
        return {"messages": [], "completed": False}
    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", run)
    result = conversation_loop.run_conversation(agent, "hello")
    assert "not consumed" in result["pending_steer"]
    assert "EXTERNAL SESSION MESSAGE" in result["pending_steer"]
    assert not agent.steer_session_message("too late")


def test_mixed_human_and_external_steer_preserves_provenance_after_requeue(caplog):
    from agent.agent_runtime_helpers import _requeue_pending_steer, apply_pending_steer_to_tool_results
    from agent.prompt_builder import STEER_MARKER_OPEN, STEER_MARKER_CLOSE, steer_user_row
    agent = Agent()
    agent._session_message_steer_active = True
    agent.steer("human before")
    agent.steer_session_message("private peer payload")
    drained = agent._drain_pending_steer()
    agent.steer("human after")
    _requeue_pending_steer(agent, drained)
    messages = [{"role": "tool", "content": "result", "tool_call_id": "call"}]
    with caplog.at_level("INFO"):
        apply_pending_steer_to_tool_results(agent, messages, 1)
    content = messages[-1]["content"]
    assert "EXTERNAL SESSION MESSAGE" in content
    assert "private peer payload" in content
    for human in ("human before", "human after"):
        assert STEER_MARKER_OPEN + "\n" + human + "\n" + STEER_MARKER_CLOSE in content
    assert STEER_MARKER_OPEN + "\nprivate peer payload" not in content
    assert "private peer payload" not in caplog.text
    agent.steer_session_message("peer only")
    peer_row = steer_user_row(agent._drain_pending_steer())
    assert STEER_MARKER_OPEN not in peer_row["content"]
    assert "peer only" in peer_row["content"]
