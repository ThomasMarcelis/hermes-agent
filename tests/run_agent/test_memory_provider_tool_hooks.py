"""Coverage for plugin hooks around memory-provider tools in run_agent."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call(name: str = "hindsight_recall", arguments: str = '{"query": "jd memory"}', call_id: str = "call_mem"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture()
def agent(monkeypatch):
    hermes_home = Path(os.environ["HERMES_HOME"])
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.fetch_model_metadata", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        monkeypatch.setattr("run_agent._hermes_home", hermes_home)
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        a.client = MagicMock()
        return a


def test_invoke_tool_pre_hook_receives_session_and_tool_call_id(monkeypatch):
    agent = object.__new__(AIAgent)
    agent.session_id = "session-123"
    seen = {}

    def _block(tool_name, args, **kwargs):
        seen["tool_name"] = tool_name
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "blocked"

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_block_message", _block)

    result = AIAgent._invoke_tool(
        agent,
        "hindsight_retain",
        {"content": "store this"},
        "task-1",
        tool_call_id="call-123",
    )

    assert json.loads(result) == {"error": "blocked"}
    assert seen["tool_name"] == "hindsight_retain"
    assert seen["kwargs"]["session_id"] == "session-123"
    assert seen["kwargs"]["tool_call_id"] == "call-123"


def test_invoke_tool_memory_provider_path_emits_post_tool_call(monkeypatch):
    agent = object.__new__(AIAgent)
    agent.session_id = "session-123"
    manager = MagicMock()
    manager.has_tool.return_value = True
    manager.handle_tool_call.return_value = '{"result":"ok"}'
    agent._memory_manager = manager
    hook_calls = []

    def _invoke_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)

    result = AIAgent._invoke_tool(
        agent,
        "hindsight_recall",
        {"query": "jd"},
        "task-1",
        tool_call_id="call-123",
        pre_tool_block_checked=True,
    )

    assert result == '{"result":"ok"}'
    manager.handle_tool_call.assert_called_once_with("hindsight_recall", {"query": "jd"})
    assert hook_calls
    name, kwargs = hook_calls[0]
    assert name == "post_tool_call"
    assert kwargs["tool_name"] == "hindsight_recall"
    assert kwargs["session_id"] == "session-123"
    assert kwargs["tool_call_id"] == "call-123"
    assert kwargs["duration_ms"] >= 0


def test_sequential_memory_provider_tool_pre_and_post_hooks(agent, monkeypatch):
    manager = MagicMock()
    manager.has_tool.side_effect = lambda name: name == "hindsight_recall"
    manager.handle_tool_call.return_value = '{"result":"ok"}'
    agent._memory_manager = manager
    pre_calls = []
    post_calls = []

    def _pre(tool_name, args, **kwargs):
        pre_calls.append((tool_name, args, kwargs))
        return None

    def _post(name, **kwargs):
        if name == "post_tool_call":
            post_calls.append(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_block_message", _pre)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _post)

    assistant_message = SimpleNamespace(tool_calls=[_tool_call()])
    messages = []
    agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert pre_calls[0][0] == "hindsight_recall"
    assert pre_calls[0][2]["session_id"] == agent.session_id
    assert pre_calls[0][2]["tool_call_id"] == "call_mem"
    assert post_calls
    assert post_calls[0]["tool_name"] == "hindsight_recall"
    assert post_calls[0]["session_id"] == agent.session_id
    assert post_calls[0]["tool_call_id"] == "call_mem"
    assert json.loads(messages[0]["content"]) == {"result": "ok"}
