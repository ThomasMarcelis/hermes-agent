"""Config-owned top-level joining through the real model and registry dispatch paths."""

import json
import threading
from unittest.mock import MagicMock

import pytest


def _profile(tmp_path, monkeypatch, settings):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (home / "config.yaml").write_text(json.dumps({"delegation": settings}))
    return home


@pytest.mark.parametrize("settings,detached", [
    ({}, False), ({"top_level_completion": None}, False),
    ({"top_level_completion": ""}, False), ({"top_level_completion": "  JoIn  "}, False),
    ({"top_level_completion": "  DeTaCh  "}, True), ({"top_level_completion": "typo"}, False),
])
@pytest.mark.parametrize("depth", [0, 1])
def test_profile_policy_controls_every_model_dispatch(settings, detached, depth, tmp_path, monkeypatch, caplog):
    import tools.delegate_tool as dt
    from run_agent import AIAgent
    from tools.registry import registry

    _profile(tmp_path, monkeypatch, settings)
    parent = MagicMock()
    parent._delegate_depth = depth
    calls = []

    def delegate(**kwargs):
        calls.append(kwargs)
        return "{}"

    monkeypatch.setattr(dt, "delegate_task", delegate)
    for background in (True, False):
        arguments = {"tasks": [{"goal": "Check lifecycle ownership"}], "background": background}
        AIAgent._dispatch_delegate_task(parent, arguments)
        registry.dispatch("delegate_task", arguments, parent_agent=parent)
    assert len(calls) == 4
    assert all(call["background"] is (detached and depth == 0) for call in calls)
    if settings.get("top_level_completion") == "typo" and depth == 0:
        assert "is invalid" in caplog.text


def test_shipped_loader_defaults_agree_on_join(tmp_path, monkeypatch):
    _profile(tmp_path, monkeypatch, {})
    from cli import load_cli_config
    from hermes_cli.config import load_config_readonly
    from tools.delegate_tool_config import _get_top_level_completion_mode

    assert load_cli_config()["delegation"]["top_level_completion"] == "join"
    assert load_config_readonly()["delegation"]["top_level_completion"] == "join"
    assert _get_top_level_completion_mode() == "join"


@pytest.mark.parametrize("mode", ["join", "detach"])
def test_model_schema_describes_the_configured_completion_contract(mode, tmp_path, monkeypatch):
    _profile(tmp_path, monkeypatch, {"top_level_completion": mode})
    import tools.delegate_tool  # noqa: F401 -- register real handler/schema
    from tools.registry import registry

    definition = registry.get_definitions({"delegate_task"})[0]["function"]
    description = definition["description"]
    assert len(description) <= 2200
    assert "background" not in definition["parameters"]["properties"]
    if mode == "join":
        assert "Runs joined" in description
        assert "terminal outcome" in description and "one final answer" in description
        assert "END YOUR TURN" not in description
    else:
        assert "Runs detached" in description
        assert "independent_completions" in description and "group" in description
        assert "wait or poll" in description and "BETWEEN your turns" in description


def test_root_join_waits_for_parallel_terminal_outcomes(tmp_path, monkeypatch):
    """A joined model call cannot finalize early or serialize its independent child workers."""
    import tools.delegate_tool as dt
    from run_agent import AIAgent

    _profile(tmp_path, monkeypatch, {"independent_completions": True})
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.session_id = "owned-join-test"
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    returned = threading.Event()
    result = {}
    failures = []

    def build(**kwargs):
        child = MagicMock()
        child._delegate_role = "leaf"
        child._subagent_id = f"child-{kwargs['task_index']}"
        return child

    def run_child(task_index, goal, **kwargs):
        started[task_index].set()
        assert release.wait(timeout=10)
        status = "completed" if task_index == 0 else "error"
        return {"task_index": task_index, "status": status, "summary": goal,
                "api_calls": 0, "duration_seconds": 0.01, "model": "test-model",
                "exit_reason": status, "error": "child failed" if task_index else None}

    monkeypatch.setattr(dt, "_build_child_agent", build)
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *args, **kwargs: {
        "model": "test-model", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    })

    def dispatch():
        try:
            result.update(json.loads(AIAgent._dispatch_delegate_task(parent, {
                "tasks": [{"goal": "Independent first worker", "group": "pair"},
                          {"goal": "Independent second worker"}],
                "background": True,
            })))
        except BaseException as exc:
            failures.append(exc)
        finally:
            returned.set()

    worker = threading.Thread(target=dispatch, daemon=True)
    worker.start()
    try:
        assert all(event.wait(timeout=5) for event in started)
        assert not returned.is_set()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and not failures
    assert [entry["task_index"] for entry in result["results"]] == [0, 1]
    assert [entry["status"] for entry in result["results"]] == ["completed", "error"]
    assert "mode" not in result and "delegation_id" not in result
