"""Explicitly detached delegation prints a later-resume notice; joined results remain silent.

With delegation.top_level_completion=detach the tool returns a handle and the result
re-enters as a new turn. The default joined result has no dispatched/background markers.
"""

import json

import cli
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._pending_edit_snapshots = {}
    return cli_obj


def _capture(monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr(cli, "_cprint", lambda text: printed.append(text))
    return printed


def test_background_dispatch_prints_resume_notice(monkeypatch):
    cli_obj = _make_cli()
    printed = _capture(monkeypatch)

    result = json.dumps({"status": "dispatched", "mode": "background", "count": 1})
    cli_obj._on_tool_complete("tc1", "delegate_task", {"goal": "x"}, result)

    joined = "\n".join(printed)
    assert "resume" in joined.lower()
    assert "it finishes" in joined


def test_background_batch_dispatch_pluralizes(monkeypatch):
    cli_obj = _make_cli()
    printed = _capture(monkeypatch)

    result = json.dumps({"status": "dispatched", "mode": "background", "count": 3})
    cli_obj._on_tool_complete("tc2", "delegate_task", {"tasks": []}, result)

    joined = "\n".join(printed)
    assert "3 tasks" in joined
    assert "they finish" in joined


def test_synchronous_delegate_result_prints_no_notice(monkeypatch):
    """Joined results, including the default root policy, must not claim a later dispatch."""
    cli_obj = _make_cli()
    printed = _capture(monkeypatch)

    result = json.dumps({"results": [{"status": "completed", "summary": "done"}]})
    cli_obj._on_tool_complete("tc3", "delegate_task", {"goal": "x"}, result)

    assert not any("resume" in p.lower() for p in printed)


def test_non_delegate_tool_prints_no_notice(monkeypatch):
    cli_obj = _make_cli()
    printed = _capture(monkeypatch)

    cli_obj._on_tool_complete("tc4", "read_file", {"path": "a"}, '{"ok": true}')

    assert not any("resume" in p.lower() for p in printed)
