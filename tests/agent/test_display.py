"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

import agent.display as display_module
from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    prepare_tool_preview,
    redact_tool_args_for_display,
    set_tool_preview_max_len,
    truncate_tool_preview,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


@pytest.fixture(autouse=True)
def reset_tool_preview_max_len():
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


def test_cute_tool_message_falls_back_when_renderer_raises(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("cosmetic failure")

    monkeypatch.setattr(display_module, "_get_cute_tool_message", _boom)

    assert get_cute_tool_message("web_extract", {"urls": []}, 0.25) == (
        "┊ ⚡ web_extra completed  0.2s"
    )


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None








    def test_browser_type_preview_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": secret})
        assert result is not None
        assert secret not in result
        assert "sk-pro" in result and "..." in result

    def test_browser_type_preview_keeps_normal_text(self):
        text = "hello world search query"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": text})
        assert result is not None
        assert text in result

    def test_browser_type_display_args_redact_api_key(self):
        secret = "ghp_ABCDEFGHIJ1234567890"
        safe_args = redact_tool_args_for_display(
            "browser_type", {"ref": "@e3", "text": secret}
        )
        assert secret not in str(safe_args)
        assert safe_args["ref"] == "@e3"
        assert safe_args["text"].startswith("ghp_AB")















    def test_delegate_task_batch_preview_respects_max_len(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": "A" * 80}, {"goal": "B" * 80}]},
            max_len=30,
        )
        assert result == "2 tasks: AAAAAAAAAAAAAAAAAA..."
        assert len(result) == 30

    def test_delegate_task_batch_preview_has_no_hidden_per_goal_cap(self):
        first_goal = "Trace every Discord preview layer without dropping context"
        second_goal = "Verify the complete gateway rendering path with tests"

        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": first_goal}, {"goal": second_goal}]},
            max_len=0,
        )

        assert result == f"2 tasks: {first_goal} | {second_goal}"

    def test_memory_add_preview_uses_only_the_configured_cap(self):
        content = (
            "User prefers complete Discord tool previews for paths, memory "
            "changes, and delegated work."
        )

        result = build_tool_preview(
            "memory",
            {"action": "add", "target": "user", "content": content},
            max_len=0,
        )

        assert result == f'+user: "{content}"'

    def test_memory_replace_preview_shows_old_and_new_text(self):
        old = "Discord progress hides useful details"
        new = "Discord progress shows complete useful details"

        result = build_tool_preview(
            "memory",
            {
                "action": "replace",
                "target": "memory",
                "old_text": old,
                "content": new,
            },
            max_len=0,
        )

        assert result == f'~memory: "{old}" → "{new}"'

    def test_memory_batch_preview_lists_every_operation(self):
        result = build_tool_preview(
            "memory",
            {
                "target": "memory",
                "operations": [
                    {"action": "add", "content": "Keep complete paths"},
                    {
                        "action": "replace",
                        "old_text": "short previews",
                        "new_text": "complete previews",
                    },
                    {"action": "remove", "old_text": "stale display rule"},
                ],
            },
            max_len=0,
        )

        assert result == (
            '3 ops: +memory: "Keep complete paths" | '
            '~memory: "short previews" → "complete previews" | '
            '-memory: "stale display rule"'
        )

    def test_session_search_preview_has_no_hidden_query_cap(self):
        query = "Discord progress previews filenames paths memory and subagents"

        result = build_tool_preview(
            "session_search", {"query": query}, max_len=0,
        )

        assert result == f'recall: "{query}"'

    def test_process_preview_has_no_hidden_session_or_data_cap(self):
        session_id = "proc_4dae56ca81f6-complete-session-identifier"
        data = "complete payload sent to the tracked background process"

        result = build_tool_preview(
            "process",
            {"action": "submit", "session_id": session_id, "data": data},
            max_len=0,
        )

        assert result == f'submit {session_id} "{data}"'

    def test_send_message_preview_has_no_hidden_body_cap(self):
        message = "Complete outbound message text remains inspectable in progress"

        result = build_tool_preview(
            "send_message",
            {"target": "origin", "message": message},
            max_len=0,
        )

        assert result == f'to origin: "{message}"'

    def test_complete_previews_still_redact_secrets_before_delivery(self):
        secret = "sk-proj-" + "A" * 48
        command = f"curl -H 'Authorization: Bearer {secret}' https://example.com"

        prepared = prepare_tool_preview(
            "terminal", {"command": command}, fallback=command, max_len=0,
        )

        assert secret not in prepared.text
        assert "***" in prepared.text
        assert prepared.truncated is False

    def test_long_read_path_preserves_meaningful_tail(self):
        path = (
            "/home/example/.hermes/skills/research/references/"
            "important-target-file.md"
        )

        result = build_tool_preview("read_file", {"path": path}, max_len=40)

        assert result is not None
        assert len(result) == 40
        assert result.startswith("...")
        assert result.endswith("important-target-file.md")
        assert "/home/example" not in result

    def test_unlimited_read_preview_keeps_the_complete_path(self):
        path = (
            "/home/example/.hermes/skills/research/references/"
            "important-target-file.md"
        )

        result = build_tool_preview(
            "read_file", {"path": path, "offset": 4, "limit": 3}, max_len=0,
        )

        assert result == f"{path} L4-6"

    def test_path_cap_preserves_tail_for_file_tools(self):
        path = "/home/example/workspace/project/deep/path/final-report.md"

        result = truncate_tool_preview("write_file", path, 36)

        assert len(result) == 36
        assert result.startswith("...")
        assert result.endswith("final-report.md")

    def test_non_path_cap_keeps_command_front(self):
        command = "python scripts/really-long-command-name.py --with --many --flags"

        result = truncate_tool_preview("terminal", command, 40)

        assert result.startswith("python scripts/")
        assert result.endswith("...")
        assert "--many --flags" not in result

    def test_skill_reference_cap_preserves_the_file_path_tail(self):
        result = build_tool_preview(
            "skill_view",
            {
                "name": "hermes-agent",
                "file_path": "references/hermes-tool-preview-path-truncation.md",
            },
            max_len=40,
        )

        assert result is not None
        assert len(result) == 40
        assert result.startswith("...")
        assert result.endswith("tool-preview-path-truncation.md")

    def test_prepared_skill_reference_preserves_the_file_path_tail(self):
        prepared = prepare_tool_preview(
            "skill_view",
            {
                "name": "hermes-agent",
                "file_path": "references/hermes-tool-preview-path-truncation.md",
            },
            fallback="",
            max_len=40,
        )

        assert prepared.truncated is True
        assert prepared.text.startswith("...")
        assert prepared.text.endswith("tool-preview-path-truncation.md")

    def test_v4a_patch_preview_lists_every_target_path(self):
        patch_text = """*** Begin Patch
*** Update File: src/first.py
@@
-old
+new
*** Add File: docs/second.md
+content
*** Move File: config/old.yaml -> config/new.yaml
*** End Patch"""

        result = build_tool_preview(
            "patch", {"mode": "patch", "patch": patch_text}, max_len=0,
        )

        assert result == (
            "3 files: src/first.py | docs/second.md | "
            "config/old.yaml → config/new.yaml"
        )

    def test_codex_apply_patch_preview_lists_every_changed_path(self):
        paths = [f"src/component_{index}.py" for index in range(5)]

        result = build_tool_preview(
            "apply_patch",
            {"changes": [{"kind": "update", "path": path} for path in paths]},
            max_len=0,
        )

        assert result == "5 files: " + " | ".join(paths)

        capped = prepare_tool_preview(
            "apply_patch",
            {"changes": [{"kind": "update", "path": path} for path in paths]},
            fallback="",
            max_len=40,
        )
        assert capped.truncated is True
        assert capped.text.startswith("...")
        assert capped.text.endswith("src/component_4.py")

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestPrepareToolPreview:
    def test_file_path_preserves_filename_tail(self):
        path = "/home/example/workspace/project/deep/path/final-report.md"

        preview = prepare_tool_preview(
            "write_file", {"path": path}, fallback=path, max_len=36,
        )

        assert preview.text.startswith("...")
        assert preview.text.endswith("final-report.md")
        assert preview.truncated is True
        assert preview.url is None

    def test_recovers_and_describes_truncated_url(self):
        url = "https://example.com/a/very/long/path/to/a/page"
        set_tool_preview_max_len(20)

        preview = prepare_tool_preview(
            "web_extract",
            {"urls": [url]},
            fallback=url[:17] + "...",
            max_len=20,
        )

        assert preview.text == url[:17] + "..."
        assert preview.truncated is True
        assert preview.url == url

    def test_untruncated_url_has_no_link_target(self):
        url = "https://example.com/page"
        preview = prepare_tool_preview(
            "browser_navigate", None, fallback=url, max_len=40
        )

        assert preview.text == url
        assert preview.truncated is False
        assert preview.url is None

    def test_truncated_non_url_has_no_link_target(self):
        preview = prepare_tool_preview(
            "web_search",
            {"query": "how to parse a URL"},
            fallback="how to parse a URL",
            max_len=12,
        )

        assert preview.truncated is True
        assert preview.url is None


class TestCuteToolMessagePreviewLength:


    def test_search_files_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)

        assert pattern in line
        assert "..." not in line





    def test_browser_type_cute_message_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@password", "text": secret},
            0.1,
            result='{"success": true, "typed": "sk-pro...EFGH"}',
        )

        assert secret not in line
        assert "sk-pro" in line

    def test_browser_type_cute_message_keeps_normal_text(self):
        text = "hello world"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@search", "text": text},
            0.1,
            result='{"success": true, "typed": "hello world"}',
        )

        assert text in line


class TestEditDiffPreview:



    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff



    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0


    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


class TestBuildToolLabel:
    """Friendly human-phrased tool labels for built-in tools."""

    @pytest.fixture(autouse=True)
    def _enable_friendly(self):
        from agent.display import set_friendly_tool_labels
        set_friendly_tool_labels(True)
        yield
        set_friendly_tool_labels(True)

    def test_web_search_uses_for_connector(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_search", {"query": "weather in NYC"})
        assert label == 'Searching the web for weather in NYC'

    def test_web_extract_reads_url(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_extract", {"urls": ["https://example.com/page"]})
        assert label is not None
        assert label.startswith("Reading ")
        assert "example.com/page" in label







    def test_disabled_falls_back_to_preview(self):
        from agent.display import (
            build_tool_label,
            build_tool_preview,
            set_friendly_tool_labels,
        )
        set_friendly_tool_labels(False)
        args = {"query": "weather in NYC"}
        label = build_tool_label("web_search", args)
        # With the feature off, must match the raw preview exactly
        assert label == build_tool_preview("web_search", args)
        assert "Searching the web" not in (label or "")



class TestBuildStatusPhrase:
    """build_status_phrase — live working-state text for Slack's status line."""



    def test_verb_only_when_args_none(self):
        # live_status: "verb" mode passes args=None to suppress previews.
        from agent.display import build_status_phrase
        assert build_status_phrase("terminal", None) == "is running…"
        assert build_status_phrase("read_file", None) == "is reading…"



    def test_caps_length_for_slack_status_line(self):
        from agent.display import build_status_phrase
        phrase = build_status_phrase(
            "terminal", {"command": "x" * 300}, max_len=49
        )
        assert phrase is not None and len(phrase) <= 49
        assert phrase.endswith("…")


    def test_respects_friendly_labels_toggle(self):
        from agent.display import build_status_phrase, set_friendly_tool_labels
        set_friendly_tool_labels(False)
        try:
            assert build_status_phrase("terminal", {"command": "ls"}) is None
        finally:
            set_friendly_tool_labels(True)
