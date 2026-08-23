"""Tests for the Hindsight memory provider plugin.

Tests cover config loading, tool handlers (tags, max_tokens, types),
prefetch (auto_recall, preamble, query truncation), sync_turn (auto_retain,
turn counting, tags), and schema completeness.
"""

import json
import os
import re
import stat
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from hermes_cli.memory_setup import _CANCELLED
from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    RECALL_SCHEMA,
    REFLECT_SCHEMA,
    RETAIN_SCHEMA,
    _load_config,
    _load_simple_env,
    _build_embedded_profile_env,
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _resolve_bank_id_template,
    _WRITER_SENTINEL,
)
from plugins.memory.hindsight.settings import _sanitize_bank_segment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Ensure no stale env vars or Windows home state leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT", "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_OBSERVATION_SCOPE_EXCLUDE_TAG_PREFIXES",
        "HINDSIGHT_RECALL_TAGS",
        "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)

    # On Windows pathlib.Path.home() resolves USERPROFILE/HOMEDRIVE+HOMEPATH,
    # not the POSIX HOME alias that these tests historically monkeypatched.
    # Patch the actual API and keep all legacy profile writes in tmp_path.
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))


def _make_mock_client():
    """Create a mock Hindsight client with async methods."""
    async def _aretain(
        bank_id,
        content,
        timestamp=None,
        context=None,
        document_id=None,
        metadata=None,
        entities=None,
        tags=None,
        update_mode=None,
        retain_async=None,
    ):
        return SimpleNamespace(ok=True)

    client = MagicMock()
    client.aretain = AsyncMock(side_effect=_aretain)
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[
                SimpleNamespace(text="Memory 1"),
                SimpleNamespace(text="Memory 2"),
            ]
        )
    )
    client.areflect = AsyncMock(
        return_value=SimpleNamespace(text="Synthesized answer")
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _provider_for_mode(tmp_path, monkeypatch, mode: str):
    """Create an initialized provider without pre-seeding its client."""
    config = {
        "mode": mode,
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    return provider


def _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, mode: str):
    """Cloud/local-external clients must ensure lazy deps before importing."""
    import builtins

    provider = _provider_for_mode(tmp_path, monkeypatch, mode)
    ensure_calls = []

    def fake_ensure(feature, prompt=True):
        ensure_calls.append((feature, prompt))

    class FakeHindsight:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hindsight_client":
            if ensure_calls != [("memory.hindsight", False)]:
                raise ModuleNotFoundError("No module named 'hindsight_client'")
            return SimpleNamespace(Hindsight=FakeHindsight)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    client = provider._get_client()

    assert ensure_calls == [("memory.hindsight", False)]
    assert isinstance(client, FakeHindsight)
    assert client.kwargs == {
        "base_url": "http://localhost:9999",
        "timeout": 120.0,
        "api_key": "test-key",
    }


class _FakeSessionDB:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    def get_messages_as_conversation(self, session_id):
        return list(self._messages)


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create an initialized HindsightMemoryProvider with a mock client."""
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    p._client = _make_mock_client()
    return p


@pytest.fixture()
def provider_with_config(tmp_path, monkeypatch):
    """Create a provider factory that accepts custom config overrides."""
    def _make(**overrides):
        config = {
            "mode": "cloud",
            "apiKey": "test-key",
            "api_url": "http://localhost:9999",
            "bank_id": "test-bank",
            "budget": "mid",
            "memory_mode": "hybrid",
        }
        config.update(overrides)
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        p._client = _make_mock_client()
        return p
    return _make


def test_normalize_retain_tags_accepts_csv_and_dedupes():
    assert _normalize_retain_tags("agent:fakeassistantname, source_system:hermes-agent, agent:fakeassistantname") == [
        "agent:fakeassistantname",
        "source_system:hermes-agent",
    ]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_retain_schema_has_content(self):
        assert RETAIN_SCHEMA["name"] == "hindsight_retain"
        assert "content" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "tags" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "content" in RETAIN_SCHEMA["parameters"]["required"]


    def test_get_tool_schemas_returns_three(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}

    def test_context_mode_returns_no_tools(self, provider_with_config):
        p = provider_with_config(memory_mode="context")
        assert p.get_tool_schemas() == []

    def test_retain_tool_can_be_hidden_without_disabling_auto_retain(
        self, provider_with_config
    ):
        p = provider_with_config(expose_retain_tool=False, retain_async=False)

        assert [schema["name"] for schema in p.get_tool_schemas()] == [
            "hindsight_recall",
            "hindsight_reflect",
        ]
        assert "hindsight_retain" not in p.system_prompt_block()
        denied = json.loads(
            p.handle_tool_call("hindsight_retain", {"content": "do not store"})
        )
        assert "error" in denied
        p._client.aretain_batch.assert_not_called()

        p.sync_turn("automatic user", "automatic assistant")
        p._retain_queue.join()
        p._client.aretain_batch.assert_called_once()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_cloud_client_lazy_installs_dependency_before_import(self, tmp_path, monkeypatch):
        _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, "cloud")


    def test_default_values(self, provider):
        assert provider._auto_retain is True
        assert provider._auto_recall is True
        assert provider._retain_every_n_turns == 1
        assert provider._recall_max_tokens == 4096
        assert provider._recall_max_input_chars == 800
        assert provider._tags is None
        assert provider._observation_scopes is None
        assert provider._recall_tags is None
        # Default recall narrowed to observation-only; world/experience are
        # aggregate facts that often crowd out concrete-event signal during
        # auto-recall. Users opt back in via the recall_types config key.
        assert provider._recall_types == ["observation"]
        assert provider._bank_mission == ""
        assert provider._bank_retain_mission is None
        assert provider._retain_context == "conversation between Hermes Agent and the User"

    def test_recall_types_default_is_observation_only(self, provider):
        """Auto-recall must filter to observation by default."""
        assert provider._recall_types == ["observation"]


    def test_observation_scopes_keyword_config(self, provider_with_config):
        p = provider_with_config(observation_scopes="per_tag")
        assert p._observation_scopes == "per_tag"

    def test_recall_and_scope_filters_fall_back_to_environment(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setenv("HINDSIGHT_RECALL_TAGS", "profile:one, shared")
        monkeypatch.setenv(
            "HINDSIGHT_RETAIN_OBSERVATION_SCOPE_EXCLUDE_TAG_PREFIXES",
            "session:, parent:",
        )

        p = provider_with_config()

        assert p._recall_tags == ["profile:one", "shared"]
        assert p._observation_scope_exclude_tag_prefixes == [
            "session:",
            "parent:",
        ]

    def test_present_empty_recall_and_scope_config_override_environment(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setenv("HINDSIGHT_RECALL_TAGS", "profile:environment")
        monkeypatch.setenv(
            "HINDSIGHT_RETAIN_OBSERVATION_SCOPE_EXCLUDE_TAG_PREFIXES",
            "session:",
        )

        p = provider_with_config(
            recall_tags="",
            observation_scope_exclude_tag_prefixes="",
        )

        assert p._recall_tags is None
        assert p._observation_scope_exclude_tag_prefixes == []

    def test_quoted_boolean_values_use_shared_truth_parser(self, provider_with_config):
        p = provider_with_config(
            auto_retain="false",
            auto_recall="false",
            recall_sync="false",
            recall_indicator="false",
            retain_indicator="false",
            retain_async="false",
            prefetch_waits_for_retain="false",
            expose_retain_tool="false",
        )

        assert p._auto_retain is False
        assert p._auto_recall is False
        assert p._recall_sync is False
        assert p._recall_indicator is False
        assert p._retain_indicator is False
        assert p._retain_async is False
        assert p._prefetch_waits_for_retain is False
        assert p._expose_retain_tool is False

    def test_quoted_true_boolean_values_remain_enabled(self, provider_with_config):
        p = provider_with_config(
            auto_retain="true",
            auto_recall="yes",
            recall_sync="on",
            recall_indicator="1",
            retain_indicator="true",
            retain_async="yes",
            prefetch_waits_for_retain="on",
            expose_retain_tool="1",
        )

        assert all(
            (
                p._auto_retain,
                p._auto_recall,
                p._recall_sync,
                p._recall_indicator,
                p._retain_indicator,
                p._retain_async,
                p._prefetch_waits_for_retain,
                p._expose_retain_tool,
            )
        )


    def test_custom_config_values(self, provider_with_config):
        p = provider_with_config(
            retain_tags=["tag1", "tag2"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
            recall_tags=["recall-tag"],
            recall_tags_match="all",
            auto_retain=False,
            auto_recall=False,
            retain_every_n_turns=3,
            retain_context="custom-ctx",
            bank_retain_mission="Extract key facts",
            recall_max_tokens=2048,
            recall_types=["world", "experience"],
            recall_prompt_preamble="Custom preamble:",
            recall_max_input_chars=500,
            bank_mission="Test agent mission",
        )
        assert p._tags == ["tag1", "tag2"]
        assert p._retain_tags == ["tag1", "tag2"]
        assert p._retain_source == "hermes"
        assert p._retain_user_prefix == "User (fakeusername)"
        assert p._retain_assistant_prefix == "Assistant (fakeassistantname)"
        assert p._recall_tags == ["recall-tag"]
        assert p._recall_tags_match == "all"
        assert p._auto_retain is False
        assert p._auto_recall is False
        assert p._retain_every_n_turns == 3
        assert p._retain_context == "custom-ctx"
        assert p._bank_retain_mission == "Extract key facts"
        assert p._recall_max_tokens == 2048
        assert p._recall_types == ["world", "experience"]
        assert p._recall_prompt_preamble == "Custom preamble:"
        assert p._recall_max_input_chars == 500
        assert p._bank_mission == "Test agent mission"

    def test_retain_source_defaults_empty(self, provider):
        # Opt-in per AGENTS.md: no attribution tag ships by default.
        assert provider._retain_source == ""

    def test_retain_source_absent_from_metadata_by_default(self, provider):
        # metadata.source is stamped only when the user sets retain_source.
        meta = provider._build_metadata(message_count=2, turn_index=1)
        assert "source" not in meta

    def test_retain_source_user_override_wins(self, provider_with_config):
        # Users can still opt in explicitly (config key / env var).
        p = provider_with_config(retain_source="cogoport")
        assert p._retain_source == "cogoport"
        assert p._build_metadata(message_count=2, turn_index=1)["source"] == "cogoport"

    def test_embedded_profile_env_includes_idle_timeout_from_config(self):
        env = _build_embedded_profile_env({
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "idle_timeout": 0,
        })

        assert env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"


    def test_get_client_passes_idle_timeout_to_hindsight_embedded(self, monkeypatch):
        captured = {}

        class FakeHindsightEmbedded:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeHindsightEmbedded))
        monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

        p = HindsightMemoryProvider()
        p._mode = "local_embedded"
        p._config = {
            "profile": "hermes",
            "llm_provider": "openai_compatible",
            "llm_api_key": "test-key",
            "llm_model": "test-model",
            "idle_timeout": 0,
        }
        p._llm_base_url = "http://localhost:8060/v1"

        p._get_client()

        assert captured["idle_timeout"] == 0
        assert captured["llm_provider"] == "openai"


class TestPostSetup:
    def test_setup_cancel_at_mode_picker_writes_nothing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)

        save_config = MagicMock()
        which = MagicMock(return_value="/usr/bin/uv")
        run = MagicMock()
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: _CANCELLED)
        monkeypatch.setattr("shutil.which", which)
        monkeypatch.setattr("subprocess.run", run)
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("getpass.getpass", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("hermes_cli.config.save_config", save_config)

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {"provider": "builtin"}})

        save_config.assert_not_called()
        which.assert_not_called()
        run.assert_not_called()
        assert not (hermes_home / ".env").exists()
        assert not (hermes_home / "hindsight" / "config.json").exists()
        assert not (user_home / ".hindsight" / "profiles" / "hermes.env").exists()


    def test_local_embedded_setup_materializes_profile_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))

        selections = iter([1, 0])  # local_embedded, openai
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-local-test")
        saved_configs = []
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved_configs.append(cfg.copy()))

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {}})

        assert saved_configs[-1]["memory"]["provider"] == "hindsight"
        env_text = (hermes_home / ".env").read_text()
        assert "HINDSIGHT_LLM_API_KEY=sk-local-test\n" in env_text
        assert "HINDSIGHT_TIMEOUT=120\n" in env_text
        assert "HINDSIGHT_IDLE_TIMEOUT=300\n" in env_text

        profile_env = user_home / ".hindsight" / "profiles" / "hermes.env"
        assert profile_env.exists()
        assert profile_env.read_text() == (
            "HINDSIGHT_API_LLM_PROVIDER=openai\n"
            "HINDSIGHT_API_LLM_API_KEY=sk-local-test\n"
            "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
            "HINDSIGHT_API_LOG_LEVEL=info\n"
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=300\n"
        )


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_retain_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "user likes dark mode"}
        ))
        assert result["result"] == "Memory stored successfully."
        provider._client.aretain_batch.assert_called_once()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        item = call_kwargs["items"][0]
        assert item["content"] == "user likes dark mode"
        # bank_id/retain_async are call-level args, never item keys.
        assert "bank_id" not in item
        assert "retain_async" not in item

    def test_retain_defaults_item_timestamp_when_no_occurred_at(self, provider, monkeypatch):
        event_time = datetime(2026, 8, 24, 9, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
        monkeypatch.setattr("plugins.memory.hindsight._hermes_now", lambda: event_time)
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "user likes dark mode"}
        ))
        assert result["result"] == "Memory stored successfully."
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        # Non-temporal retains still carry a defaulted event timestamp so the
        # server can resolve any relative time phrases (#93568).
        assert item["timestamp"] == event_time.isoformat(timespec="seconds")

    def test_retain_threads_explicit_occurred_at_into_item_timestamp(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain",
            {"content": "user visited Paris", "occurred_at": "2026-03-03"},
        ))
        assert result["result"] == "Memory stored successfully."
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["timestamp"] == "2026-03-03"

    def test_retain_ignores_blank_occurred_at(self, provider, monkeypatch):
        event_time = datetime(2026, 8, 24, 9, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
        monkeypatch.setattr("plugins.memory.hindsight._hermes_now", lambda: event_time)
        json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "hello", "occurred_at": "   "}
        ))
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["timestamp"] == event_time.isoformat(timespec="seconds")

    def test_build_retain_kwargs_accepts_explicit_occurred_at(self, provider):
        item = provider._build_retain_kwargs("dinner with Sam", occurred_at="2026-08-20T19:00:00+02:00")
        assert item["timestamp"] == "2026-08-20T19:00:00+02:00"

    def test_retain_schema_exposes_occurred_at(self):
        from plugins.memory.hindsight import RETAIN_SCHEMA

        props = RETAIN_SCHEMA["parameters"]["properties"]
        assert "occurred_at" in props
        assert props["occurred_at"]["type"] == "string"
        # The description must steer the model to pass event times.
        assert "event" in props["occurred_at"]["description"].lower()
        assert "occurred_at" not in RETAIN_SCHEMA["parameters"]["required"]
    def test_retain_derives_scopes_from_merged_tags_without_volatile_lineage(
        self, provider_with_config
    ):
        p = provider_with_config(
            retain_tags=["scope:default", "source:auto"],
            observation_scope_exclude_tag_prefixes=["session:", "parent:"],
        )

        p.handle_tool_call(
            "hindsight_retain",
            {
                "content": "likes dark mode",
                "tags": ["project:hermes", "session:s1", "parent:p1"],
            },
        )

        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["tags"] == [
            "scope:default",
            "source:auto",
            "project:hermes",
            "session:s1",
            "parent:p1",
        ]
        assert item["observation_scopes"] == [[
            "scope:default",
            "source:auto",
            "project:hermes",
        ]]

    def test_all_excluded_tags_send_explicit_empty_scope(self, provider_with_config):
        p = provider_with_config(
            observation_scope_exclude_tag_prefixes=["session:", "parent:"],
        )

        item = p._build_retain_kwargs(
            "fact",
            tags=["session:s1", "parent:p1"],
        )

        assert item["observation_scopes"] == [[]]

    def test_explicit_observation_scopes_override_derived_filtering(
        self, provider_with_config
    ):
        p = provider_with_config(
            observation_scopes=[["session:s1"]],
            observation_scope_exclude_tag_prefixes=["session:"],
        )

        item = p._build_retain_kwargs("fact", tags=["session:s1", "stable"])

        assert item["observation_scopes"] == [["session:s1"]]


    def test_recall_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "dark mode"}
        ))
        assert "Memory 1" in result["result"]
        assert "Memory 2" in result["result"]


    def test_reflect_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_reflect", {"query": "summarize"}
        ))
        assert result["result"] == "Synthesized answer"


    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_unknown", {}
        ))
        assert "error" in result


    def test_local_embedded_recall_reconnects_after_idle_shutdown(self, provider, monkeypatch):
        first_client = _make_mock_client()
        first_client.arecall.side_effect = RuntimeError("Cannot connect to host 127.0.0.1:8888")
        second_client = _make_mock_client()
        second_client.arecall.return_value = SimpleNamespace(
            results=[SimpleNamespace(text="Recovered memory")]
        )
        clients = iter([first_client, second_client])

        provider._mode = "local_embedded"
        provider._client = first_client
        monkeypatch.setattr(provider, "_get_client", lambda: next(clients))

        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "test"}
        ))

        assert result["result"] == "1. Recovered memory"
        assert provider._client is second_client
        first_client.arecall.assert_called_once()
        second_client.arecall.assert_called_once()


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_when_no_result(self, provider):
        assert provider.prefetch("test") == ""


    def test_recall_sync_defaults_off(self, provider):
        assert provider._recall_sync is False

    def test_recall_sync_recalls_current_query_synchronously(self, provider_with_config):
        # recall_sync=True: prefetch() must do a live recall against the
        # *current* query (not read a previously queued buffer). #5820
        p = provider_with_config(recall_sync=True)
        captured = {}

        def _capture_recall(**kwargs):
            captured["query"] = kwargs.get("query", "")
            return SimpleNamespace(results=[SimpleNamespace(text="fresh memory")])

        p._client.arecall = AsyncMock(side_effect=_capture_recall)

        # Nothing pre-buffered — proves the result comes from a live recall.
        assert p._prefetch_result == ""
        result = p.prefetch("fix tests")

        assert captured["query"] == "fix tests"       # current query, not ignored
        assert "fresh memory" in result
        p._client.arecall.assert_called_once()

    def test_recall_sync_skips_background_queue(self, provider_with_config):
        # With sync recall there's nothing to prime in the background.
        p = provider_with_config(recall_sync=True)
        p.queue_prefetch("anything")
        assert p._prefetch_thread is None

    def test_async_default_ignores_current_query_and_reads_buffer(self, provider):
        # Default (recall_sync off): prefetch returns the buffered result and
        # does NOT issue a live recall for the current query.
        provider._prefetch_result = "- buffered from previous turn"
        result = provider.prefetch("a totally different current query")
        assert "buffered from previous turn" in result
        provider._client.arecall.assert_not_called()

    def test_queue_prefetch_skipped_in_tools_mode(self, provider_with_config):
        p = provider_with_config(memory_mode="tools")
        p.queue_prefetch("test")
        # Should not start a thread
        assert p._prefetch_thread is None

    def test_prefetch_waits_for_pending_retain_before_recall(self, provider):
        """The background prefetch must wait for queued retains to drain so the
        next turn's recall observes the just-completed turn (no retain race)."""
        import threading

        order = []
        release = threading.Event()

        async def _slow_retain(*args, **kwargs):
            release.wait(timeout=5.0)
            order.append("retain")

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        provider._client.aretain_batch = AsyncMock(side_effect=_slow_retain)
        provider._client.arecall = AsyncMock(side_effect=_recall)

        # Enqueue a slow retain, then immediately queue the next-turn prefetch.
        provider.sync_turn("hello", "world")
        provider.queue_prefetch("next turn query")

        # Let the prefetch thread start and reach the drain barrier.
        time.sleep(0.2)
        assert order == [], "recall ran before the pending retain drained"

        # Release the retain; the prefetch should now proceed AFTER it.
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        provider._retain_queue.join()
        assert order and order[0] == "retain"
        assert "recall" in order

    def test_prefetch_wait_for_retain_can_be_disabled(self, provider_with_config):
        p = provider_with_config(prefetch_waits_for_retain=False)
        p._client = _make_mock_client()
        assert p._prefetch_waits_for_retain is False


class TestPrefetchServerRetainVisibility:
    """PR #62871 review follow-up: draining the local writer queue is not a
    read-after-write signal for async retains. With ``retain_async=True`` the
    server accepts the write and returns an ``operation_id`` that stays
    ``pending`` until the write is durable/recall-visible. The background
    prefetch must gate on server-side operation completion, not just the local
    queue, before recalling.
    """

    def _client_with_ops(self, statuses):
        """Mock client whose aretain_batch returns an async operation_id and
        whose operations.get_operation_status yields *statuses* in order
        (last value repeats)."""
        client = _make_mock_client()
        client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-1", operation_ids=None)
        )
        seq = list(statuses)

        async def _status(**kwargs):
            value = seq.pop(0) if len(seq) > 1 else seq[0]
            return SimpleNamespace(status=value)

        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(side_effect=_status)
        return client

    def test_tracks_async_operation_id_from_retain(self, provider):
        provider._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-async-1", operation_ids=None)
        )
        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert "op-async-1" in provider._pending_retain_ops

    def test_tracks_multiple_operation_ids(self, provider):
        provider._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(
                operation_id=None, operation_ids=["op-a", "op-b"]
            )
        )
        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert {"op-a", "op-b"} <= provider._pending_retain_ops

    def test_sync_retain_tracks_no_ops(self, provider_with_config):
        p = provider_with_config(retain_async=False)
        p._client = _make_mock_client()
        p._client.aretain_batch = AsyncMock(
            return_value=SimpleNamespace(operation_id="op-x", operation_ids=None)
        )
        p.sync_turn("hello", "world")
        p._retain_queue.join()
        # retain_async=False → no server-side op to wait on.
        assert p._pending_retain_ops == set()

    def test_prefetch_waits_for_server_completion_before_recall(self, provider):
        """Recall must not run until the tracked async op reports completed."""
        order = []

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        provider._client = self._client_with_ops(["pending", "pending", "completed"])
        provider._client.arecall = AsyncMock(side_effect=_recall)

        provider.sync_turn("hello", "world")
        provider._retain_queue.join()
        assert "op-1" in provider._pending_retain_ops

        provider.queue_prefetch("next turn query")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        # Recall ran, the op was polled to completion, and the pending set
        # was cleared (so a later prefetch won't re-poll it).
        assert order == ["recall"]
        assert provider._client.operations.get_operation_status.await_count >= 3
        assert provider._pending_retain_ops == set()

    def test_prefetch_proceeds_after_server_wait_timeout(self, provider_with_config):
        """A wedged/never-completing async op must not hang prefetch forever;
        it recalls anyway once the drain budget is exhausted."""
        p = provider_with_config(prefetch_retain_drain_timeout=0.3)
        order = []

        async def _recall(**kwargs):
            order.append("recall")
            return SimpleNamespace(results=[SimpleNamespace(text="m")])

        p._client = self._client_with_ops(["pending"])  # never completes
        p._client.arecall = AsyncMock(side_effect=_recall)

        p.sync_turn("hello", "world")
        p._retain_queue.join()

        start = time.monotonic()
        p.queue_prefetch("next turn query")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        elapsed = time.monotonic() - start

        assert order == ["recall"], "prefetch should recall after the timeout"
        assert elapsed < 3.0, "prefetch must not block well past the drain budget"

    def test_timed_out_ops_remain_retryable_without_reburning_timeout(self, provider_with_config):
        """Accepted ops survive timeout while persistent backoff keeps the
        next prefetch from immediately re-burning the full wait budget."""
        p = provider_with_config(prefetch_retain_drain_timeout=0.3)
        p._client = self._client_with_ops(["pending"])  # never completes
        p._client.arecall = AsyncMock(
            return_value=SimpleNamespace(results=[SimpleNamespace(text="m")])
        )

        p.sync_turn("hello", "world")
        p._retain_queue.join()
        assert p._pending_retain_ops, "op should be tracked before the wait"

        # First prefetch leaves the accepted-but-unresolved op durable state.
        p.queue_prefetch("q1")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        assert p._pending_retain_ops == {"op-1"}
        first_poll_count = p._client.operations.get_operation_status.await_count

        # Persistent next-poll backoff makes an immediate later prefetch cheap.
        start = time.monotonic()
        p.queue_prefetch("q2")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        assert time.monotonic() - start < 0.25, (
            "second prefetch re-burned the unresolved operation timeout"
        )
        assert p._client.operations.get_operation_status.await_count == first_poll_count

        # A later durability barrier can settle the very same operation.
        p._client.operations.get_operation_status = AsyncMock(
            return_value=SimpleNamespace(status="completed")
        )
        assert p.drain_pending(timeout=1.0) is True
        assert p._pending_retain_ops == set()

    def test_operation_notfound_treated_as_complete(self, provider):
        """A NotFound (completed+evicted) op is treated as done, not pending."""
        from hindsight_client_api.exceptions import NotFoundException

        client = _make_mock_client()
        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(
            side_effect=NotFoundException(status=404, reason="gone")
        )
        provider._client = client

        assert provider._is_retain_op_complete("bank", "op-gone") is True

    def test_transient_status_error_keeps_waiting(self, provider):
        """A transient status-check error means 'unknown', so keep waiting."""
        client = _make_mock_client()
        client.operations = MagicMock()
        client.operations.get_operation_status = AsyncMock(
            side_effect=RuntimeError("temporary blip")
        )
        provider._client = client

        assert provider._is_retain_op_complete("bank", "op-1") is False

    def test_operation_polling_preserves_each_immutable_bank(self, provider):
        seen = []

        async def _status(**kwargs):
            seen.append((kwargs["bank_id"], kwargs["operation_id"]))
            return SimpleNamespace(status="completed")

        provider._client.operations = MagicMock()
        provider._client.operations.get_operation_status = AsyncMock(
            side_effect=_status
        )
        provider._track_retain_ops(
            SimpleNamespace(operation_id="op-a", operation_ids=None),
            "bank-a",
        )
        provider._track_retain_ops(
            SimpleNamespace(operation_id="op-b", operation_ids=None),
            "bank-b",
        )

        assert provider._wait_for_server_retain_ops(
            time.monotonic() + 1.0,
            1.0,
        )
        assert set(seen) == {("bank-a", "op-a"), ("bank-b", "op-b")}

    def test_operation_status_poll_respects_shared_drain_budget(self, provider):
        import asyncio

        async def _slow_status(**_kwargs):
            await asyncio.sleep(0.35)
            return SimpleNamespace(status="pending")

        provider._client.operations = MagicMock()
        provider._client.operations.get_operation_status = AsyncMock(
            side_effect=_slow_status
        )
        provider._track_retain_ops(
            SimpleNamespace(operation_id="op-slow", operation_ids=None),
            "bank-slow",
        )

        started = time.monotonic()
        assert provider.drain_pending(timeout=0.05) is False
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert provider._pending_retain_ops == {"op-slow"}

    def test_append_watermark_advances_only_after_remote_completion(
        self, provider, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        provider._client = self._client_with_ops(["completed"])

        provider.sync_turn("hello", "world")
        provider._retain_queue.join()

        state = provider._active_delivery_state
        assert state.committed == 0
        assert state.queued == 1
        assert provider._last_retained_turn_count == 0
        assert provider._queued_retained_turn_count == 1

        assert provider.drain_pending(timeout=1.0)
        assert state.committed == 1
        assert provider._last_retained_turn_count == 1

    def test_failed_remote_append_retries_bounded_and_remains_retryable(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_async=True)
        client = MagicMock()
        sequence = iter(range(1, 20))
        client.aretain_batch.side_effect = lambda **_kwargs: SimpleNamespace(
            operation_id=f"op-{next(sequence)}",
            operation_ids=None,
        )
        client.operations.get_operation_status.return_value = SimpleNamespace(
            status="failed"
        )
        p._run_hindsight_operation = (
            lambda operation, **_kwargs: operation(client)
        )
        p._client = None

        p.sync_turn("first user", "first assistant")
        p._retain_queue.join()
        started = time.monotonic()
        assert p.drain_pending(timeout=0.75) is False

        state = p._active_delivery_state
        assert time.monotonic() - started < 1.0
        assert 2 <= client.aretain_batch.call_count <= 5
        assert state.committed == 0
        assert state.queued == 0
        assert state.failed is True
        assert p._pending_retain_ops == set()


# ---------------------------------------------------------------------------
# recall_status (deterministic recall indicator) tests
# ---------------------------------------------------------------------------


class TestRecallStatus:
    def test_none_before_any_prefetch(self, provider):
        # Nothing recalled yet → no indicator.
        assert provider.recall_status() is None

    def test_reports_count_after_recall(self, provider):
        # Mock client returns 2 memories; prefetch consumes the block.
        provider.queue_prefetch("test")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        provider.prefetch("test")

        status = provider.recall_status()
        assert status is not None
        assert status.provider_label == "Hindsight"
        assert status.count == 2

    def test_reports_count_in_recall_sync_mode(self, provider_with_config):
        # recall_sync path does a live recall inside prefetch() (no background
        # prime) — the indicator must still report the count for that turn.
        p = provider_with_config(recall_sync=True)
        assert p.prefetch("test")  # live recall returns the 2 mock memories
        status = p.recall_status()
        assert status is not None
        assert status.count == 2

    def test_none_when_recall_returned_nothing(self, provider):
        provider._client.arecall = AsyncMock(
            return_value=SimpleNamespace(results=[])
        )
        provider.queue_prefetch("test")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert provider.prefetch("test") == ""
        assert provider.recall_status() is None

    def test_stale_count_cleared_on_empty_turn(self, provider):
        # First turn recalls 2 memories.
        provider.queue_prefetch("test")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        provider.prefetch("test")
        assert provider.recall_status().count == 2

        # Next turn recalls nothing — the prior count must not linger.
        provider._client.arecall = AsyncMock(
            return_value=SimpleNamespace(results=[])
        )
        provider.queue_prefetch("test2")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        provider.prefetch("test2")
        assert provider.recall_status() is None

    def test_suppressed_when_indicator_off(self, provider_with_config):
        p = provider_with_config(recall_indicator=False)
        p.queue_prefetch("test")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        p.prefetch("test")
        # Memory was injected, but the indicator is turned off.
        assert p._last_recall_returned is True
        assert p.recall_status() is None

    def test_reflect_mode_reports_generic_count(self, provider_with_config):
        p = provider_with_config(recall_prefetch_method="reflect")
        p.queue_prefetch("test")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        p.prefetch("test")
        status = p.recall_status()
        assert status is not None
        # Reflect synthesizes across memories → no discrete count (0).
        assert status.count == 0


# ---------------------------------------------------------------------------
# sync_turn tests
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_sync_turn_retains_metadata_rich_turn(self, provider_with_config, monkeypatch):
        event_time = datetime(2026, 8, 10, 11, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
        monkeypatch.setattr("plugins.memory.hindsight._hermes_now", lambda: event_time)
        p = provider_with_config(
            retain_tags=["conv", "session1"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
        )
        p.initialize(
            session_id="session-1",
            platform="discord",
            user_id="fakeusername-123",
            user_name="fakeusername",
            chat_id="1485316232612941897",
            chat_name="fakeassistantname-forums",
            chat_type="thread",
            thread_id="1491249007475949698",
            agent_identity="fakeassistantname",
        )
        p._client = _make_mock_client()

        p.sync_turn("hello", "hi there")
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["document_id"].startswith("session-1-")
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        item = call_kwargs["items"][0]
        assert item["context"] == "conversation between Hermes Agent and the User"
        assert item["tags"] == ["conv", "session1", "session:session-1"]
        content = json.loads(item["content"])
        assert len(content) == 1
        assert content[0][0]["role"] == "user"
        assert content[0][0]["content"] == "User (fakeusername): hello"
        assert content[0][1]["role"] == "assistant"
        assert content[0][1]["content"] == "Assistant (fakeassistantname): hi there"
        assert item["metadata"]["source"] == "hermes"
        assert item["metadata"]["session_id"] == "session-1"
        assert item["metadata"]["platform"] == "discord"
        assert item["metadata"]["user_id"] == "fakeusername-123"
        assert item["metadata"]["user_name"] == "fakeusername"
        assert item["metadata"]["chat_id"] == "1485316232612941897"
        assert item["metadata"]["chat_name"] == "fakeassistantname-forums"
        assert item["metadata"]["chat_type"] == "thread"
        assert item["metadata"]["thread_id"] == "1491249007475949698"
        assert item["metadata"]["agent_identity"] == "fakeassistantname"
        assert item["metadata"]["turn_index"] == "1"
        assert item["metadata"]["message_count"] == "2"
        assert content[0][0]["timestamp"] == event_time.isoformat(timespec="seconds")
        assert content[0][1]["timestamp"] == event_time.isoformat(timespec="seconds")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", item["metadata"]["retained_at"])
        assert item["timestamp"] == event_time.isoformat(timespec="seconds")

    def test_retain_timestamp_normalizes_a_naive_clock(self, provider, monkeypatch):
        event_time = datetime(2026, 8, 10, 11, 9)
        monkeypatch.setattr("plugins.memory.hindsight._hermes_now", lambda: event_time)

        timestamp = provider._build_retain_kwargs("hello")["timestamp"]
        parsed = datetime.fromisoformat(timestamp)

        assert parsed.tzinfo is not None
        assert parsed.utcoffset() is not None

    @pytest.mark.asyncio
    async def test_retain_timestamp_is_serialized_by_pinned_client(self, provider):
        hindsight_client = pytest.importorskip(
            "hindsight_client", reason="pinned hindsight-client SDK not installed"
        )
        Hindsight = hindsight_client.Hindsight

        item = provider._build_retain_kwargs("hello")
        item.pop("bank_id", None)
        item.pop("retain_async", None)

        client = Hindsight(base_url="http://localhost:9999", api_key="test-key")
        client._memory_api.retain_memories = AsyncMock(return_value=SimpleNamespace(ok=True))
        try:
            await client.aretain_batch(bank_id="test-bank", items=[item])
            call = client._memory_api.retain_memories.await_args
            assert call is not None
            request = call.args[1]
            assert request.to_dict()["items"][0]["timestamp"] == item["timestamp"]
        finally:
            await client.aclose()


    def test_resume_creates_new_document(self, tmp_path, monkeypatch):
        """Resuming a session (re-initializing) gets a new document_id
        so previously stored content is not overwritten."""
        config = {"mode": "cloud", "apiKey": "k", "api_url": "http://x", "bank_id": "b"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p1 = HindsightMemoryProvider()
        p1.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Sleep just enough that the microsecond timestamp differs
        import time
        time.sleep(0.001)

        p2 = HindsightMemoryProvider()
        p2.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Same session, but each process gets its own document_id
        assert p1._document_id != p2._document_id
        assert p1._document_id.startswith("resumed-session-")
        assert p2._document_id.startswith("resumed-session-")


# ---------------------------------------------------------------------------
# retain indicator ("saving to memory") tests
# ---------------------------------------------------------------------------


class TestRetainIndicator:
    _SAVING = "👁️ Hindsight — saving to memory…"

    def test_emits_saving_on_dispatch(self, provider_with_config):
        calls = []
        p = provider_with_config(retain_async=False)
        p._status_callback = calls.append
        p.sync_turn("hello", "hi")
        p._retain_queue.join()
        assert self._SAVING in calls

    def test_suppressed_when_indicator_off(self, provider_with_config):
        calls = []
        p = provider_with_config(retain_indicator=False, retain_async=False)
        p._status_callback = calls.append
        p.sync_turn("hello", "hi")
        p._retain_queue.join()
        assert calls == []

    def test_no_emit_when_auto_retain_off(self, provider_with_config):
        calls = []
        p = provider_with_config(auto_retain=False)
        p._status_callback = calls.append
        p.sync_turn("hello", "hi")  # returns early — nothing dispatched
        assert calls == []

    def test_no_emit_on_buffered_turn(self, provider_with_config):
        # retain_every_n_turns=2: turn 1 buffers (no write, no line),
        # turn 2 flushes (one line) — "saving" only fires on a real write.
        calls = []
        p = provider_with_config(retain_every_n_turns=2, retain_async=False)
        p._status_callback = calls.append
        p.sync_turn("t1-u", "t1-a")
        assert calls == []
        p.sync_turn("t2-u", "t2-a")
        p._retain_queue.join()
        assert calls == [self._SAVING]

    def test_no_crash_without_callback(self, provider_with_config):
        p = provider_with_config(retain_async=False)
        assert p._status_callback is None
        p.sync_turn("hello", "hi")  # must not raise
        p._retain_queue.join()

    def test_status_callback_wired_from_initialize(self, tmp_path, monkeypatch):
        cb = lambda _m: None
        p = _provider_for_mode(tmp_path, monkeypatch, "cloud")
        p.initialize(session_id="s", hermes_home=str(tmp_path), status_callback=cb)
        assert p._status_callback is cb


# ---------------------------------------------------------------------------
# Shutdown / writer tests
# ---------------------------------------------------------------------------


class TestShutdownRace:
    def test_sync_turn_uses_single_writer_thread(self, provider):
        """All retains run through one long-lived writer thread."""
        provider.sync_turn("a", "b")
        provider._retain_queue.join()
        first_writer = provider._writer_thread
        assert first_writer is not None
        assert first_writer.is_alive()

        provider.sync_turn("c", "d")
        provider._retain_queue.join()
        # Same thread reused — no ad-hoc thread per call.
        assert provider._writer_thread is first_writer
        assert provider._client.aretain_batch.call_count == 2


    def test_shutdown_drains_pending_retains(self, provider):
        """Shutdown must wait for queued retains to complete, not abandon them.

        Otherwise the LAST in-flight turn — typically the most important —
        is silently lost.
        """
        client = provider._client
        provider.sync_turn("a", "b")
        provider.sync_turn("c", "d")
        provider.shutdown()
        # Both retains drained before shutdown returned.
        assert client.aretain_batch.call_count == 2
        assert provider._retain_queue.empty()

    @pytest.mark.parametrize("retain_async", [False, True])
    def test_shutdown_flushes_below_threshold_buffer_once(
        self, provider_with_config, monkeypatch, retain_async
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: False,
        )
        p = provider_with_config(
            retain_every_n_turns=3,
            retain_async=retain_async,
        )
        client = MagicMock()
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None
        p.sync_turn("first user", "first assistant")
        p.sync_turn("second user", "second assistant")
        client.aretain_batch.assert_not_called()

        p.shutdown(settlement_timeout=1.0)

        client.aretain_batch.assert_called_once()
        kwargs = client.aretain_batch.call_args.kwargs
        assert kwargs["retain_async"] is retain_async
        assert len(json.loads(kwargs["items"][0]["content"])) == 2

    def test_shutdown_does_not_duplicate_exact_boundary_retain(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: False,
        )
        p = provider_with_config(retain_every_n_turns=2, retain_async=False)
        client = MagicMock()
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None

        p.sync_turn("first user", "first assistant")
        p.sync_turn("second user", "second assistant")
        p.shutdown(settlement_timeout=1.0)

        client.aretain_batch.assert_called_once()

    def test_shutdown_retries_failed_legacy_write_without_losing_full_document(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: False,
        )
        p = provider_with_config(retain_every_n_turns=2, retain_async=False)
        client = MagicMock()
        client.aretain_batch.side_effect = [RuntimeError("temporary"), None]
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None

        p.sync_turn("first user", "first assistant")
        p.sync_turn("second user", "second assistant")
        p.shutdown(settlement_timeout=1.0)

        assert client.aretain_batch.call_count == 2
        retry_content = json.loads(
            client.aretain_batch.call_args_list[1].kwargs["items"][0]["content"]
        )
        assert len(retry_content) == 2

    def test_shutdown_retains_unresolved_remote_operation_with_bounded_result(
        self, provider_with_config, monkeypatch, caplog
    ):
        import logging

        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_async=True)
        client = MagicMock()
        client.aretain_batch.return_value = SimpleNamespace(
            operation_id="op-never",
            operation_ids=None,
        )
        client.operations.get_operation_status.return_value = SimpleNamespace(
            status="pending"
        )
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None
        p.sync_turn("user", "assistant")
        p._retain_queue.join()

        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            p.shutdown(settlement_timeout=0.1)

        assert time.monotonic() - started < 0.75
        assert p._last_drain_ok is False
        assert p._pending_retain_ops == {"op-never"}
        assert any(
            "keeping 1 accepted operation" in record.getMessage()
            for record in caplog.records
        )


class TestDeliveryLedger:
    def test_failed_local_append_retries_full_uncommitted_suffix_once(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_async=False)
        client = MagicMock()
        client.aretain_batch.side_effect = [RuntimeError("temporary"), None]
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None

        p.sync_turn("first user", "first assistant")
        p._retain_queue.join()
        state = p._active_delivery_state
        assert state.committed == 0
        assert state.queued == 0

        p.sync_turn("second user", "second assistant")
        p._retain_queue.join()

        assert client.aretain_batch.call_count == 2
        retry_content = json.loads(
            client.aretain_batch.call_args_list[1].kwargs["items"][0]["content"]
        )
        assert len(retry_content) == 2
        assert state.committed == 2
        assert state.queued == 2

    def test_completed_active_ledger_is_pruned_then_reowned_for_partial_suffix(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_every_n_turns=2, retain_async=False)
        client = p._client

        p.sync_turn("user 1", "assistant 1")
        p.sync_turn("user 2", "assistant 2")
        p._retain_queue.join()
        completed_state = p._active_delivery_state
        assert completed_state not in p._delivery_states

        # The same active buffer gains a below-threshold suffix after its
        # completed ledger was pruned.  Shutdown must re-own and flush it.
        p.sync_turn("user 3", "assistant 3")
        p.shutdown(settlement_timeout=1.0)

        assert client.aretain_batch.call_count == 2
        suffix = json.loads(
            client.aretain_batch.call_args_list[1].kwargs["items"][0]["content"]
        )
        assert len(suffix) == 1

    def test_failed_delivery_exhaustion_is_not_reburned_by_later_drains(
        self, provider_with_config, monkeypatch
    ):
        """A permanently failed suffix keeps explicit unresolved state, but a
        later flush/shutdown barrier must not start the retry budget over."""
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_async=False)
        client = MagicMock()
        client.aretain_batch.side_effect = RuntimeError("permanent")
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None

        # Initial turn admission fails.  A durability drain then gets one
        # lifecycle admission plus the two configured automatic retries.
        p.sync_turn("user", "assistant")
        p._retain_queue.join()
        assert p.drain_pending(timeout=1.0) is False
        exhausted_call_count = client.aretain_batch.call_count
        assert exhausted_call_count == 4

        # Neither another explicit drain nor shutdown may re-admit the same
        # failed immutable range after the retry budget has been exhausted.
        assert p.drain_pending(timeout=0.2) is False
        p.shutdown(settlement_timeout=0.2)
        assert client.aretain_batch.call_count == exhausted_call_count
        assert p._active_delivery_state.failed is True
        assert p._last_drain_ok is False

    def test_switch_settles_old_remote_range_then_only_remaining_suffix(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_every_n_turns=2, retain_async=True)
        client = MagicMock()
        client.aretain_batch.side_effect = [
            SimpleNamespace(operation_id="op-first", operation_ids=None),
            SimpleNamespace(operation_id=None, operation_ids=None),
        ]
        client.operations.get_operation_status.return_value = SimpleNamespace(
            status="completed"
        )
        p._run_hindsight_operation = lambda operation, **_kwargs: operation(client)
        p._client = None

        p.sync_turn("first user", "first assistant")
        p.sync_turn("second user", "second assistant")
        p._retain_queue.join()
        old_state = p._active_delivery_state
        p.sync_turn("third user", "third assistant")

        p.on_session_switch(
            "new-session",
            parent_session_id="test-session",
            reset=True,
        )
        assert client.aretain_batch.call_count == 1
        assert old_state.bank_id == "test-bank"
        assert old_state.document_id == "test-session"

        assert p.drain_pending(timeout=1.0)

        assert client.aretain_batch.call_count == 2
        first = json.loads(
            client.aretain_batch.call_args_list[0].kwargs["items"][0]["content"]
        )
        remainder = json.loads(
            client.aretain_batch.call_args_list[1].kwargs["items"][0]["content"]
        )
        assert len(first) == 2
        assert len(remainder) == 1
        assert old_state.committed == 3
        assert p._session_id == "new-session"


# ---------------------------------------------------------------------------
# on_session_switch — flush + prefetch reset behavior
# ---------------------------------------------------------------------------


class TestSessionSwitchBufferFlush:
    def test_buffered_turns_flushed_before_clear(self, provider_with_config):
        """retain_every_n_turns > 1 must not silently drop partial buffers
        on session switch. Whatever's in _session_turns at switch time
        should land in the OLD document under the OLD session id."""
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        old_doc = p._document_id

        # Two turns buffered, no retain yet (boundary is at turn 3). The
        # writer hasn't been started either — sync_turn's early return
        # skips _ensure_writer when no retain is due.
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None
        p._client.aretain_batch.assert_not_called()

        # Switch — flush should fire under OLD document_id via the writer queue.
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        kw = p._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        item = kw["items"][0]
        # Both buffered turns must be present in the flushed payload.
        content = json.loads(item["content"])
        flat = json.dumps(content)
        assert "turn1-user" in flat
        assert "turn2-user" in flat
        # Old session id must appear in lineage tags / metadata.
        assert "session:test-session" in item["tags"]
        assert item["metadata"]["session_id"] == "test-session"

        # And the new session must start with a clean slate.
        assert p._session_id == "new-sid"
        assert p._session_turns == []
        assert p._turn_counter == 0
        assert p._document_id != old_doc
        assert p._document_id.startswith("new-sid-")


    def test_in_flight_prefetch_thread_drained_on_switch(self, provider, monkeypatch):
        """on_session_switch must wait for an in-flight prefetch from the
        old session to settle before clearing _prefetch_result, otherwise
        the thread can race and re-populate the field after the clear."""
        import threading

        gate = threading.Event()
        finished = threading.Event()

        def _slow_prefetch():
            gate.wait(timeout=5.0)
            with provider._prefetch_lock:
                provider._prefetch_result = "old-session recall"
            finished.set()

        provider._prefetch_thread = threading.Thread(target=_slow_prefetch, daemon=True)
        provider._prefetch_thread.start()

        # Release the prefetch worker so it writes _prefetch_result, then
        # call on_session_switch — it must join the thread before clearing.
        gate.set()
        provider.on_session_switch("new-sid")

        assert finished.is_set(), "switch returned before prefetch thread settled"
        assert provider._prefetch_result == ""

    def test_switch_to_root_session_clears_stale_parent(self, provider):
        provider._parent_session_id = "old-parent"

        provider.on_session_switch("root-session", parent_session_id="")

        assert provider._parent_session_id == ""

    def test_rewind_discards_active_suffix_without_flushing(self, provider_with_config):
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        old_document_id = p._document_id
        p.sync_turn("removed user 1", "removed assistant 1")
        p.sync_turn("removed user 2", "removed assistant 2")

        p.on_session_switch("test-session", rewound=True)
        p._retain_queue.join()

        p._client.aretain_batch.assert_not_called()
        assert p._session_id == "test-session"
        assert p._document_id == old_document_id
        assert p._session_turns == []

    def test_rewind_invalidates_append_job_blocked_in_writer_queue(
        self, provider_with_config, monkeypatch
    ):
        import threading

        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *_args, **_kwargs: True,
        )
        p = provider_with_config(retain_async=False)
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        def _block_writer():
            blocker_started.set()
            release_blocker.wait(timeout=2.0)

        p._ensure_writer()
        p._retain_queue.put(_block_writer)
        assert blocker_started.wait(timeout=1.0)
        p.sync_turn("removed user", "removed assistant")
        removed_state = p._active_delivery_state

        p.on_session_switch("test-session", rewound=True)
        release_blocker.set()
        p._retain_queue.join()

        assert removed_state.invalidated is True
        p._client.aretain_batch.assert_not_called()

    def test_sync_turn_rejects_mismatched_session_ownership(self, provider):
        provider.sync_turn(
            "wrong user",
            "wrong assistant",
            session_id="different-session",
        )

        assert provider._session_turns == []
        provider._client.aretain_batch.assert_not_called()

    def test_prefetch_completing_after_switch_cannot_publish_or_set_indicator(
        self, provider
    ):
        import threading

        started = threading.Event()
        release = threading.Event()
        old_client = SimpleNamespace(
            arecall=MagicMock(
                return_value=SimpleNamespace(
                    results=[SimpleNamespace(text="old-session recall")]
                )
            )
        )

        def _blocked_operation(operation, **_kwargs):
            started.set()
            release.wait(timeout=2.0)
            return operation(old_client)

        provider._prefetch_waits_for_retain = False
        provider._run_hindsight_operation = _blocked_operation
        provider.queue_prefetch("old query", session_id="test-session")
        old_worker = provider._prefetch_thread
        assert started.wait(timeout=1.0)
        threading.Timer(0.05, release.set).start()

        provider.on_session_switch("new-session")
        old_worker.join(timeout=1.0)

        assert provider.prefetch("new query", session_id="new-session") == ""
        assert provider.recall_status() is None

    def test_newest_same_session_prefetch_wins_and_preserves_count(self, provider):
        import threading

        older_started = threading.Event()
        release_older = threading.Event()

        def _recall(**kwargs):
            if kwargs["query"] == "older":
                older_started.set()
                release_older.wait(timeout=2.0)
                text = "older-result"
            else:
                text = "newer-result"
            return SimpleNamespace(results=[SimpleNamespace(text=text)])

        fake = SimpleNamespace(arecall=_recall)
        provider._run_hindsight_operation = lambda operation, **_kwargs: operation(fake)
        provider._prefetch_waits_for_retain = False

        provider.queue_prefetch("older", session_id="test-session")
        older_worker = provider._prefetch_thread
        assert older_started.wait(timeout=1.0)
        provider.queue_prefetch("newer", session_id="test-session")
        newer_worker = provider._prefetch_thread
        newer_worker.join(timeout=1.0)
        release_older.set()
        older_worker.join(timeout=1.0)

        context = provider.prefetch("next", session_id="test-session")
        status = provider.recall_status()
        assert "newer-result" in context
        assert "older-result" not in context
        assert status is not None
        assert status.count == 1

    def test_flush_serializes_behind_pending_retains_via_writer_queue(
        self, provider_with_config
    ):
        """The flush closure must ride the same _retain_queue sync_turn
        uses, so it lands FIFO behind any still-queued old-session
        retains rather than racing them on a separate thread.

        Regression guard: an earlier draft spawned a raw threading.Thread
        for flush, overwriting _sync_thread and racing the writer against
        the same document_id.
        """
        import threading as _threading

        p = provider_with_config(retain_every_n_turns=2, retain_async=False)

        # Block the first writer job until we've enqueued the flush
        # behind it. This proves ordering — the flush MUST wait.
        gate = _threading.Event()
        call_order: list[str] = []

        def _aretain_batch_tracking(**kw):
            idx = kw["items"][0]["metadata"].get("turn_index", "")
            call_order.append(str(idx))
            if idx == "2":
                # First retain blocks until we've enqueued the flush.
                gate.wait(timeout=5.0)

        p._client.aretain_batch = AsyncMock(side_effect=_aretain_batch_tracking)

        # Turn 1+2 → boundary hit → retain enqueued (will block).
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")

        # One more buffered turn so flush has something to land.
        p.sync_turn("turn3-user", "turn3-asst")

        # Switch while the first retain is still blocked on `gate`.
        p.on_session_switch("new-sid", parent_session_id="test-session")

        # Release the first retain. Flush must have been enqueued
        # BEHIND it, and run second.
        gate.set()
        p._retain_queue.join()

        # The flush carries all buffered turns; sync_turn's retain #2
        # carried the batch at boundary time. Two distinct calls.
        assert p._client.aretain_batch.call_count == 2
        # First call landed while buffer was [t1, t2]; flush landed
        # after we added t3. So the second call must be strictly after.
        assert call_order[0] == "2"
        # Flush retain has turn_index matching the buffered count at
        # switch time (3 turns accumulated, _turn_index was set to 3
        # by the last sync_turn).
        assert call_order[1] == "3"


# ---------------------------------------------------------------------------
# update_mode='append' capability probe + retain dispatch
# ---------------------------------------------------------------------------


class TestUpdateModeAppendCapability:
    def _clear_capability_cache(self):
        from plugins.memory.hindsight import _append_capability_cache, _append_capability_lock
        with _append_capability_lock:
            _append_capability_cache.clear()

    def test_legacy_api_falls_back_to_per_process_doc_id(self, provider, monkeypatch):
        """API returns no /version (or pre-0.5.0) — sync_turn must use the
        per-process unique doc_id and NOT pass update_mode."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: None,
        )
        old_doc = provider._document_id
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        assert kw["document_id"].startswith("test-session-")
        item = kw["items"][0]
        assert "update_mode" not in item

    def test_modern_api_uses_stable_doc_id_with_append(self, provider, monkeypatch):
        """API on >=0.5.0 — retain uses stable session_id and sets update_mode='append'."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        # Stable: just the session id, no per-process timestamp suffix.
        assert kw["document_id"] == "test-session"
        item = kw["items"][0]
        assert item["update_mode"] == "append"

    def test_version_probe_returns_metadata_through_credential_safe_opener(
        self, monkeypatch
    ):
        from plugins.memory import hindsight as hindsight_mod

        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "version": "0.5.6",
                "features": {"store_document_text": True},
            }
        ).encode()
        response.__enter__.return_value = response
        captured = {}

        def _open(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        monkeypatch.setattr(hindsight_mod, "open_credentialed_url", _open)

        metadata = hindsight_mod._fetch_hindsight_api_version(
            "https://memory.example/api/",
            "secret-token",
            timeout=1.25,
        )

        assert metadata["features"]["store_document_text"] is True
        assert captured["request"].full_url == "https://memory.example/api/version"
        assert captured["request"].get_header("Authorization") == "Bearer secret-token"
        assert captured["timeout"] == 1.25

    def test_modern_api_without_stored_document_text_uses_safe_create(
        self, provider, monkeypatch
    ):
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *args, **kwargs: {
                "version": "0.5.6",
                "features": {"store_document_text": False},
            },
        )

        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kwargs = provider._client.aretain_batch.call_args.kwargs
        assert kwargs["document_id"].startswith("test-session-")
        assert "update_mode" not in kwargs["items"][0]


    def test_session_switch_flush_picks_capability_against_old_session(
        self, provider_with_config, monkeypatch
    ):
        """When the API supports append, the flush on /reset must land
        in the OLD session's stable document, not a per-process id."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        kw = p._client.aretain_batch.call_args.kwargs
        # Flush goes to the OLD session's stable doc, not new-sid's.
        assert kw["document_id"] == "test-session"
        assert kw["items"][0]["update_mode"] == "append"


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_hybrid_mode_prompt(self, provider):
        block = provider.system_prompt_block()
        assert "Hindsight Memory" in block
        assert "hindsight_recall" in block
        assert "automatically injected" in block


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_has_all_new_fields(self, provider):
        schema = provider.get_config_schema()
        keys = {f["key"] for f in schema}
        expected_keys = {
            "mode", "api_url", "api_key", "llm_provider", "llm_api_key",
            "llm_model", "bank_id", "bank_id_template", "bank_mission", "bank_retain_mission",
            "recall_budget", "memory_mode", "recall_prefetch_method",
            "retain_tags", "retain_source",
            "retain_user_prefix", "retain_assistant_prefix",
            "recall_tags", "recall_tags_match",
            "observation_scopes", "observation_scope_exclude_tag_prefixes",
            "auto_recall", "recall_sync", "recall_indicator",
            "auto_retain", "expose_retain_tool", "retain_indicator",
            "retain_every_n_turns", "retain_async", "retain_context",
            "prefetch_waits_for_retain",
            "recall_max_tokens", "recall_max_input_chars",
            "recall_prompt_preamble",
        }
        assert expected_keys.issubset(keys), f"Missing: {expected_keys - keys}"

    def test_declarative_schema_exposes_isolation_and_tool_controls(self):
        from plugins.memory.config_schema import get_provider_config_schema

        schema = get_provider_config_schema("hindsight")
        fields = {field.key: field for field in schema.fields}

        assert fields["bank_id_template"].default == ""
        assert fields["recall_tags"].default == ""
        assert fields["recall_tags"].env_fallbacks == ("HINDSIGHT_RECALL_TAGS",)
        assert fields["observation_scope_exclude_tag_prefixes"].default == ""
        assert fields[
            "observation_scope_exclude_tag_prefixes"
        ].env_fallbacks == (
            "HINDSIGHT_RETAIN_OBSERVATION_SCOPE_EXCLUDE_TAG_PREFIXES",
        )
        assert fields["expose_retain_tool"].default is True


# ---------------------------------------------------------------------------
# bank_id_template tests
# ---------------------------------------------------------------------------


class TestBankIdTemplate:
    def test_sanitize_bank_segment_passthrough(self):
        assert _sanitize_bank_segment("hermes") == "hermes"
        assert _sanitize_bank_segment("my-agent_1") == "my-agent_1"


    def test_resolve_empty_template_uses_fallback(self):
        result = _resolve_bank_id_template(
            "", fallback="hermes", profile="coder"
        )
        assert result == "hermes"


    def test_resolve_sanitizes_placeholder_values(self):
        result = _resolve_bank_id_template(
            "user-{user}", fallback="hermes",
            profile="", workspace="", platform="",
            user="josh@example.com", session="",
        )
        assert result == "user-josh-example-com"


    def test_provider_uses_bank_id_template_from_config(self, tmp_path, monkeypatch):
        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "fallback-bank",
            "bank_id_template": "hermes-{profile}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="coder",
            agent_workspace="hermes",
        )
        assert p._bank_id == "hermes-coder"
        assert p._bank_id_template == "hermes-{profile}"

    def test_session_placeholder_and_static_fallback_are_recomputed_on_switch(
        self, provider_with_config
    ):
        p = provider_with_config(
            bank_id="fallback-bank",
            bank_id_template="hermes-{session}",
        )
        assert p._bank_id == "hermes-test-session"

        p.on_session_switch("next-session")

        assert p._bank_id == "hermes-next-session"
        assert p._bank_id_fallback == "fallback-bank"


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_API_KEY", "test-key")
        p = HindsightMemoryProvider()
        assert p.is_available()


    def test_local_mode_unavailable_when_runtime_import_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_MODE", "local")

        def _raise(_name):
            raise RuntimeError(
                "NumPy was built with baseline optimizations: (x86_64-v2)"
            )

        monkeypatch.setattr(
            "importlib.import_module",
            _raise,
        )
        p = HindsightMemoryProvider()
        assert not p.is_available()

    def test_initialize_disables_local_mode_when_runtime_import_fails(self, tmp_path, monkeypatch):
        config = {"mode": "local_embedded"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        def _raise(_name):
            raise RuntimeError("x86_64-v2 unsupported")

        monkeypatch.setattr(
            "importlib.import_module",
            _raise,
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        assert p._mode == "disabled"


class TestSharedEventLoopLifecycle:
    """Regression tests for #11923 — Hindsight leaking aiohttp ClientSession /
    TCPConnector objects in long-running gateway processes.

    Root cause: the module-global ``_loop`` / ``_loop_thread`` pair is shared
    across every HindsightMemoryProvider instance in the process (the plugin
    loader builds one provider per AIAgent, and the gateway builds one AIAgent
    per concurrent chat session). When a session ended, ``shutdown()`` stopped
    the shared loop, which orphaned every *other* live provider's aiohttp
    ClientSession on a dead loop. Those sessions were never closed and surfaced
    as ``Unclosed client session`` / ``Unclosed connector`` errors.
    """

    def test_shutdown_does_not_stop_shared_event_loop(self, provider_with_config):
        from plugins.memory import hindsight as hindsight_mod

        async def _noop():
            return 1

        # Prime the shared loop by scheduling a trivial coroutine — mirrors
        # the first time any real async call (arecall/aretain/areflect) runs.
        assert hindsight_mod._run_sync(_noop()) == 1

        loop_before = hindsight_mod._loop
        thread_before = hindsight_mod._loop_thread
        assert loop_before is not None and loop_before.is_running()
        assert thread_before is not None and thread_before.is_alive()

        # Build two independent providers (two concurrent chat sessions).
        provider_a = provider_with_config()
        provider_b = provider_with_config()

        # End session A.
        provider_a.shutdown()

        # Module-global loop/thread must still be the same live objects —
        # provider B (and any other sibling provider) is still relying on them.
        assert hindsight_mod._loop is loop_before, (
            "shutdown() swapped out the shared event loop — sibling providers "
            "would have their aiohttp ClientSession orphaned (#11923)"
        )
        assert hindsight_mod._loop.is_running(), (
            "shutdown() stopped the shared event loop — sibling providers' "
            "aiohttp sessions would leak (#11923)"
        )
        assert hindsight_mod._loop_thread is thread_before
        assert hindsight_mod._loop_thread.is_alive()

        # Provider B can still dispatch async work on the shared loop.
        async def _still_working():
            return 42

        assert hindsight_mod._run_sync(_still_working()) == 42

        provider_b.shutdown()

    def test_client_aclose_called_on_cloud_mode_shutdown(self, provider):
        """Per-provider session cleanup still runs even though the shared
        loop is preserved. Each provider's own aiohttp session is closed
        via ``self._client.aclose()``; only the (empty) shared loop survives.
        """
        assert provider._client is not None
        mock_client = provider._client

        provider.shutdown()

        mock_client.aclose.assert_called_once()
        assert provider._client is None


class TestShutdown:
    def test_local_embedded_shutdown_closes_inner_async_client_on_shared_loop(self, provider):
        inner_client = _make_mock_client()
        embedded = MagicMock()
        embedded._client = inner_client
        embedded.close = MagicMock()

        provider._mode = "local_embedded"
        provider._client = embedded

        provider.shutdown()

        inner_client.aclose.assert_awaited_once()
        embedded.close.assert_called_once()
        assert embedded._client is None
        assert provider._client is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """hindsight/config.json must be written with 0o600 so API key is not world-readable."""
    provider = HindsightMemoryProvider()
    provider.save_config({"api_key": "hd-test-key"}, str(tmp_path))
    config_file = tmp_path / "hindsight" / "config.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"


class TestLoadSimpleEnv:
    def test_bom_first_key_is_recognized(self, tmp_path):
        """A Notepad-edited .env carries a BOM; the first key must still parse
        instead of becoming '\ufeffHINDSIGHT_LLM_API_KEY'."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿HINDSIGHT_LLM_API_KEY=sk-test\n".encode("utf-8"))
        values = _load_simple_env(env_path)
        assert values.get("HINDSIGHT_LLM_API_KEY") == "sk-test"


class TestPostSetupEnvEncoding:
    def _run_cloud_post_setup(self, tmp_path, monkeypatch):
        """Drive post_setup through the cloud path with piped stdin."""
        import io

        monkeypatch.setattr("hermes_cli.memory_setup._curses_select",
                            lambda *a, **kw: 0)  # cloud mode
        monkeypatch.setattr("hermes_cli.config.save_config", lambda c: None)
        # Skip the dependency install (now routed through lazy_deps, NS-605).
        import tools.lazy_deps as lazy_deps_mod
        monkeypatch.setattr(
            lazy_deps_mod, "install_specs",
            lambda *a, **kw: lazy_deps_mod.InstallSpecsResult(ok=True),
        )
        # First line: API key prompt (readline). Second line: API URL (input).
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-new\n\n"))

        provider = HindsightMemoryProvider()
        provider.post_setup(str(tmp_path), {"memory": {}})

    def test_bom_first_key_updated_in_place(self, tmp_path, monkeypatch):
        """The setup writer reads the existing .env BOM-tolerantly, so a
        BOM'd first key is matched and rewritten, not duplicated."""
        env_path = tmp_path / ".env"
        env_path.write_bytes("﻿HINDSIGHT_API_KEY=old\n".encode("utf-8"))

        self._run_cloud_post_setup(tmp_path, monkeypatch)

        content = env_path.read_text(encoding="utf-8")
        assert content.count("HINDSIGHT_API_KEY=") == 1
        assert "HINDSIGHT_API_KEY=sk-new" in content
        assert "old" not in content
        assert "﻿" not in content


class TestClientAutoUpgradeRoutesThroughLazyDeps:
    """The initialize()-time hindsight-client auto-upgrade must go through
    lazy_deps.install_specs() (environment-aware, durable-target on sealed
    hosted venvs) — never a direct `uv pip install --python sys.executable`
    subprocess, which fails with EROFS/EACCES on immutable images (NS-605)."""

    def _init_with_outdated_client(self, tmp_path, monkeypatch, outcome):
        import importlib.metadata as md
        import subprocess as subprocess_mod
        import tools.lazy_deps as lazy_deps_mod

        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"mode": "cloud"}))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        # Simulate an installed-but-outdated client.
        monkeypatch.setattr(md, "version", lambda name: "0.0.1")

        calls = []
        monkeypatch.setattr(
            lazy_deps_mod, "install_specs",
            lambda specs, **kw: calls.append(tuple(specs)) or outcome,
        )

        # Regression guard: no direct pip subprocess may run.
        def _no_subprocess(*a, **kw):  # pragma: no cover - fails loudly
            raise AssertionError(f"unexpected subprocess.run during auto-upgrade: {a}")
        monkeypatch.setattr(subprocess_mod, "run", _no_subprocess)

        provider = HindsightMemoryProvider()
        provider.initialize(session_id="s", hermes_home=str(tmp_path), platform="cli")
        return calls

    def test_upgrade_uses_install_specs_not_subprocess(self, tmp_path, monkeypatch):
        from plugins.memory.hindsight import _MIN_CLIENT_VERSION
        from tools.lazy_deps import InstallSpecsResult

        calls = self._init_with_outdated_client(
            tmp_path, monkeypatch, InstallSpecsResult(ok=True)
        )
        assert calls == [(f"hindsight-client>={_MIN_CLIENT_VERSION}",)]

    def test_blocked_upgrade_is_nonfatal_and_surfaces_reason(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging
        from tools.lazy_deps import InstallSpecsResult

        with caplog.at_level(logging.WARNING):
            calls = self._init_with_outdated_client(
                tmp_path, monkeypatch,
                InstallSpecsResult(ok=False, blocked=True,
                                   reason="runtime installs are disabled on this deployment"),
            )
        assert len(calls) == 1  # attempted exactly once, init still completed
        assert any("runtime installs are disabled" in r.getMessage()
                   for r in caplog.records)



class TestMultiplexBackgroundScope:
    """Under multiplex_profiles get_secret fails closed on an unscoped thread;
    the writer / daemon-start threads are spawned from a scoped context and
    must carry it along (#92608, #94933)."""

    @pytest.fixture()
    def scoped_embedded(self, tmp_path, monkeypatch):
        from agent.secret_scope import (
            build_profile_secret_scope, reset_secret_scope, set_multiplex_active, set_secret_scope,
        )
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        created = []

        class FakeHindsightEmbedded:
            def __init__(self, **kwargs):
                created.append(kwargs["llm_api_key"])
                self._manager = SimpleNamespace(is_running=lambda profile: False, stop=lambda profile: None)
                self._ensure_started = lambda: None

        dem = SimpleNamespace(console=None)
        monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeHindsightEmbedded))
        monkeypatch.setitem(sys.modules, "hindsight_embed", SimpleNamespace(daemon_embed_manager=dem))
        monkeypatch.setitem(sys.modules, "hindsight_embed.daemon_embed_manager", dem)
        monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

        home = tmp_path / "profiles" / "p1"
        (home / "hindsight").mkdir(parents=True)
        (home / ".env").write_text("HINDSIGHT_LLM_API_KEY=p1-secret\n")
        (home / "hindsight" / "config.json").write_text(json.dumps(
            {"mode": "local_embedded", "llm_provider": "openai", "llm_model": "m", "memory_mode": "hybrid"}
        ))
        # Enter the profile scope the way gateway _profile_runtime_scope does.
        set_multiplex_active(True)
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: home)
        home_tok = set_hermes_home_override(str(home))
        scope_tok = set_secret_scope(build_profile_secret_scope(home))
        yield created, home
        set_multiplex_active(False)
        reset_secret_scope(scope_tok)
        reset_hermes_home_override(home_tok)

    def test_writer_thread_resolves_profile_secret(self, scoped_embedded):
        created, home = scoped_embedded
        p = HindsightMemoryProvider()
        p._mode = "local_embedded"
        p._config = {"profile": "hermes", "llm_provider": "openai", "llm_model": "m"}
        p._ensure_writer()
        p._retain_queue.put(p._get_client)   # real body: get_secret(HINDSIGHT_LLM_API_KEY)
        p._retain_queue.put(_WRITER_SENTINEL)
        p._writer_thread.join(timeout=5)
        assert created == ["p1-secret"]

    def test_daemon_start_thread_resolves_profile_secret(self, scoped_embedded):
        created, home = scoped_embedded
        p = HindsightMemoryProvider()
        p.initialize(session_id="s1", hermes_home=str(home), platform="cli")
        for t in threading.enumerate():
            if t.name == "hindsight-daemon-start":
                t.join(timeout=5)
        assert created == ["p1-secret"]
        assert "Daemon started successfully" in (home / "logs" / "hindsight-embed.log").read_text()
