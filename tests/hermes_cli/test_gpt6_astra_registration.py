"""Owned local Codex Astra contracts alongside upstream official API behavior."""

import json

import httpx
import pytest


LOCAL_CODEX = "http://127.0.0.1:2455/backend-api/codex"


def _configure_codex(tmp_path, monkeypatch, base_url=LOCAL_CODEX):
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-6-astra\n  base_url: " + base_url + "\n"
    )
    return home


@pytest.mark.parametrize("source", ["offline", "cache", "stale_api"])
def test_local_astra_selectable_with_verified_context(source, tmp_path, monkeypatch):
    from agent.model_metadata import get_model_context_length
    from hermes_cli.models import provider_model_ids
    from hermes_cli.models_validate import validate_requested_model

    _configure_codex(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda **kwargs: {"api_key": "test-token"} if source == "stale_api" else {},
    )
    if source == "cache":
        (tmp_path / "models_cache.json").write_text(json.dumps({"models": [{"slug": "gpt-5.6-sol"}]}))
    calls = []

    def discover(url, **kwargs):
        calls.append(str(url))
        assert str(url) == "http://127.0.0.1:2455/v1/models"
        return httpx.Response(200, json={"data": [{"id": "gpt-5.6-sol"}]})

    monkeypatch.setattr(httpx, "get", discover)
    model = "gpt-6-astra"
    choices = provider_model_ids("openai-codex")
    assert model in choices and choices.count(model) == 1
    selected = validate_requested_model(model, "openai-codex")
    assert selected["accepted"] and selected["recognized"]
    assert not selected.get("corrected_model")
    assert get_model_context_length(model, provider="openai-codex") == 272_000
    assert (model + "-900k") in choices  # upstream explicit large-context opt-in remains available
    assert bool(calls) == (source == "stale_api")


def test_local_catalog_state_cannot_leak_into_official_account(tmp_path, monkeypatch):
    from hermes_cli import models
    from hermes_cli.codex_models import get_codex_model_ids

    home = _configure_codex(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", lambda **kwargs: {})
    local_fp = models._credential_fingerprint("openai-codex")
    models.update_provider_cache_entry("openai-codex", ["gpt-5.6-sol", "gpt-6-astra"])
    assert not models._model_requires_account_discovery("openai-codex", "gpt-6-astra")
    assert models._model_requires_account_discovery("openai-api", "gpt-6-astra")
    (tmp_path / "config.toml").write_text('model = "gpt-6-astra"\n')
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": [{"slug": "gpt-6-astra"}]}))
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  base_url: https://chatgpt.com/backend-api/codex\n"
    )
    assert models._credential_fingerprint("openai-codex") != local_fp
    assert models._model_requires_account_discovery("openai-codex", "gpt-6-astra")
    assert not any("astra" in model for model in get_codex_model_ids())
    assert not any("astra" in model for model in models.cached_provider_model_ids("openai-codex"))


def test_codex_setup_keeps_the_selected_local_catalog_route(tmp_path, monkeypatch):
    from hermes_cli import model_setup_flows as flows

    _configure_codex(tmp_path, monkeypatch)
    monkeypatch.setattr(flows, "_oauth_gate", lambda *args, **kwargs: True)
    monkeypatch.setattr("hermes_cli.auth.get_codex_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", lambda **kwargs: {})
    selected = {}

    def choose(models, **kwargs):
        assert "gpt-6-astra" in models
        selected.update(kwargs)
        return "gpt-6-astra"

    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", choose)
    monkeypatch.setattr(flows, "_activate_provider_model", lambda *args: selected.update(activated=args))
    flows._model_flow_openai_codex({}, current_model="gpt-6-astra")
    assert selected["confirm_base_url"] == LOCAL_CODEX
    assert selected["activated"][:3] == ("gpt-6-astra", "openai-codex", LOCAL_CODEX)


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra"])
def test_astra_context_fallback_keeps_provider_limits(model, monkeypatch):
    from agent.model_metadata import get_model_context_length

    monkeypatch.setattr("agent.model_metadata.get_cached_context_length", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.model_metadata._resolve_endpoint_context_length", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.model_metadata._probe_local_context_length", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.model_metadata._query_ollama_api_show", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda: {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda **kwargs: {})
    assert get_model_context_length(model, provider="openai-codex", base_url=LOCAL_CODEX) == 272_000
    assert get_model_context_length(model) == 1_050_000


def test_local_astra_live_context_overrides_offline_fallback(monkeypatch):
    from agent.model_metadata import get_model_context_length

    monkeypatch.setattr("agent.model_metadata.get_cached_context_length", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.model_metadata._resolve_endpoint_context_length", lambda *args, **kwargs: 192_000)
    assert get_model_context_length("gpt-6-astra", provider="openai-codex", base_url=LOCAL_CODEX) == 192_000


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra"])
def test_astra_offline_capabilities_route_images_natively(model, monkeypatch):
    from agent.image_routing import decide_image_input_mode
    from agent.models_dev import get_model_capabilities, get_model_info

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda **kwargs: {})
    caps = get_model_capabilities("openai-codex", model)
    info = get_model_info("openai-codex", model)
    assert caps is not None and info is not None
    assert caps.supports_tools and caps.supports_vision and caps.supports_reasoning
    assert caps.context_window == info.context_window == 272_000
    assert info.input_modalities == ("text", "image")
    assert info.output_modalities == ("text",)
    assert not info.has_cost_data() and info.max_output == 0
    assert decide_image_input_mode("openai-codex", model, {}) == "native"
    direct = get_model_info("openai", model)
    assert direct.context_window > info.context_window  # verified public API metadata stays separate


@pytest.mark.parametrize("source", ["catalog", "override"])
def test_astra_metadata_fallback_yields_to_resolved_metadata(source, monkeypatch):
    from agent.models_dev import get_model_capabilities, get_model_info

    model = "gpt-6-astra"
    catalog = {"openai": {"models": {model: {
        "limit": {"context": 372000},
        "modalities": {"input": ["text"]},
    }}}} if source == "catalog" else {}
    overrides = {"openai-codex": {model: {
        "context_window": 372000, "supports_vision": False,
    }}} if source == "override" else {}
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda **kwargs: catalog)
    monkeypatch.setattr("agent.models_dev._load_model_overrides", lambda: overrides)
    caps = get_model_capabilities("openai-codex", model)
    info = get_model_info("openai-codex", model)
    assert caps.context_window == info.context_window == 372_000
    assert not caps.supports_vision and not info.supports_vision()


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", "ultra"])
def test_astra_responses_preserves_supported_effort_and_capabilities(effort):
    import agent.transports.codex  # noqa: F401
    from agent.transports import get_transport

    image_url = "data:image/png;base64,aW1hZ2U="
    kwargs = get_transport("codex_responses").build_kwargs(
        model="gpt-6-astra",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        tools=[{"type": "function", "function": {
            "name": "terminal", "parameters": {"type": "object", "properties": {}},
        }}],
        provider="openai-codex", base_url=LOCAL_CODEX, is_codex_backend=True,
        reasoning_config={"effort": effort},
    )
    assert kwargs["model"] == "gpt-6-astra"
    assert kwargs["reasoning"] == {"effort": effort, "summary": "auto"}
    assert "reasoning.encrypted_content" in kwargs["include"]
    assert kwargs["tools"][0]["name"] == "terminal"
    assert kwargs["parallel_tool_calls"] is True
    assert any(
        part.get("type") == "input_image" and part.get("image_url") == image_url
        for item in kwargs["input"] for part in item.get("content", [])
    )


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_astra_unsupported_effort_uses_lowest_supported_level(effort):
    from agent.reasoning_effort import clamp_effort, codex_supported_efforts

    assert clamp_effort(effort, codex_supported_efforts("openai/gpt-6-astra")) == "low"


@pytest.mark.parametrize("base_url,expected", [
    (LOCAL_CODEX, "ultra"),
    ("https://chatgpt.com/backend-api/codex", "ultra"),
    ("https://api.openai.com/v1", "max"),
    ("https://responses.example.com/v1", "xhigh"),
    ("http://127.0.0.1.evil.example:2455/backend-api/codex", "xhigh"),
])
def test_astra_ultra_respects_endpoint_contract(base_url, expected):
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-6-astra", messages=[{"role": "user", "content": "Hi"}], tools=[],
        base_url=base_url,
        is_codex_backend=base_url == "https://chatgpt.com/backend-api/codex",
        reasoning_config={"effort": "ultra"},
    )
    assert kwargs["reasoning"]["effort"] == expected
