"""Explicit endpoints and frozen auxiliary credential selections compose (#121359)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.auxiliary_client as aux
import hermes_yaml as yaml
from hermes_cli.auth import write_credential_pool


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
def test_explicit_endpoint_wins_over_pool_without_reselecting(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(aux, "_client_cache", {})
    endpoint = f"http://127.0.0.1:9002/{provider}"
    write_credential_pool(provider, [{
        "id": "selected-entry", "label": "selected", "auth_type": "api_key",
        "source": "manual", "priority": 1, "access_token": "sk-test-selected",
        "base_url": f"http://127.0.0.1:9003/{provider}",
    }])

    def openai_client(**kwargs):
        return SimpleNamespace(api_key=kwargs["api_key"], base_url=kwargs["base_url"])

    with (
        patch.object(aux, "_select_pool_entry", wraps=aux._select_pool_entry) as select,
        patch.object(aux, "_create_openai_client", side_effect=openai_client),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=SimpleNamespace()) as anthropic,
    ):
        client, _ = aux._get_cached_client(provider, "test-model", base_url=endpoint)

    select.assert_called_once_with(provider)
    assert client.api_key == "sk-test-selected"
    assert client.base_url == endpoint
    assert client._hermes_pool_entry_id == "selected-entry"
    cache_key = next(iter(aux._client_cache))
    assert cache_key.pool_hint == client._hermes_pool_cache_revision
    if provider == "anthropic":
        anthropic.assert_called_once_with("sk-test-selected", endpoint)


@pytest.mark.parametrize("source", ["env:ACTUAL_API_KEY", "manual"])
def test_actual_config_override_preserves_manual_pool_endpoint(tmp_path, monkeypatch, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configured_endpoint = "http://127.0.0.1:9002/v1"
    pool_endpoint = "http://127.0.0.1:9003/v1"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "aci", "default": "test-model", "base_url": configured_endpoint},
    }))
    monkeypatch.setenv("ACTUAL_API_KEY", "sk-test-selected")
    monkeypatch.setenv("ACTUAL_BASE_URL", pool_endpoint)
    write_credential_pool("actual", [{
        "id": "selected-entry", "label": "selected", "auth_type": "api_key",
        "source": source, "priority": 1, "access_token": "sk-test-selected",
        "base_url": pool_endpoint,
    }])

    selected = aux.auxiliary_pool.select_pool_context("actual")

    assert selected.api_key == "sk-test-selected"
    assert selected.base_url == (configured_endpoint if source.startswith("env:") else pool_endpoint)
