"""Frozen Codex selection keeps credentials and their endpoint paired on every client path."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("raw", [False, True])
def test_frozen_pool_selection_keeps_configured_route_on_raw_and_wrapped_clients(tmp_path, monkeypatch, raw):
    from agent import auxiliary_client as aux
    from agent.auxiliary_pool import SelectedPoolContext

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    gateway = "https://selected.example/backend-api/codex"
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai-codex\n  base_url: " + gateway + "\n")
    selection = SelectedPoolContext(provider="openai-codex", pool_present=True, entry_id="chosen",
                                    api_key="frozen-token", base_url="https://chatgpt.com/backend-api/codex")
    monkeypatch.setattr(aux, "_select_pool_entry", lambda provider: pytest.fail("credential reselected"))
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(api_key=kwargs["api_key"], base_url=kwargs["base_url"])
    monkeypatch.setattr(aux, "_create_openai_client", create)
    client, model = aux.resolve_provider_client("openai-codex", "gpt-6-sol", raw_codex=raw,
                                               selected_pool=selection)
    assert client is not None and model == "gpt-6-sol"
    assert len(calls) == 1
    assert (calls[0]["api_key"], calls[0]["base_url"]) == ("frozen-token", gateway)
    assert aux._resolve_codex_credential_and_base(selection) == ("frozen-token", gateway)
