"""Credential port contracts through real local stores and client construction."""

import os
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


def _configure_store(tmp_path, monkeypatch, provider):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        f"credential_pool_strategies:\n  {provider}: least_used\n"
    )
    from hermes_cli.auth import write_credential_pool

    write_credential_pool(provider, [
        {
            "id": entry_id, "label": entry_id, "auth_type": "api_key",
            "source": "manual", "priority": index,
            "access_token": f"local-test-token-{entry_id}",
            "base_url": "https://credential-port.invalid/v1",
        }
        for index, entry_id in enumerate(("a", "b"))
    ])
    return hermes_home


def test_independent_processes_share_durable_selection_counts(tmp_path, monkeypatch):
    _configure_store(tmp_path, monkeypatch, "openrouter")
    from hermes_cli.auth import read_credential_pool

    code = """
from agent.credential_pool import load_pool
pool = load_pool('openrouter')
for _ in range(6):
    assert pool.select() is not None
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    try:
        for child in children:
            child.communicate(timeout=20)
            assert child.returncode == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    counts = {row["id"]: row["request_count"] for row in read_credential_pool("openrouter")}
    assert counts == {"a": 6, "b": 6}


@pytest.mark.parametrize("remove_root_provider", [False, True])
def test_borrowed_pool_selection_updates_only_the_owning_store(
    tmp_path, monkeypatch, remove_root_provider,
):
    root = _configure_store(tmp_path, monkeypatch, "anthropic")
    profile = root / "profiles" / "borrower"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("credential_pool_strategies:\n  anthropic: least_used\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.has_credentials()
    if remove_root_provider:
        root_store = json.loads((root / "auth.json").read_text())
        del root_store["credential_pool"]["anthropic"]
        (root / "auth.json").write_text(json.dumps(root_store))
    selected = pool.select()
    assert (selected is None) is remove_root_provider
    local_file = profile / "auth.json"
    local_store = json.loads(local_file.read_text()) if local_file.exists() else {}
    assert "anthropic" not in local_store.get("credential_pool", {})
    root_store = json.loads((root / "auth.json").read_text())
    if remove_root_provider:
        assert "anthropic" not in root_store["credential_pool"]
    else:
        assert sum(row.get("request_count", 0) for row in root_store["credential_pool"]["anthropic"]) == 1


@pytest.mark.parametrize("provider", ["openrouter", "deepinfra"])
def test_real_cached_client_uses_selected_disk_revision(tmp_path, monkeypatch, provider):
    _configure_store(tmp_path, monkeypatch, provider)
    from agent import auxiliary_client as aux
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    rows = read_credential_pool(provider)
    rows[1]["request_count"] = 100
    write_credential_pool(provider, rows)
    clients = []
    aux.shutdown_cached_clients()
    try:
        first, _ = aux._get_cached_client(provider, "local-test-model")
        clients.append(first)
        assert first._hermes_pool_entry_id == "a"
        assert first.api_key == "local-test-token-a"
        assert str(first.base_url).rstrip("/") == "https://credential-port.invalid/v1"
        rows = read_credential_pool(provider)
        selected = next(row for row in rows if row["id"] == "a")
        assert selected["request_count"] == 1
        selected["access_token"] = "local-test-refreshed-a"
        write_credential_pool(provider, rows)

        second, _ = aux._get_cached_client(provider, "local-test-model")
        clients.append(second)
        assert second is not first
        assert second._hermes_pool_entry_id == first._hermes_pool_entry_id
        assert second.api_key == "local-test-refreshed-a"
        assert second._hermes_pool_cache_revision != first._hermes_pool_cache_revision
        assert not first.is_closed()
        rows = read_credential_pool(provider)
        assert {row["id"]: row["request_count"] for row in rows} == {"a": 2, "b": 100}
    finally:
        aux.shutdown_cached_clients()
        for client in clients:
            if client is not None:
                client.close()


@pytest.mark.parametrize("pool_state", ["absent", "empty", "unrefreshable"])
def test_exact_identity_refresh_cannot_spend_another_singleton(pool_state):
    from agent import auxiliary_client as aux

    pool = None if pool_state == "absent" else MagicMock()
    if pool is not None:
        pool.has_credentials.return_value = pool_state != "empty"
        pool.runtime_api_key_matches.return_value = True
        pool.try_refresh_matching.return_value = None
    singleton = MagicMock()
    with (
        patch("agent.auxiliary_client.load_pool", return_value=pool),
        patch.dict(aux._CREDENTIAL_REFRESHERS, {"openai-codex": singleton}),
    ):
        assert not aux._refresh_provider_credentials(
            "openai-codex", failed_api_key="local-test-token", failed_entry_id="removed-id",
        )
    singleton.assert_not_called()
