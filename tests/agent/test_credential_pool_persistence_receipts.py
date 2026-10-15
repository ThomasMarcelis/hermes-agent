"""Persistence receipts retain durable selection and concurrent token generations."""

from pathlib import Path

import pytest

from agent import credential_pool as pool_module
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        "credential_pool_strategies:\n  openrouter: least_used\n"
    )
    write_credential_pool("openrouter", [{
        "id": "owned-row", "label": "fixture", "source": "manual",
        "auth_type": "api_key", "access_token": "initial-access",
        "refresh_token": "initial-refresh", "priority": 0,
    }])
    return pool_module.load_pool("openrouter")


def _disk_pair():
    row, = read_credential_pool("openrouter")
    return row["access_token"], row["refresh_token"]


def test_durable_selection_advances_the_base_for_next_local_refresh(pool):
    row, = read_credential_pool(pool.provider)
    row.update(access_token="peer-access", refresh_token="peer-refresh")
    write_credential_pool(pool.provider, [row])

    selected = pool.select()
    assert selected.access_token == "peer-access"
    assert selected.request_count == 1
    refreshed = pool._adopt(
        selected, access_token="local-access", refresh_token="local-refresh",
    )
    assert refreshed.access_token == "local-access"
    assert _disk_pair() == ("local-access", "local-refresh")

    # The same pool must still adopt a later peer generation on an ordinary flush.
    row, = read_credential_pool(pool.provider)
    row.update(access_token="later-peer-access", refresh_token="later-peer-refresh")
    write_credential_pool(pool.provider, [row])
    pool._persist()
    assert pool.current().access_token == "later-peer-access"
    assert _disk_pair() == ("later-peer-access", "later-peer-refresh")


@pytest.mark.parametrize("persist_refresh", [False, True])
def test_delayed_receipt_cannot_replace_a_newer_in_process_refresh(
    pool, monkeypatch, persist_refresh,
):
    pool.select()
    row, = read_credential_pool(pool.provider)
    row.update(access_token="peer-access", refresh_token="peer-refresh")
    write_credential_pool(pool.provider, [row])
    persist = pool_module.persist_pool_entries

    def delayed_receipt(*args, **kwargs):
        receipt = persist(*args, **kwargs)
        # A refresh completes after the store transaction but before its receipt
        # is adopted. No scheduling assumptions are needed to exercise the race.
        monkeypatch.setattr(pool_module, "persist_pool_entries", persist)
        if persist_refresh:
            # The refresher observes the peer generation before rotating it.
            pool._persist()
        pool._adopt(
            pool.current(), persist=persist_refresh,
            access_token="local-access", refresh_token="local-refresh",
        )
        return receipt

    monkeypatch.setattr(pool_module, "persist_pool_entries", delayed_receipt)
    pool._persist()
    assert pool.current().access_token == "local-access"
    assert pool.current().refresh_token == "local-refresh"
    pool._persist()
    assert _disk_pair() == ("local-access", "local-refresh")
