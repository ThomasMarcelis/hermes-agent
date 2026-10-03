"""Regression coverage for implicit live-runtime auxiliary cache keys.

#49151/#49156 is specifically the ``provider='auto'`` path where callers omit
``main_runtime`` after a mid-session model switch.  This is distinct from
#56889, which isolates callers that pass different explicit ``model=`` values.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as aux


@pytest.fixture(autouse=True)
def _clean_aux_state():
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    yield
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()


def _runtime(model: str, *, provider: str = "custom:llama-swap") -> dict:
    return {
        "provider": provider,
        "model": model,
        "base_url": "http://llama-swap.test/v1",
        "api_key": "local-key",
        "api_mode": "chat_completions",
        "auth_mode": "api_key",
    }




def test_implicit_runtime_cache_key_covers_full_connection_and_auth_surface():
    """Provider/endpoint/credential/wire/auth changes all isolate auto clients."""
    base = _runtime("same-model")
    variants = [
        {**base, "provider": "custom:other"},
        {**base, "base_url": "https://other.test/v1"},
        {**base, "api_key": "other-key"},
        {**base, "api_mode": "codex_responses"},
        {**base, "auth_mode": "entra_id", "api_key": lambda: "token"},
    ]

    aux.set_runtime_main(**base)
    baseline = aux._client_cache_key("auto", async_mode=False)
    keys = []
    for variant in variants:
        aux.set_runtime_main(**variant)
        keys.append(aux._client_cache_key("auto", async_mode=False))

    assert all(key != baseline for key in keys)
    assert len(set(keys)) == len(keys)






def test_runtime_context_token_restores_previous_value_after_turn():
    """Turn-scoped runtime binding must not leak into later work in the same context."""
    token = aux.set_runtime_main(**_runtime("turn-model"))
    assert aux._normalize_main_runtime(None)["model"] == "turn-model"

    aux.reset_runtime_main(token)

    assert aux._normalize_main_runtime(None) == {}


















def test_unhashable_callable_runtime_api_keys_are_safe_secret_free_discriminators():
    """Callable token providers remain cacheable without leaking returned tokens."""

    class TokenProvider(list):
        def __init__(self, token: str):
            super().__init__()
            self.token = token

        def __call__(self) -> str:
            return self.token

    first_provider = TokenProvider("first-super-secret-token")
    second_provider = TokenProvider("second-super-secret-token")

    first = aux._client_cache_key(
        "auto", async_mode=False, main_runtime={**_runtime("same"), "api_key": first_provider}
    )
    second = aux._client_cache_key(
        "auto", async_mode=False, main_runtime={**_runtime("same"), "api_key": second_provider}
    )

    hash(first)
    hash(second)
    assert first != second
    rendered = repr((first, second))
    assert "first-super-secret-token" not in rendered
    assert "second-super-secret-token" not in rendered


def test_string_api_keys_are_not_retained_in_cache_key_repr():
    """String credentials discriminate clients without living in cache-key memory."""
    first_secret = "first-literal-super-secret"
    second_secret = "second-literal-super-secret"
    first = aux._client_cache_key(
        "auto",
        async_mode=False,
        api_key=first_secret,
        main_runtime={**_runtime("same"), "api_key": first_secret},
    )
    second = aux._client_cache_key(
        "auto",
        async_mode=False,
        api_key=second_secret,
        main_runtime={**_runtime("same"), "api_key": second_secret},
    )

    assert first != second
    rendered = repr((first, second))
    assert first_secret not in rendered
    assert second_secret not in rendered




def test_client_cache_key_is_scoped_per_profile_home(tmp_path):
    """Callers that omit api_key (pool / Nous auth.json paths) must not share a client across
    multiplex profiles: the per-turn HERMES_HOME override has to participate in the key."""
    import hermes_constants

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    keys = []
    for home in (a, b):
        tok = hermes_constants.set_hermes_home_override(str(home))
        try:
            keys.append(aux._client_cache_key("nous", async_mode=False, base_url="https://inf.example", model="m"))
        finally:
            hermes_constants.reset_hermes_home_override(tok)
    assert keys[0] != keys[1]


def test_pool_revision_pruning_is_scoped_to_profile_home(tmp_path):
    """The same provider/entry id in another profile owns an independent cache entry."""
    from agent.auxiliary_pool import SelectedPoolContext
    import hermes_constants

    def key(home, revision):
        home.mkdir(exist_ok=True)
        token = hermes_constants.set_hermes_home_override(str(home))
        try:
            return aux._client_cache_key(
                "openrouter", async_mode=False, model="test/model",
                selected_pool=SelectedPoolContext(
                    provider="openrouter", pool_present=True, entry_id="shared-id",
                    cache_revision=f"openrouter:shared-id:{revision}",
                ),
            )
        finally:
            hermes_constants.reset_hermes_home_override(token)

    profile_a_old = key(tmp_path / "profile-a", "old")
    profile_a_fresh = key(tmp_path / "profile-a", "fresh")
    profile_b_old = key(tmp_path / "profile-b", "old")
    profile_a_client = MagicMock(name="profile-a-old")
    profile_b_client = MagicMock(name="profile-b-old")
    fresh_client = MagicMock(name="profile-a-fresh")

    aux._store_cached_client(profile_a_old, profile_a_client, "test/model")
    aux._store_cached_client(profile_b_old, profile_b_client, "test/model")
    aux._store_cached_client(profile_a_fresh, fresh_client, "test/model")

    assert profile_a_old not in aux._client_cache
    assert aux._client_cache[profile_a_fresh][0] is fresh_client
    assert aux._client_cache[profile_b_old][0] is profile_b_client
    profile_a_client.close.assert_not_called()
    profile_b_client.close.assert_not_called()


def test_explicit_refresh_eviction_is_scoped_to_profile_home(tmp_path):
    """A refresh evicts only the owning profile and never closes in-flight clients."""
    import hermes_constants

    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in homes:
        home.mkdir()
    keys = []
    for home in homes:
        token = hermes_constants.set_hermes_home_override(str(home))
        try:
            keys.append(aux._client_cache_key("vertex", async_mode=False, model="test/model"))
        finally:
            hermes_constants.reset_hermes_home_override(token)

    clients = [MagicMock(name="profile-a"), MagicMock(name="profile-b")]
    aux._client_cache.update({
        keys[0]: (clients[0], "test/model", None),
        keys[1]: (clients[1], "test/model", None),
    })

    token = hermes_constants.set_hermes_home_override(str(homes[0]))
    try:
        aux._evict_cached_clients("vertex")
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert keys[0] not in aux._client_cache
    assert aux._client_cache[keys[1]][0] is clients[1]
    clients[0].close.assert_not_called()
    clients[1].close.assert_not_called()


def test_pruned_pool_client_is_not_closed_during_in_flight_request():
    """Cache retirement must not close a transport still owned by its caller."""
    from agent.auxiliary_pool import SelectedPoolContext

    def key(revision):
        return aux._client_cache_key(
            "openrouter", async_mode=False, model="test/model",
            selected_pool=SelectedPoolContext(
                provider="openrouter", pool_present=True, entry_id="primary",
                cache_revision=f"openrouter:primary:{revision}",
            ),
        )

    old_key, fresh_key = key("old"), key("fresh")
    request_started = Event()
    release_request = Event()

    class InUseClient:
        def __init__(self):
            self.closed = False

        def request(self):
            request_started.set()
            assert release_request.wait(timeout=2.0)
            assert not self.closed

        def close(self):
            self.closed = True

    old_client = InUseClient()
    aux._store_cached_client(old_key, old_client, "test/model")
    request = Thread(target=old_client.request)
    request.start()
    assert request_started.wait(timeout=2.0)

    aux._store_cached_client(fresh_key, MagicMock(), "test/model")

    assert old_key not in aux._client_cache
    assert not old_client.closed
    release_request.set()
    request.join(timeout=2.0)
    assert not request.is_alive()
    assert not old_client.closed
