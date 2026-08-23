"""Immutable pooled-credential selection for auxiliary client construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Any


@dataclass(frozen=True)
class SelectedPoolContext:
    """One selected credential identity and runtime revision."""

    provider: str
    pool_present: bool
    entry_id: str = ""
    entry_label: str = ""
    api_key: str = ""
    base_url: str = ""
    cache_revision: str = ""


def select_pool_context(provider: str) -> SelectedPoolContext:
    """Select once and freeze the identity used by cache key and resolver."""
    # Late import is intentional: auxiliary_client is the facade whose
    # credential-loading bindings tests and provider integrations patch.
    from agent import auxiliary_client as aux

    pool_present, entry = aux._select_pool_entry(provider)
    if entry is None:
        return SelectedPoolContext(provider=provider, pool_present=pool_present)
    entry_id = str(getattr(entry, "id", "") or "").strip()
    api_key = aux._pool_runtime_api_key(entry)
    base_url = aux._pool_runtime_base_url(entry)
    if provider == "xai-oauth":
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL, _xai_validate_inference_base_url

        base_url = _xai_validate_inference_base_url(
            str(os.getenv("HERMES_XAI_BASE_URL", "") or "").strip().rstrip("/")
            or str(os.getenv("XAI_BASE_URL", "") or "").strip().rstrip("/")
            or base_url,
            fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )
    revision = ""
    if entry_id:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(api_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(base_url.encode("utf-8"))
        revision = f"{provider}:{entry_id}:{digest.hexdigest()}"
    return SelectedPoolContext(
        provider=provider,
        pool_present=pool_present,
        entry_id=entry_id,
        entry_label=str(getattr(entry, "label", "") or ""),
        api_key=api_key,
        base_url=base_url,
        cache_revision=revision,
    )


def bind_selected_pool_context(client: Any, selected: SelectedPoolContext | None) -> None:
    """Attach immutable failure identity to every concrete wrapper layer."""
    if client is None or selected is None or not selected.entry_id:
        return
    pending = [client]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            current._hermes_pool_provider = selected.provider
            current._hermes_pool_entry_id = selected.entry_id
            current._hermes_pool_entry_label = selected.entry_label
            current._hermes_pool_cache_revision = selected.cache_revision
            # Existing runtime integrations consume these aliases.
            current._credential_pool_provider = selected.provider
            current._credential_pool_entry_id = selected.entry_id
            current._credential_pool_api_key = selected.api_key
            current._credential_pool_revision = selected.cache_revision
        except (AttributeError, TypeError):
            pass
        state = getattr(current, "__dict__", None)
        if not isinstance(state, dict):
            continue
        for attr in ("_real_client", "_client", "_sync"):
            child = state.get(attr)
            if child is not None and child is not current:
                pending.append(child)


def client_pool_failure_identity(client: Any) -> dict[str, str]:
    """Read the frozen identity carried by the client that made a request."""
    def string_attr(name: str) -> str:
        value = getattr(client, name, "")
        return value.strip() if isinstance(value, str) else ""

    return {
        "failed_api_key": string_attr("_credential_pool_api_key") or string_attr("api_key"),
        "failed_entry_id": string_attr("_hermes_pool_entry_id") or string_attr("_credential_pool_entry_id"),
    }


def cached_pool_client_is_current(client: Any) -> bool:
    """Whether a cached client's selected id still has the same token revision."""
    from agent import auxiliary_client as aux

    identity = client_pool_failure_identity(client)
    provider = getattr(client, "_hermes_pool_provider", "")
    if not isinstance(provider, str) or not provider.strip():
        provider = getattr(client, "_credential_pool_provider", "")
    provider = provider.strip() if isinstance(provider, str) else ""
    entry_id = identity["failed_entry_id"]
    if not provider or not entry_id:
        return True
    try:
        pool = aux.load_pool(provider)
        matches = getattr(pool, "runtime_api_key_matches", None)
        if not pool or not pool.has_credentials() or not callable(matches):
            return False
        return matches(entry_id, identity["failed_api_key"]) is True
    except Exception as exc:
        # A transient store read must not churn every cache hit. Request
        # recovery still evicts and revalidates after an actual failure.
        aux.logger.debug(
            "Auxiliary client: could not validate cached pool entry %s/%s: %s",
            provider, entry_id, exc,
        )
        return True
