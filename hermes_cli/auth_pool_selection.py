"""Cross-process durable least-used credential selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agent.credential_persistence import sanitize_borrowed_credential_payload


def _target_store(provider_id: str) -> Optional[Path]:
    """The owning store for a provider, including borrowed profile grants."""
    import hermes_cli.auth as auth_mod
    from agent.credential_pool import _borrowed_single_use_pool_root, _profile_owns_pool_provider

    if (
        provider_id in auth_mod.SINGLE_USE_REFRESH_POOL_PROVIDERS
        and not _profile_owns_pool_provider(provider_id)
    ):
        return _borrowed_single_use_pool_root()
    return None


def select_and_increment_credential_pool_entry(
    provider_id: str,
    candidate_entries: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Atomically choose and count the durable least-used available row.

    Candidate order is the deterministic tie break. Existing rows come from
    disk, so a stale process cannot overwrite refreshed tokens or metadata.
    """
    candidates: list[Dict[str, Any]] = []
    candidate_order: dict[str, int] = {}
    for candidate in candidate_entries:
        if not isinstance(candidate, dict):
            raise TypeError("candidate_entries must contain credential dictionaries")
        entry_id = str(candidate.get("id") or "").strip()
        if not entry_id:
            raise ValueError("candidate entry id is required")
        if entry_id in candidate_order:
            continue
        candidate_order[entry_id] = len(candidates)
        candidates.append(dict(candidate))
    if not candidates:
        raise ValueError("at least one candidate entry is required")

    import hermes_cli.auth as auth_mod
    from agent.credential_pool import PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED, _exhausted_until

    target_path = _target_store(provider_id)
    with auth_mod._auth_store_lock(target_path=target_path):
        auth_store = auth_mod._load_auth_store(target_path)
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool
        existing = pool.get(provider_id)
        has_provider_rows = isinstance(existing, list)
        existing_list = existing if has_provider_rows else []
        existing_by_id = {
            str(row.get("id")): row
            for row in existing_list
            if isinstance(row, dict) and row.get("id")
        }

        def count(row: Dict[str, Any]) -> int:
            try:
                return max(0, int(row.get("request_count", 0)))
            except (TypeError, ValueError):
                return 0

        nondead_count = sum(
            1
            for row in existing_list
            if isinstance(row, dict)
            and PooledCredential.from_dict(provider_id, row).last_status != STATUS_DEAD
        )
        sole_credential = nondead_count <= 1
        effective: list[Dict[str, Any]] = []
        for candidate in candidates:
            entry_id = str(candidate["id"])
            disk_row = existing_by_id.get(entry_id)
            if disk_row is None and (has_provider_rows or target_path is not None):
                # Existing rows make absence an authoritative removal. A
                # borrowed root grant is update-only even if its whole
                # provider list was removed after this process loaded it.
                continue
            row = disk_row if disk_row is not None else candidate
            durable = PooledCredential.from_dict(provider_id, row)
            if durable.last_status == STATUS_DEAD:
                continue
            if durable.last_status == STATUS_EXHAUSTED:
                until = _exhausted_until(durable, sole_credential=sole_credential)
                if until is None or until > time.time():
                    continue
            effective.append(row)
        if not effective:
            return None

        selected = min(
            effective,
            key=lambda row: (count(row), candidate_order[str(row.get("id"))]),
        )
        selected_id = str(selected.get("id"))
        updated = (
            dict(selected)
            if selected_id in existing_by_id
            else sanitize_borrowed_credential_payload(dict(selected), provider_id)
        )
        updated["request_count"] = count(selected) + 1

        merged: list[Dict[str, Any]] = []
        replaced = False
        for disk_row in existing_list:
            if (
                not replaced
                and isinstance(disk_row, dict)
                and str(disk_row.get("id") or "") == selected_id
            ):
                merged.append(updated)
                replaced = True
            else:
                merged.append(disk_row)
        if not replaced:
            merged.append(updated)
        pool[provider_id] = merged
        auth_mod._save_auth_store(auth_store, target_path=target_path)
        return dict(updated)
