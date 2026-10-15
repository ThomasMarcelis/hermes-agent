"""Endpoint-qualified provider declarations, independent of credential ownership."""
from __future__ import annotations

import inspect
from typing import Any


class ProviderDeclarationError(RuntimeError):
    """A declared route policy failed; do not silently change its wire or context bound."""


def endpoint_profile(base_url: str):
    """Find an explicitly claimed endpoint; never infer policy from a loopback hostname."""
    if not base_url:
        return None
    from providers import list_providers
    for profile in list_providers():
        try:
            if profile.matches_endpoint(base_url) is True:
                return profile
        except Exception as exc:
            raise ProviderDeclarationError(f"Provider {profile.name!r} endpoint declaration failed") from exc
    return None


def call_model_declaration(profile: Any, method: str, *args: Any, **context: Any):
    """Pass additive context only when accepted; contain faults without policy fallback."""
    try:
        callback = getattr(profile, method)
        params = inspect.signature(callback).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            context = {k: v for k, v in context.items() if k in params}
        return callback(*args, **context)
    except Exception as exc:
        raise ProviderDeclarationError(f"Provider {profile.name!r} declaration {method!r} failed") from exc
