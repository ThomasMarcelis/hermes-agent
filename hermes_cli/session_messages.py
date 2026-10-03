"""Live, profile-owned conversation routes for native plugin messages.

Surfaces own admission and lifetime; the registry only addresses, scopes and deduplicates.
A route survives compression and agent replacement, but not conversation replacement.
"""
from __future__ import annotations

import contextvars
import logging
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ROUTES: dict[tuple[str, str], "_Route"] = {}
_OWNERS: dict[str, "_Route"] = {}
_RETIRED: OrderedDict[tuple[str, str], None] = OrderedDict()
_RECEIPT_LIMIT = 2048
_RETIRED_LIMIT = 8192


@dataclass
class _Route:
    home: str
    session_id: str
    owner: str
    callback: Callable
    context: contextvars.Context
    aliases: set[str] = field(default_factory=set)
    receipts: OrderedDict = field(default_factory=OrderedDict)
    pending: set[str] = field(default_factory=set)
    active: bool = True


@contextmanager
def profile_message_scope(profile_home):
    """Bind all runtime scopes for a callback invoked outside its owning turn."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    home = Path(profile_home)
    home_token = set_hermes_home_override(str(home))
    secret_token = terminal_token = None
    try:
        secret_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
        terminal_token = install_profile_terminal_scope(home)
        yield
    finally:
        if terminal_token is not None:
            reset_terminal_scope(terminal_token)
        if secret_token is not None:
            reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def register_session_message_route(profile_home, session_id, callback, *, owner=None):
    """Return an opaque ownership token. Re-registration with that token is idempotent."""
    if not session_id or not callable(callback):
        raise ValueError("A durable session identity and delivery callback are required")
    home = hermes_home_key(profile_home)
    session_id = str(session_id)
    previous = None
    with _LOCK:
        route = _OWNERS.get(owner) if owner else None
        if route is not None and route.home == home and route.active:
            route.callback = callback
            route.context = contextvars.copy_context()
            alias_session_message_route(home, owner, session_id)
            return owner
        previous = _ROUTES.get((home, session_id))
        if previous:
            _retire_locked(previous)
        token = uuid.uuid4().hex
        route = _Route(home, session_id, token, callback, contextvars.copy_context(), {session_id})
        _OWNERS[token] = route
        _ROUTES[(home, session_id)] = route
        _RETIRED.pop((home, session_id), None)
    if previous:
        _notify_closed(previous)
    return token


def alias_session_message_route(profile_home, owner, session_id):
    """Publish a compression continuation; never call this for /new or branching."""
    home = hermes_home_key(profile_home)
    session_id = str(session_id)
    with _LOCK:
        route = _OWNERS.get(owner)
        if route is None or route.home != home or not route.active or not session_id:
            return False
        existing = _ROUTES.get((home, session_id))
        if existing is not None and existing is not route:
            return False
        route.aliases.add(session_id)
        route.session_id = session_id
        _ROUTES[(home, session_id)] = route
        _RETIRED.pop((home, session_id), None)
        return True


def alias_session_route(profile_home, previous_session_id, session_id):
    """Publish a compression alias before subsequent tools execute in the same turn."""
    with _LOCK:
        route = _ROUTES.get((hermes_home_key(profile_home), str(previous_session_id)))
        return bool(route and alias_session_message_route(profile_home, route.owner, session_id))


def _retire_locked(route):
    route.active = False
    _OWNERS.pop(route.owner, None)
    for alias in route.aliases:
        key = (route.home, alias)
        if _ROUTES.get(key) is route:
            _ROUTES.pop(key)
            _RETIRED[key] = None
    while len(_RETIRED) > _RETIRED_LIMIT:
        _RETIRED.popitem(last=False)


def _notify_closed(route):
    def notify():
        from hermes_cli.plugins import get_plugin_manager
        with profile_message_scope(route.home):
            get_plugin_manager().invoke_hook("on_session_message_route_closed", route_id=route.owner)
    try:
        route.context.copy().run(notify)
    except Exception:
        logger.warning("Session message route close hook failed", exc_info=True)


def close_session_message_route(profile_home, owner):
    """Retire only this owner; delayed cleanup cannot close its replacement."""
    with _LOCK:
        route = _OWNERS.get(owner)
        if route is None or route.home != hermes_home_key(profile_home):
            return False
        _retire_locked(route)
    _notify_closed(route)
    return True


def get_session_message_route(profile_home, session_id):
    with _LOCK:
        route = _ROUTES.get((hermes_home_key(profile_home), str(session_id)))
        if route is None or not route.active:
            return None
        return {"route_id": route.owner, "session_id": route.session_id}


def inject_session_message(profile_home, session_id, content, *, message_id, busy_mode="steer", expected_route_id=None):
    """Return surface admission, never claim model consumption or successful completion."""
    if not isinstance(content, str) or not content.strip() or not isinstance(message_id, str) or not message_id:
        return {"accepted": False, "status": "denied"}
    if busy_mode not in {"steer", "queue"}:
        return {"accepted": False, "status": "denied"}
    key = (hermes_home_key(profile_home), str(session_id))
    with _LOCK:
        route = _ROUTES.get(key)
        if route is None or not route.active:
            return {"accepted": False, "status": "stale_session" if key in _RETIRED else "offline"}
        if expected_route_id is not None and route.owner != expected_route_id:
            return {"accepted": False, "status": "stale_session"}
        if message_id in route.receipts:
            return {"accepted": True, "status": "duplicate"}
        if message_id in route.pending:
            return {"accepted": False, "status": "offline", "reason": "admission_in_progress"}
        route.pending.add(message_id)
        callback = route.callback
        context = route.context.copy()
    def deliver():
        with profile_message_scope(route.home):
            return callback(session_id=route.session_id, content=content, message_id=message_id, busy_mode=busy_mode)
    try:
        receipt = context.run(deliver)
        if (not isinstance(receipt, dict) or receipt.get("accepted") is not True
                or receipt.get("status") not in {"steered", "queued", "started", "duplicate"}):
            status = receipt.get("status") if isinstance(receipt, dict) else "offline"
            receipt = {"accepted": False, "status": status if status in {"offline", "stale_session", "denied"} else "offline"}
        else:
            receipt = {"accepted": True, "status": receipt["status"]}
    except Exception:
        logger.warning("Session message admission failed", exc_info=True)
        receipt = {"accepted": False, "status": "offline"}
    with _LOCK:
        if receipt["accepted"]:
            route.receipts[message_id] = receipt
            while len(route.receipts) > _RECEIPT_LIMIT:
                route.receipts.popitem(last=False)
        route.pending.discard(message_id)
    return receipt
