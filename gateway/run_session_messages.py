"""Session-addressed plugin delivery, independent of the gateway's agent cache.

Routes belong to conversations. Cache eviction leaves them live; conversation boundaries
retire them. All adapter admission runs on the gateway loop under the owning profile.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("gateway.run")


@dataclass
class _Route:
    home: Path
    session_key: str
    session_id: str
    loop: asyncio.AbstractEventLoop
    owner: str = ""


def ensure_session_message_route(runner, source, session_key: str, session_id: str) -> None:
    """Called inside the turn's profile scope, before any model tools can send."""
    from hermes_cli.session_messages import get_session_message_route, register_session_message_route
    from hermes_constants import get_hermes_home

    if not session_key:
        return
    entry = getattr(getattr(runner, "session_store", None), "_entries", {}).get(session_key)
    if entry is None or entry.session_id != session_id:
        return
    routes = getattr(runner, "_session_message_routes", None)
    if routes is None:
        routes = runner._session_message_routes = {}
    home = Path(get_hermes_home())
    previous = routes.get(session_key)
    current = get_session_message_route(home, session_id)
    if previous is not None:
        if previous.home == home and current and current["route_id"] == previous.owner:
            previous.session_id = session_id
            return
        close_session_message_route(runner, session_key)
    route = _Route(home, session_key, session_id, asyncio.get_running_loop())

    def deliver(**kwargs):
        return _schedule_message(runner, route, **kwargs)

    route.owner = register_session_message_route(home, session_id, deliver)
    routes[session_key] = route


def alias_session_message_route(runner, session_key: str, session_id: str) -> None:
    """Publish an adopted compression child without changing the private reply address."""
    from hermes_cli.session_messages import alias_session_message_route as alias

    route = getattr(runner, "_session_message_routes", {}).get(session_key)
    if route and alias(route.home, route.owner, session_id):
        route.session_id = session_id


def close_session_message_route(runner, session_key: str) -> None:
    from hermes_cli.session_messages import close_session_message_route as close

    route = getattr(runner, "_session_message_routes", {}).pop(session_key, None)
    if route:
        close(route.home, route.owner)


def close_session_message_routes(runner) -> None:
    for key in tuple(getattr(runner, "_session_message_routes", {})):
        close_session_message_route(runner, key)


def session_message_event_is_current(runner, event, entry) -> bool:
    """Validate deferred delivery against route ownership, allowing only compression aliases."""
    from hermes_cli.session_messages import get_session_message_route

    metadata = getattr(event, "metadata", None) or {}
    owner = metadata.get("gateway_session_message_route")
    if not owner:
        return True
    route = getattr(runner, "_session_message_routes", {}).get(metadata.get("gateway_session_key"))
    if route is None or route.owner != owner or entry is None:
        return False
    current = get_session_message_route(route.home, entry.session_id)
    if not current or current["route_id"] != owner or current["session_id"] != entry.session_id:
        return False
    metadata["gateway_session_id"] = entry.session_id
    return True


def queue_leftover_steer(runner, adapter, source, session_key: str, text: str) -> bool:
    """Keep an accepted late steer even when a human follow-up already occupies the slot."""
    from gateway.platforms.event import MessageEvent, MessageType
    from hermes_cli.session_messages import get_session_message_route

    route = getattr(runner, "_session_message_routes", {}).get(session_key)
    if route is None:
        return False
    current = get_session_message_route(route.home, route.session_id)
    if current is None or current["route_id"] != route.owner:
        return False
    event = MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, internal=True,
        allow_gateway_control=False,
        metadata={
            "gateway_session_key": session_key, "gateway_session_id": current["session_id"],
            "gateway_session_strict": True, "gateway_session_message_route": route.owner,
        },
    )
    # Already accepted by the steering buffer, so it must survive even if subsequent
    # user messages filled the admission cap in the meantime.
    runner._enqueue_fifo(session_key, event, adapter)
    return getattr(event, "_gateway_accepted", False) is True


def _schedule_message(runner, route: _Route, **kwargs) -> dict:
    loop = route.loop
    if not loop.is_running() or loop.is_closed():
        return {"accepted": False, "status": "offline"}
    try:
        caller_loop = asyncio.get_running_loop()
    except RuntimeError:
        caller_loop = None
    # Plugin model tools and bridge readers run in workers. Never block the gateway's
    # own loop waiting for its admission coroutine when an embedded plugin calls directly.
    if caller_loop is loop:
        return {"accepted": False, "status": "offline", "reason": "call from a worker thread"}
    future = asyncio.run_coroutine_threadsafe(_admit_message(runner, route, **kwargs), loop)
    try:
        return future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return {"accepted": False, "status": "offline"}
    except Exception:
        logger.warning("Session message admission failed for %s", route.session_key, exc_info=True)
        return {"accepted": False, "status": "offline"}


async def _admit_message(
    runner, route: _Route, *, session_id: str, content: str, message_id: str,
    busy_mode: str = "steer",
) -> dict:
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.session_identity import replace_source
    from gateway.wake import WakeNotAccepted, adapter_supports_push, admit_internal_event
    from hermes_cli.session_messages import get_session_message_route

    if not getattr(runner, "_running", False) or getattr(runner, "_draining", False):
        return {"accepted": False, "status": "offline"}
    if getattr(runner, "_session_message_routes", {}).get(route.session_key) is not route:
        return {"accepted": False, "status": "stale_session"}
    # Routes are only registered for already-loaded conversations. Looking directly in
    # the routing index avoids yielding between ownership validation and busy admission.
    entry = runner.session_store._entries.get(route.session_key)
    current = get_session_message_route(route.home, session_id)
    bound = get_session_message_route(route.home, entry.session_id) if entry else None
    if (entry is None or entry.origin is None or not current or not bound
            or current["route_id"] != route.owner or bound["route_id"] != route.owner):
        return {"accepted": False, "status": "stale_session"}
    source = replace_source(runner._restored_source(entry))
    with runner._profile_scope_for_source(source):
        if not runner._is_user_authorized_for_source(source, allow_adapter_delegation=False):
            return {"accepted": False, "status": "denied"}
        adapter = runner._delivery_adapter_for(source)
        if adapter is None or not adapter_supports_push(adapter):
            return {"accepted": False, "status": "offline"}
        event = MessageEvent(
            text=content, message_type=MessageType.TEXT, source=source, internal=True,
            message_id=message_id, allow_gateway_control=False,
            metadata={
                "gateway_session_key": route.session_key,
                "gateway_session_id": current["session_id"], "gateway_session_strict": True,
                "gateway_session_message_route": route.owner,
            },
        )
        state = runner._peek_session_state(route.session_key)
        agent = state.turn.agent if state else None
        if agent is not None or route.session_key in getattr(adapter, "_active_sessions", {}):
            if runner._queue_depth(route.session_key, adapter=adapter) >= runner._BUSY_QUEUE_MAX_PENDING:
                return {"accepted": False, "status": "offline", "reason": "queue_full"}
            # The core's admission method closes atomically with the final steer drain.
            # Older/custom agents without it use the conversation FIFO instead.
            steer = getattr(agent, "steer_session_message", None)
            if busy_mode == "steer" and agent is not _AGENT_PENDING_SENTINEL and callable(steer):
                try:
                    if steer(content) is True:
                        return {"accepted": True, "status": "steered"}
                except Exception:
                    logger.warning("Session message steering failed; queueing", exc_info=True)
            runner._enqueue_fifo(route.session_key, event, adapter)
            return {"accepted": getattr(event, "_gateway_accepted", False) is True, "status": "queued"}
        try:
            await admit_internal_event(adapter, event)
        except WakeNotAccepted:
            return {"accepted": False, "status": "offline"}
        return {"accepted": True, "status": "started"}
