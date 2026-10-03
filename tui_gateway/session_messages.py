"""Plugin messages addressed to one live TUI/Desktop conversation.

Routes belong to the session record, not the focused transport or current agent.
The generic plugin registry owns profile binding, duplicate receipts and aliases.
"""

from __future__ import annotations

from .method_ctx import bind_module


_PLUGIN_MESSAGE_QUEUE_LIMIT = 128
_PLUGIN_MESSAGE_QUEUE_BYTES = 1024 * 1024


def _register_plugin_session_route(sid: str, session: dict) -> None:
    from hermes_cli.session_messages import register_session_message_route

    if session.get("_finalized") or session.get("_plugin_message_route"):
        return
    key = str(session.get("session_key") or "")
    if not key:
        return
    home = _session_home(session)
    owner = None

    def receive(*, session_id, content, message_id, busy_mode):
        return _receive_plugin_session_message(
            sid, session, owner, content=content, message_id=message_id, busy_mode=busy_mode)

    owner = register_session_message_route(home, key, receive)
    session["_plugin_message_route"] = owner
    session["_plugin_message_home"] = home


def _alias_plugin_session_route(session: dict, key: str) -> None:
    from hermes_cli.session_messages import alias_session_message_route

    if owner := session.get("_plugin_message_route"):
        alias_session_message_route(session["_plugin_message_home"], owner, key)


def _close_plugin_session_route(session: dict) -> None:
    from hermes_cli.session_messages import close_session_message_route

    if owner := session.pop("_plugin_message_route", None):
        close_session_message_route(session.pop("_plugin_message_home"), owner)


def _plugin_session_route_live(sid: str, session: dict, owner) -> bool:
    return bool(owner and session.get("_plugin_message_route") == owner
                and _sessions.get(sid) is session
                and not any(session.get(flag) for flag in ("_closing", "_finalized", "_lease_taken_over")))


def _queue_plugin_session_message(session: dict, content: str) -> bool:
    """Bound plugin arrivals without merging them into or replacing queued user input.

    Caller holds history_lock. The normal post-turn drain owns dispatch and transport
    reattachment; separate envelopes preserve arrival order and user attachments.
    """
    head = session.get("queued_prompt")
    entries = ([head] if head else []) + list(session.get("queued_prompts") or [])
    if (len(entries) >= _PLUGIN_MESSAGE_QUEUE_LIMIT
            or sum(len(str(entry.get("text", "")).encode("utf-8")) for entry in entries)
            + len(content.encode("utf-8")) > _PLUGIN_MESSAGE_QUEUE_BYTES):
        return False
    _ac_set_queue(session, [*entries, {
        "text": content, "transport": session.get("transport"), "image_paths": []}])
    session["last_active"] = time.time()
    return True


def _receive_plugin_session_message(
    sid: str, session: dict, owner, *, content: str, message_id: str, busy_mode: str
) -> dict:
    """Accept a safe steer or dispatch through the conversation's ordinary turn path."""
    rid = f"plugin-message-{message_id}"
    with _session_turn_admission(session) as admitted:
        if not admitted:
            return {"accepted": False, "status": "offline"}
        if not _plugin_session_route_live(sid, session, owner):
            return {"accepted": False, "status": "stale_session"}
        if session.get("running"):
            agent = session.get("agent")
            if busy_mode == "steer" and not session.get("_compute_host_active"):
                try:
                    if agent is not None and agent.steer_session_message(content):
                        _record_inflight_correction(session, content)
                        session["last_active"] = time.time()
                        return {"accepted": True, "status": "steered"}
                except Exception:
                    logger.debug("Plugin message steer refused for %s; queueing", sid, exc_info=True)
            accepted = _queue_plugin_session_message(session, content)
            return {"accepted": accepted, "status": "queued" if accepted else "offline"}
        session["running"] = True
    # _run_prompt_submit publishes message.start and starts the context-preserving
    # worker. It also checks lease ownership and closing again before dispatch.
    try:
        if _session_uses_compute_host(session):
            if (refusal := _ensure_active_session_slot(sid, session)) is not None:
                _notif_release_turn(session)
                return {"accepted": False, "status": "offline", "reason": str(refusal)}
            response = _submit_prompt_to_compute_host(rid, sid, session, content, image_paths=[])
            accepted = not response.get("error")
        else:
            accepted = _run_prompt_submit(rid, sid, session, content, image_paths=[])
    except Exception:
        _notif_release_turn(session)
        logger.exception("Plugin message dispatch failed for %s", sid)
        return {"accepted": False, "status": "offline"}
    if not accepted:
        _notif_release_turn(session)
    return {"accepted": bool(accepted), "status": "started" if accepted else "offline"}


def register(server) -> None:
    bind_module(globals(), server)
