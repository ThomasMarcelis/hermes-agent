"""Classic CLI host for exact-conversation native plugin delivery."""
from dataclasses import dataclass
from queue import Full
import threading

from hermes_constants import get_hermes_home
from hermes_cli.session_messages import (
    close_session_message_route, get_session_message_route, register_session_message_route,
)


@dataclass(frozen=True)
class SessionMessageInput:
    owner: str
    text: str


def close_cli_session_route(cli):
    owner = getattr(cli, "_plugin_message_owner", None)
    if owner:
        cli._plugin_message_owner = None
        close_session_message_route(cli._plugin_message_home, owner)


def register_cli_session_route(cli):
    if not hasattr(cli, "_pending_input") or not getattr(cli, "session_id", None):
        return
    home = get_hermes_home()
    current = get_session_message_route(home, cli.session_id)
    if current and current["route_id"] == getattr(cli, "_plugin_message_owner", None):
        return
    close_cli_session_route(cli)
    owner = None
    lock = getattr(cli, "_plugin_message_lock", None)
    if lock is None:
        lock = cli._plugin_message_lock = threading.RLock()

    def receive(*, session_id, content, message_id, busy_mode):
        with lock:
            route = get_session_message_route(home, getattr(cli, "session_id", None))
            if (getattr(cli, "_should_exit", False) or cli._plugin_message_owner != owner
                    or not route or route["route_id"] != owner):
                return {"accepted": False, "status": "stale_session"}
            agent = getattr(cli, "agent", None)
            if (getattr(cli, "_agent_running", False) or getattr(cli, "_single_query_mode", False)) and busy_mode == "steer":
                steer = getattr(agent, "steer_session_message", None)
                if callable(steer) and steer(content):
                    return {"accepted": True, "status": "steered"}
            # Keep structured ownership until consumption: a queued /new must never
            # redirect text accepted for its predecessor into the fresh conversation.
            if cli._pending_input.qsize() >= 128:
                return {"accepted": False, "status": "offline"}
            try:
                cli._pending_input.put_nowait(SessionMessageInput(owner, content))
            except Full:
                return {"accepted": False, "status": "offline"}
            return {"accepted": True, "status": "queued"}

    cli._plugin_message_home = home
    owner = register_session_message_route(home, cli.session_id, receive)
    cli._plugin_message_owner = owner


def unwrap_cli_session_message(cli, value):
    if not isinstance(value, SessionMessageInput):
        return value, False
    if value.owner != getattr(cli, "_plugin_message_owner", None):
        return None, True
    # Literal plugin messages bypass slash, shell, file-drop and resume-number parsing.
    return value.text, True


def drain_single_query_session_messages(cli, result, run_turn, *, handoff_steer=True, close=False):
    """Consume every accepted one-shot reply before exit, closing admission atomically.

    One-shot sessions have no background input worker; never acknowledge a queue which
    the process will abandon. No linger is added: once caught up the route retires.
    """
    from queue import Empty
    pending_steer = result.pop("pending_steer", None) if handoff_steer and isinstance(result, dict) else None
    while True:
        with cli._plugin_message_lock:
            owner = getattr(cli, "_plugin_message_owner", None)
            queued = []
            held = []
            while True:
                try:
                    item = cli._pending_input.get_nowait()
                except Empty:
                    break
                if isinstance(item, SessionMessageInput):
                    if item.owner == owner:
                        queued.append(item.text)
                else:
                    held.append(item)
            for item in held:
                cli._pending_input.put_nowait(item)
            if pending_steer:
                queued.insert(0, pending_steer)
                pending_steer = None
            if not queued:
                if close:
                    close_cli_session_route(cli)
                return result
        for text in queued:
            follow = run_turn(text)
            if isinstance(follow, dict):
                result = follow
                if follow.get("messages"):
                    cli.conversation_history = follow["messages"]
                if handoff_steer and follow.get("pending_steer"):
                    extra = follow.pop("pending_steer")
                    pending_steer = pending_steer + "\n" + extra if pending_steer else extra
