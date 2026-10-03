"""Plugin replies retain their profile and originating conversation across UI changes."""

import threading
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_cli import session_messages as routes
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class Peer:
    def __init__(self):
        self.frames = []
        self.received = threading.Event()
        self._closed = False

    def write(self, frame):
        self.frames.append(frame)
        self.received.set()
        return True


@pytest.fixture
def conversations(tmp_path, monkeypatch):
    sessions = {}
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session: False)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *args: threading.Event())
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *args: None)
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text("model: test\n")
        sessions[name] = {
            "profile_home": str(home), "session_key": "same-durable-id", "source": "desktop",
            "agent": SimpleNamespace(session_id="same-durable-id"),
            "history_lock": threading.Lock(), "history": [], "running": False,
            "transport": Peer(), "last_active": 0,
        }
        server._start_session_services(name, "same-durable-id", sessions[name])
    yield sessions
    for session in sessions.values():
        server._close_plugin_session_route(session)


@pytest.mark.parametrize("mode", ["idle", "steer", "steer_refused", "queue"])
def test_delivery_is_profile_and_conversation_owned(conversations, monkeypatch, mode):
    received = []
    focused = Peer()
    original_input = {"text": "my next question", "image_paths": ["attachment"], "transport": focused}

    def submit(rid, sid, session, text, **kwargs):
        received.append((str(get_hermes_home()), sid, text, kwargs))
        server._emit("message.complete", sid, {"text": text})
        session["running"] = False
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    for sid, session in conversations.items():
        session["running"] = mode != "idle"
        session["queued_prompt"] = dict(original_input)

        def steer(text, *, sid=sid):
            if mode == "steer_refused":
                return False
            received.append((str(get_hermes_home()), sid, text))
            return True

        session["agent"].steer_session_message = steer
    token = bind_transport(focused)
    try:
        for index, sid in enumerate(("a", "b", "a")):
            session = conversations[sid]
            receipt = routes.inject_session_message(
                session["profile_home"], "same-durable-id", f"reply {index}",
                message_id=str(index), busy_mode="queue" if mode == "queue" else "steer")
            expected = "started" if mode == "idle" else "steered" if mode == "steer" else "queued"
            assert receipt == {"accepted": True, "status": expected}
            assert routes.inject_session_message(
                session["profile_home"], "same-durable-id", f"reply {index}",
                message_id=str(index)) == {"accepted": True, "status": "duplicate"}
    finally:
        reset_transport(token)
    assert not focused.frames
    for sid, session in conversations.items():
        assert session["queued_prompt"] == original_input
        wanted = ["reply 0", "reply 2"] if sid == "a" else ["reply 1"]
        if mode in {"queue", "steer_refused"}:
            assert [entry["text"] for entry in session["queued_prompts"]] == wanted
            assert all(entry["transport"] is session["transport"] for entry in session["queued_prompts"])
        else:
            assert [entry[2] for entry in received if entry[1] == sid] == wanted
            assert all(entry[0] == session["profile_home"] for entry in received if entry[1] == sid)
        if mode == "idle":
            assert session["transport"].received.wait(5)
            assert all(entry[3]["image_paths"] == [] for entry in received)


def test_compression_rebuild_and_retirement_do_not_redirect_replies(conversations, monkeypatch):
    session = conversations["a"]
    home = session["profile_home"]
    owner = session["_plugin_message_route"]
    received = []
    session["running"] = True
    session["agent"] = SimpleNamespace(
        session_id="compressed", steer_session_message=lambda text: received.append(text) or True)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *args, **kwargs: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args: None)
    server._sync_session_key_after_compress("a", session)
    server._register_plugin_session_route("a", session)  # rebuilding services retains the inbox
    assert session["_plugin_message_route"] == owner
    for index, alias in enumerate(("same-durable-id", "compressed")):
        assert routes.inject_session_message(home, alias, alias, message_id=str(index))["accepted"]
    assert received == ["same-durable-id", "compressed"]

    # A new conversation can reuse the UI slot; the old closure must reject it.
    replacement = dict(session, _plugin_message_route=None)
    conversations["a"] = replacement
    server._register_plugin_session_route("a", replacement)
    assert not server._receive_plugin_session_message(
        "a", session, owner, content="stale", message_id="late", busy_mode="steer")["accepted"]
    server._close_plugin_session_route(session)  # stale cleanup must not retire the new owner
    assert routes.inject_session_message(home, "compressed", "new owner", message_id="new")["accepted"]
    server._finalize_session(replacement, end_reason="ws_orphan_reap")
    assert not routes.inject_session_message(home, "compressed", "closed", message_id="after")["accepted"]


def test_rejected_dispatch_can_retry_and_busy_queue_is_bounded(conversations, monkeypatch):
    session = conversations["a"]
    home = session["profile_home"]
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: False)
    assert not routes.inject_session_message(home, "same-durable-id", "retry", message_id="id")["accepted"]
    assert not session["running"]
    session["running"] = True
    session["queued_prompt"] = {"text": "a user message", "transport": session["transport"]}
    monkeypatch.setattr(server, "_PLUGIN_MESSAGE_QUEUE_LIMIT", 1)
    assert not routes.inject_session_message(
        home, "same-durable-id", "retry", message_id="id", busy_mode="queue")["accepted"]
    assert session["queued_prompt"]["text"] == "a user message"
    monkeypatch.setattr(server, "_PLUGIN_MESSAGE_QUEUE_LIMIT", 2)
    assert routes.inject_session_message(
        home, "same-durable-id", "retry", message_id="id", busy_mode="queue")["accepted"]
    queued = session["queued_prompts"][0]
    session["attached_images"] = ["an unsent composer image"]
    server._ac_set_queue(session, [queued])  # the user's preceding turn has drained
    session["running"] = False
    submitted = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: submitted.append(kwargs) or True)
    assert server._drain_queued_prompt("drain", "a", session)
    assert submitted[0]["image_paths"] == []
    assert session["attached_images"] == ["an unsent composer image"]
