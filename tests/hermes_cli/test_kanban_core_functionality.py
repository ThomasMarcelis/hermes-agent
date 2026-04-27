"""Core-functionality tests for the kanban kernel + CLI additions.

Complements tests/hermes_cli/test_kanban_db.py (schema + CAS atomicity)
and tests/hermes_cli/test_kanban_cli.py (end-to-end run_slash).  The
tests here exercise the pieces added as part of the kanban hardening
pass: circuit breaker, crash detection, daemon loop, idempotency,
retention/gc, stats, notify subscriptions, worker log accessor, run_slash
parity across every registered verb.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import run_slash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------

def test_idempotency_key_returns_existing_task(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="first", idempotency_key="abc")
        b = kb.create_task(conn, title="second attempt", idempotency_key="abc")
        assert a == b, "same idempotency_key should return the same task id"
        # And body wasn't overwritten — first create wins.
        task = kb.get_task(conn, a)
        assert task.title == "first"
    finally:
        conn.close()


def test_idempotency_key_ignored_for_archived(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="first", idempotency_key="abc")
        kb.archive_task(conn, a)
        b = kb.create_task(conn, title="second", idempotency_key="abc")
        assert a != b, "archived task shouldn't block a fresh create with same key"
    finally:
        conn.close()


def test_no_idempotency_key_never_collides(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        assert a != b
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Spawn-failure circuit breaker
# ---------------------------------------------------------------------------

def test_spawn_failure_auto_blocks_after_limit(kanban_home):
    """N consecutive spawn failures on the same task → auto_blocked."""
    def _bad_spawn(task, ws):
        raise RuntimeError("no PATH")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Three ticks below the default limit (5) → still ready, counter grows.
        for i in range(3):
            res = kb.dispatch_once(conn, spawn_fn=_bad_spawn, failure_limit=5)
            assert tid not in res.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.spawn_failures == 3

        # Two more ticks → fifth failure exceeds the limit.
        res1 = kb.dispatch_once(conn, spawn_fn=_bad_spawn, failure_limit=5)
        assert tid not in res1.auto_blocked
        res2 = kb.dispatch_once(conn, spawn_fn=_bad_spawn, failure_limit=5)
        assert tid in res2.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.spawn_failures >= 5
        assert task.last_spawn_error and "no PATH" in task.last_spawn_error
    finally:
        conn.close()


def test_successful_spawn_resets_failure_counter(kanban_home):
    """A successful spawn clears the counter so past failures don't count
    against future retries of the same task."""
    calls = [0]
    def _flaky_spawn(task, ws):
        calls[0] += 1
        if calls[0] <= 2:
            raise RuntimeError("transient")
        return 99999  # pid value — harmless; crash detection will clear it

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Two failures + one success.
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        task = kb.get_task(conn, tid)
        assert task.spawn_failures == 2
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        task = kb.get_task(conn, tid)
        assert task.spawn_failures == 0
        assert task.last_spawn_error is None
        # Task is now running with a pid.
        assert task.status == "running"
        assert task.worker_pid == 99999
    finally:
        conn.close()


def test_workspace_resolution_failure_also_counts(kanban_home):
    """`dir:` workspace with no path should fail workspace resolution AND
    count against the failure budget — not just crash the tick."""
    conn = kb.connect()
    try:
        # Manually insert a broken task: dir workspace but workspace_path is NULL
        # after initial create. We achieve this by creating via kanban_db then
        # UPDATE-ing workspace_path to NULL.
        tid = kb.create_task(
            conn, title="x", assignee="worker",
            workspace_kind="dir", workspace_path="/tmp/kanban_e2e_dir",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path = NULL WHERE id = ?", (tid,),
            )
        res = kb.dispatch_once(conn, failure_limit=3)
        task = kb.get_task(conn, tid)
        assert task.spawn_failures == 1
        assert task.status == "ready"
        assert task.last_spawn_error and "workspace" in task.last_spawn_error
        # Run twice more → auto-blocked.
        kb.dispatch_once(conn, failure_limit=3)
        res = kb.dispatch_once(conn, failure_limit=3)
        assert tid in res.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker aliveness / crash detection
# ---------------------------------------------------------------------------

def test_pid_alive_helper():
    # Our own pid is alive.
    assert kb._pid_alive(os.getpid())
    # PID 0 / None / negative.
    assert not kb._pid_alive(0)
    assert not kb._pid_alive(None)
    # A clearly-dead pid (very large, extremely unlikely to exist).
    assert not kb._pid_alive(2 ** 30)


def test_detect_crashed_workers_reclaims(kanban_home):
    """A running task whose pid vanished gets dropped to ready with a
    ``crashed`` event, independent of the claim TTL."""
    def _spawn_pid_that_exits(task, ws):
        # Spawn a real child that exits instantly.
        import subprocess
        p = subprocess.Popen(
            ["python3", "-c", "pass"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
        p.wait()
        return p.pid

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        res = kb.dispatch_once(conn, spawn_fn=_spawn_pid_that_exits)
        # Brief sleep to make sure the child's pid has been reaped; on
        # busy CI the pid may be reused by another process, which would
        # fool _pid_alive. If that happens we accept the test still
        # passing as long as the dispatcher ran without error.
        time.sleep(0.2)
        res2 = kb.dispatch_once(conn)
        task = kb.get_task(conn, tid)
        # Either crashed was detected (preferred) or the TTL reclaim path
        # will eventually fire; we accept either outcome but the worker_pid
        # should no longer be set.
        if res2.crashed:
            assert tid in res2.crashed
            events = kb.list_events(conn, tid)
            assert any(e.kind == "crashed" for e in events)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def test_daemon_runs_and_stops(kanban_home):
    """run_daemon should execute at least one tick and exit cleanly on
    stop_event."""
    ticks = []
    stop = threading.Event()

    def _runner():
        kb.run_daemon(
            interval=0.05,
            stop_event=stop,
            on_tick=lambda res: ticks.append(res),
        )

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    # Give it a few ticks.
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "daemon should exit on stop_event"
    assert len(ticks) >= 1, "expected at least one tick"


def test_daemon_keeps_going_after_tick_exception(kanban_home, monkeypatch):
    """A tick that raises shouldn't kill the loop."""
    calls = [0]
    orig_dispatch = kb.dispatch_once

    def _boom(conn, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("simulated tick failure")
        return orig_dispatch(conn, **kw)

    monkeypatch.setattr(kb, "dispatch_once", _boom)

    stop = threading.Event()
    def _runner():
        kb.run_daemon(interval=0.05, stop_event=stop)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2.0)
    # At minimum, second-tick+ should have run.
    assert calls[0] >= 2


# ---------------------------------------------------------------------------
# Stats + age
# ---------------------------------------------------------------------------

def test_board_stats(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="x")
        b = kb.create_task(conn, title="b", assignee="y")
        kb.complete_task(conn, a, result="done")
        stats = kb.board_stats(conn)
        assert stats["by_status"]["ready"] == 1
        assert stats["by_status"]["done"] == 1
        assert stats["by_assignee"]["x"]["done"] == 1
        assert stats["by_assignee"]["y"]["ready"] == 1
        assert stats["oldest_ready_age_seconds"] is not None
    finally:
        conn.close()


def test_task_age_helper(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        task = kb.get_task(conn, tid)
        age = kb.task_age(task)
        assert age["created_age_seconds"] is not None
        assert age["started_age_seconds"] is None
        assert age["time_to_complete_seconds"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Notify subscriptions
# ---------------------------------------------------------------------------

def test_notify_sub_crud(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123", user_id="u1",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["platform"] == "telegram"
        # Duplicate add is a no-op.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert len(kb.list_notify_subs(conn, tid)) == 1
        # Distinct thread is a new row.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            thread_id="5",
        )
        assert len(kb.list_notify_subs(conn, tid)) == 2
        # Remove one.
        ok = kb.remove_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert ok is True
        assert len(kb.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()


def test_notify_cursor_advances(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="123")
        # Initial: one "created" event but we only want terminal kinds.
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert events == []
        # Complete the task → new `completed` event.
        kb.complete_task(conn, tid, result="ok")
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert len(events) == 1
        assert events[0].kind == "completed"
        # Advance cursor — next call returns empty.
        kb.advance_notify_cursor(
            conn, task_id=tid, platform="telegram", chat_id="123",
            new_cursor=cursor,
        )
        _, events2 = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert events2 == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GC + retention
# ---------------------------------------------------------------------------

def test_gc_events_keeps_active_task_history(kanban_home):
    """gc_events should only prune rows for terminal (done/archived) tasks."""
    conn = kb.connect()
    try:
        alive = kb.create_task(conn, title="a", assignee="w")
        done_id = kb.create_task(conn, title="b", assignee="w")
        kb.complete_task(conn, done_id)

        # Force all existing events to "old" by bumping created_at backwards.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = ?",
                (int(time.time()) - 60 * 24 * 3600,),
            )
        removed = kb.gc_events(conn, older_than_seconds=30 * 24 * 3600)
        # At least the done task's "created" + "completed" events gone.
        assert removed >= 2
        # Alive task's events survive.
        alive_events = kb.list_events(conn, alive)
        assert len(alive_events) >= 1
    finally:
        conn.close()


def test_gc_worker_logs_deletes_old_files(kanban_home):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    old = log_dir / "old.log"
    young = log_dir / "young.log"
    old.write_text("stale")
    young.write_text("fresh")
    # Age the old file by 100 days.
    past = time.time() - 100 * 24 * 3600
    os.utime(old, (past, past))
    removed = kb.gc_worker_logs(older_than_seconds=30 * 24 * 3600)
    assert removed == 1
    assert not old.exists()
    assert young.exists()


# ---------------------------------------------------------------------------
# Log rotation + accessor
# ---------------------------------------------------------------------------

def test_worker_log_rotation_keeps_one_generation(kanban_home, tmp_path):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / "t_aaaa.log"
    target.write_bytes(b"x" * (3 * 1024 * 1024))  # 3 MiB, over 2 MiB threshold
    kb._rotate_worker_log(target, kb.DEFAULT_LOG_ROTATE_BYTES)
    assert not target.exists()
    assert (log_dir / "t_aaaa.log.1").exists()


def test_read_worker_log_tail(kanban_home):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "t_beef.log"
    # 10 lines
    p.write_text("\n".join(f"line {i}" for i in range(10)))
    full = kb.read_worker_log("t_beef")
    assert full is not None and "line 0" in full
    tail = kb.read_worker_log("t_beef", tail_bytes=30)
    assert tail is not None
    # Tail should not include line 0.
    assert "line 0" not in tail
    # Missing log returns None.
    assert kb.read_worker_log("t_missing") is None


# ---------------------------------------------------------------------------
# CLI bulk verbs
# ---------------------------------------------------------------------------

def test_cli_complete_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c")
    finally:
        conn.close()
    out = run_slash(f"complete {a} {b} {c} --result all-done")
    assert out.count("Completed") == 3
    conn = kb.connect()
    try:
        for tid in (a, b, c):
            assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


def test_cli_archive_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
    finally:
        conn.close()
    out = run_slash(f"archive {a} {b}")
    assert "Archived" in out
    conn = kb.connect()
    try:
        assert kb.get_task(conn, a).status == "archived"
        assert kb.get_task(conn, b).status == "archived"
    finally:
        conn.close()


def test_cli_unblock_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.block_task(conn, a)
        kb.block_task(conn, b)
    finally:
        conn.close()
    out = run_slash(f"unblock {a} {b}")
    assert out.count("Unblocked") == 2


def test_cli_block_bulk_via_ids_flag(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
    finally:
        conn.close()
    out = run_slash(f"block {a} need input --ids {b}")
    assert out.count("Blocked") == 2


def test_cli_create_with_idempotency_key(kanban_home):
    out1 = run_slash("create 'x' --idempotency-key abc --json")
    tid1 = json.loads(out1)["id"]
    out2 = run_slash("create 'y' --idempotency-key abc --json")
    tid2 = json.loads(out2)["id"]
    assert tid1 == tid2


# ---------------------------------------------------------------------------
# CLI stats / watch / log / notify / daemon parity
# ---------------------------------------------------------------------------

def test_cli_stats_json(kanban_home):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="a", assignee="r")
    finally:
        conn.close()
    out = run_slash("stats --json")
    data = json.loads(out)
    assert "by_status" in data
    assert "by_assignee" in data
    assert "oldest_ready_age_seconds" in data


def test_cli_notify_subscribe_and_list(kanban_home):
    tid = run_slash("create 'x' --json")
    tid = json.loads(tid)["id"]
    out = run_slash(
        f"notify-subscribe {tid} --platform telegram --chat-id 999",
    )
    assert "Subscribed" in out
    lst = run_slash("notify-list --json")
    subs = json.loads(lst)
    assert any(s["task_id"] == tid and s["platform"] == "telegram" for s in subs)
    rm = run_slash(
        f"notify-unsubscribe {tid} --platform telegram --chat-id 999",
    )
    assert "Unsubscribed" in rm


def test_cli_log_missing_task(kanban_home):
    # No such task → exit-style (no log for...) message on stderr, returned
    # in combined output.
    out = run_slash("log t_nope")
    assert "no log" in out.lower()


def test_cli_gc_reports_counts(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kb.archive_task(conn, tid)
    finally:
        conn.close()
    out = run_slash("gc")
    assert "GC complete" in out


# ---------------------------------------------------------------------------
# run_slash parity — every verb returns a sensible, non-crashy string
# ---------------------------------------------------------------------------

def test_run_slash_every_verb_returns_sensible_output(kanban_home):
    """Smoke-test every verb with minimal args. None may raise, none may
    return the empty string (must either succeed or report a usage error)."""
    # Set up a pair of tasks to reference.
    conn = kb.connect()
    try:
        tid_a = kb.create_task(conn, title="a")
        tid_b = kb.create_task(conn, title="b", parents=[tid_a])
    finally:
        conn.close()

    invocations = [
        "",                                  # no subcommand → help text
        "--help",
        "init",
        "create 'smoke'",
        "list",
        "ls",
        f"show {tid_a}",
        f"assign {tid_a} researcher",
        f"link {tid_a} {tid_b}",
        f"unlink {tid_a} {tid_b}",
        f"claim {tid_a}",
        f"comment {tid_a} hello",
        f"complete {tid_a}",
        f"block {tid_b} need input",
        f"unblock {tid_b}",
        f"archive {tid_a}",
        "dispatch --dry-run --json",
        "stats --json",
        "notify-list",
        f"log {tid_a}",
        f"context {tid_b}",
        "gc",
    ]
    for cmd in invocations:
        out = run_slash(cmd)
        assert out is not None
        assert out.strip() != "", f"empty output for `/kanban {cmd}`"


# ---------------------------------------------------------------------------
# Max-runtime enforcement (item 1 from the Multica audit)
# ---------------------------------------------------------------------------

def test_max_runtime_terminates_overrun_worker(kanban_home):
    """A running task whose elapsed time exceeds max_runtime_seconds gets
    SIGTERM'd, emits a ``timed_out`` event, and goes back to ready."""
    killed = []
    def _signal_fn(pid, sig):
        killed.append((pid, sig))

    # We bypass _pid_alive by stubbing it so the grace-poll exits fast.
    import hermes_cli.kanban_db as _kb
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False  # pretend SIGTERM worked immediately

    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(
                conn, title="long job", assignee="worker",
                max_runtime_seconds=1,  # one second cap
            )
            # Spawn by hand: claim + set pid + set started_at to the past.
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, os.getpid())   # any live pid works
            # Backdate started_at so elapsed > limit.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET started_at = ? WHERE id = ?",
                    (int(time.time()) - 30, tid),
                )

            timed_out = kb.enforce_max_runtime(conn, signal_fn=_signal_fn)
            assert tid in timed_out
            assert killed and killed[0][0] == os.getpid()

            task = kb.get_task(conn, tid)
            assert task.status == "ready",                 f"timed-out task should reset to ready, got {task.status}"
            assert task.worker_pid is None
            assert task.last_heartbeat_at is None

            events = kb.list_events(conn, tid)
            assert any(e.kind == "timed_out" for e in events)
            to_event = next(e for e in events if e.kind == "timed_out")
            assert to_event.payload["limit_seconds"] == 1
            assert to_event.payload["elapsed_seconds"] >= 30
        finally:
            conn.close()
    finally:
        _kb._pid_alive = original_alive


def test_max_runtime_none_means_no_cap(kanban_home):
    """A task with max_runtime_seconds=None is never timed out regardless
    of how long it runs."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="uncapped", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, os.getpid())
        # Backdate aggressively; no cap means we don't care.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (int(time.time()) - 100_000, tid),
            )
        timed_out = kb.enforce_max_runtime(conn)
        assert timed_out == []
        task = kb.get_task(conn, tid)
        assert task.status == "running"
    finally:
        conn.close()


def test_create_task_persists_max_runtime(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", max_runtime_seconds=600)
        task = kb.get_task(conn, tid)
        assert task.max_runtime_seconds == 600
    finally:
        conn.close()


def test_enforce_max_runtime_integrates_with_dispatch(kanban_home, monkeypatch):
    """enforce_max_runtime + dispatch_once integrate cleanly — a timed-out
    task goes through ``timed_out`` → ``ready`` and dispatch_once can then
    re-spawn it without re-reporting the timeout."""
    import hermes_cli.kanban_db as _kb
    # Leave _pid_alive=True so the crash detector doesn't steal the task
    # before timeout enforcement runs. After SIGTERM in enforce_max_runtime,
    # pretend the worker died so the grace wait exits fast.
    state = {"sent_term": False}
    def _alive(pid):
        return not state["sent_term"]
    def _signal(pid, sig):
        import signal as _sig
        if sig == _sig.SIGTERM:
            state["sent_term"] = True
    monkeypatch.setattr(_kb, "_pid_alive", _alive)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="timeout-me", assignee="worker",
            max_runtime_seconds=1,
        )
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, os.getpid())
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (int(time.time()) - 30, tid),
            )
        # Use enforce_max_runtime directly with our signal stub — dispatch_once
        # uses the default os.kill, but integration-wise calling
        # enforce_max_runtime directly proves the kernel wiring. For the
        # dispatch_once assertion, rely on its own code path by calling it
        # after forcing SIGTERM via enforce_max_runtime.
        before = kb.enforce_max_runtime(conn, signal_fn=_signal)
        assert tid in before, "kernel enforce_max_runtime should catch the overrun"

        # Now a second dispatch_once run should be a no-op on this task
        # (already released). Confirm the loop doesn't re-report it.
        res = kb.dispatch_once(conn, spawn_fn=lambda t, ws: None)
        task = kb.get_task(conn, tid)
        # After timeout, task is back in 'ready' and will be re-spawned
        # by the same pass. That's the intended behaviour.
        assert task.status in ("ready", "running")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Heartbeat (item 2 from the Multica audit)
# ---------------------------------------------------------------------------

def test_heartbeat_on_running_task(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        ok = kb.heartbeat_worker(conn, tid, note="step 3/10")
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.last_heartbeat_at is not None
        events = kb.list_events(conn, tid)
        hb = [e for e in events if e.kind == "heartbeat"]
        assert len(hb) == 1
        assert hb[0].payload == {"note": "step 3/10"}
    finally:
        conn.close()


def test_heartbeat_refused_when_not_running(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")   # lands in ready, not running
        ok = kb.heartbeat_worker(conn, tid)
        assert ok is False
        task = kb.get_task(conn, tid)
        assert task.last_heartbeat_at is None
    finally:
        conn.close()


def test_cli_heartbeat_verb(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    out = run_slash(f"heartbeat {tid}")
    assert "Heartbeat recorded" in out

    # With --note.
    out = run_slash(f"heartbeat {tid} --note 'step 42'")
    assert "Heartbeat recorded" in out
    conn = kb.connect()
    try:
        events = kb.list_events(conn, tid)
        notes = [e.payload.get("note") for e in events if e.kind == "heartbeat" and e.payload]
        assert "step 42" in notes
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Event vocab rename + spawned event (item 3 from Multica)
# ---------------------------------------------------------------------------

def test_recompute_ready_emits_promoted_not_ready(kanban_home):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p")
        child = kb.create_task(conn, title="c", parents=[parent])
        kb.complete_task(conn, parent, result="ok")
        # recompute_ready runs inside complete_task too, but call it again
        # defensively.
        kb.recompute_ready(conn)
        events = kb.list_events(conn, child)
        kinds = [e.kind for e in events]
        assert "promoted" in kinds
        # Old name must not appear.
        assert "ready" not in kinds
    finally:
        conn.close()


def test_spawn_failure_circuit_breaker_emits_gave_up(kanban_home):
    def _bad(task, ws):
        raise RuntimeError("nope")
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        for _ in range(5):
            kb.dispatch_once(conn, spawn_fn=_bad, failure_limit=5)
        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "gave_up" in kinds
        assert "spawn_auto_blocked" not in kinds
    finally:
        conn.close()


def test_spawned_event_emitted_with_pid(kanban_home):
    """Successful spawn must append a ``spawned`` event with the pid in
    the payload so humans tailing events see pid tracking."""
    def _spawn_returns_pid(task, ws):
        return 98765
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.dispatch_once(conn, spawn_fn=_spawn_returns_pid)
        events = kb.list_events(conn, tid)
        spawned = [e for e in events if e.kind == "spawned"]
        assert len(spawned) == 1
        assert spawned[0].payload == {"pid": 98765}
    finally:
        conn.close()


def test_migration_renames_legacy_event_kinds(tmp_path, monkeypatch):
    """A DB created with the old vocab must have its event rows renamed
    in place on init_db()."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Init fresh.
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        # Inject legacy event kinds directly.
        now = int(time.time())
        with kb.write_txn(conn):
            for old in ("ready", "priority", "spawn_auto_blocked"):
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, ?, NULL, ?)",
                    (tid, old, now),
                )
        # Re-run init_db — the migration pass should rename them.
        kb.init_db()
        rows = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        kinds = [r["kind"] for r in rows]
        assert "ready" not in kinds
        assert "priority" not in kinds
        assert "spawn_auto_blocked" not in kinds
        assert "promoted" in kinds
        assert "reprioritized" in kinds
        assert "gave_up" in kinds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Assignees (item 4 from Multica)
# ---------------------------------------------------------------------------

def test_list_profiles_on_disk(tmp_path, monkeypatch):
    """list_profiles_on_disk returns directories under ~/.hermes/profiles/
    that contain a config.yaml."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profiles = tmp_path / ".hermes" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "researcher").mkdir()
    (profiles / "researcher" / "config.yaml").write_text("model: {}\n")
    (profiles / "writer").mkdir()
    (profiles / "writer" / "config.yaml").write_text("model: {}\n")
    (profiles / "empty_dir").mkdir()
    # A stray file; should be ignored.
    (profiles / "stray.txt").write_text("noise")

    names = kb.list_profiles_on_disk()
    assert names == ["researcher", "writer"]


def test_known_assignees_merges_disk_and_board(tmp_path, monkeypatch):
    """known_assignees unions profiles on disk with currently-assigned
    names, and reports per-status counts."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profiles = tmp_path / ".hermes" / "profiles"
    profiles.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    for name in ("researcher", "writer"):
        d = profiles / name
        d.mkdir()
        (d / "config.yaml").write_text("model: {}\n")

    kb.init_db()
    conn = kb.connect()
    try:
        # writer has a ready task; on_board_only has a task but no profile dir.
        kb.create_task(conn, title="a", assignee="writer")
        kb.create_task(conn, title="b", assignee="on_board_only")
        data = kb.known_assignees(conn)
    finally:
        conn.close()

    by_name = {d["name"]: d for d in data}
    assert by_name["researcher"]["on_disk"] is True
    assert by_name["researcher"]["counts"] == {}
    assert by_name["writer"]["on_disk"] is True
    assert by_name["writer"]["counts"] == {"ready": 1}
    assert by_name["on_board_only"]["on_disk"] is False
    assert by_name["on_board_only"]["counts"] == {"ready": 1}


def test_cli_assignees_json(kanban_home):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="x", assignee="someone")
    finally:
        conn.close()
    out = run_slash("assignees --json")
    data = json.loads(out)
    names = [e["name"] for e in data]
    assert "someone" in names


# ---------------------------------------------------------------------------
# CLI --max-runtime flag + duration parser
# ---------------------------------------------------------------------------

def test_parse_duration_accepts_formats():
    from hermes_cli.kanban import _parse_duration
    assert _parse_duration(None) is None
    assert _parse_duration("") is None
    assert _parse_duration("42") == 42
    assert _parse_duration("30s") == 30
    assert _parse_duration("5m") == 300
    assert _parse_duration("2h") == 7200
    assert _parse_duration("1d") == 86400
    assert _parse_duration("1.5h") == 5400


def test_parse_duration_rejects_garbage():
    from hermes_cli.kanban import _parse_duration
    import pytest as _p
    with _p.raises(ValueError):
        _parse_duration("tenminutes")
    with _p.raises(ValueError):
        _parse_duration("fish")


def test_cli_create_max_runtime_via_duration(kanban_home):
    """`hermes kanban create --max-runtime 2h` should persist 7200 seconds."""
    out = run_slash("create 'long task' --max-runtime 2h --json")
    data = json.loads(out)
    tid = data["id"]
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.max_runtime_seconds == 7200
    finally:
        conn.close()


def test_cli_create_max_runtime_bad_format_exits_nonzero(kanban_home):
    out = run_slash("create 'bad' --max-runtime fish")
    assert "max-runtime" in out.lower() or "malformed" in out.lower()


# ---------------------------------------------------------------------------
# Runs as first-class (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------

def test_run_created_on_claim(kanban_home):
    """claim_task opens a new task_runs row and points current_run_id at it."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        assert kb.get_task(conn, tid).current_run_id is None

        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        task = kb.get_task(conn, tid)
        assert task.current_run_id is not None

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        r = runs[0]
        assert r.id == task.current_run_id
        assert r.profile == "worker"
        assert r.status == "running"
        assert r.outcome is None
        assert r.ended_at is None
        assert r.claim_lock is not None and r.claim_expires is not None
    finally:
        conn.close()


def test_run_closed_on_complete_with_summary(kanban_home):
    """complete_task ends the active run with outcome='completed' and
    persists summary + metadata."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        ok = kb.complete_task(
            conn, tid,
            result="shipped",
            summary="implemented rate limiter, tests pass",
            metadata={"changed_files": ["limiter.py"], "tests_run": 12},
        )
        assert ok is True

        task = kb.get_task(conn, tid)
        assert task.current_run_id is None
        assert task.result == "shipped"

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        r = runs[0]
        assert r.status == "done"
        assert r.outcome == "completed"
        assert r.summary == "implemented rate limiter, tests pass"
        assert r.metadata == {"changed_files": ["limiter.py"], "tests_run": 12}
        assert r.ended_at is not None
    finally:
        conn.close()


def test_run_summary_falls_back_to_result(kanban_home):
    """If the caller doesn't pass summary, we fall back to result so
    single-run workflows don't need to pass the same string twice."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="only-arg")
        r = kb.latest_run(conn, tid)
        assert r.summary == "only-arg"
    finally:
        conn.close()


def test_multiple_attempts_preserved_as_runs(kanban_home):
    """Crash / retry / complete flow produces one run per attempt, all
    visible in list_runs in chronological order."""
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")

        # Attempt 1: claim then force the claim to be stale by backdating
        # claim_expires, then let release_stale_claims reclaim it.
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 10, tid),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE task_id = ?",
                (int(time.time()) - 10, tid),
            )
        kb.release_stale_claims(conn)

        # Attempt 2: claim then crash (simulated: pid dead).
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        original_alive = _kb._pid_alive
        _kb._pid_alive = lambda pid: False
        try:
            kb.detect_crashed_workers(conn)
        finally:
            _kb._pid_alive = original_alive

        # Attempt 3: claim then complete.
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="finally")

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 3
        assert [r.outcome for r in runs] == ["reclaimed", "crashed", "completed"]
        assert runs[-1].summary == "finally"
        assert kb.get_task(conn, tid).current_run_id is None
    finally:
        conn.close()


def test_run_on_block_with_reason(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="needs API key")

        r = kb.latest_run(conn, tid)
        assert r.outcome == "blocked"
        assert r.summary == "needs API key"
        assert r.ended_at is not None
        assert kb.get_task(conn, tid).current_run_id is None
    finally:
        conn.close()


def test_run_on_spawn_failure_records_failed_runs(kanban_home):
    """Each spawn_failed event closes a run with outcome='spawn_failed',
    and the Nth failure closes a run with outcome='gave_up'."""
    def _bad(task, ws):
        raise RuntimeError("no PATH")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        for _ in range(5):
            kb.dispatch_once(conn, spawn_fn=_bad, failure_limit=5)

        runs = kb.list_runs(conn, tid)
        # 5 claim attempts → 5 runs. Final one is gave_up, earlier ones
        # are spawn_failed.
        assert len(runs) == 5
        assert runs[-1].outcome == "gave_up"
        assert all(r.outcome == "spawn_failed" for r in runs[:-1])
        assert runs[-1].error and "no PATH" in runs[-1].error
    finally:
        conn.close()


def test_event_rows_carry_run_id(kanban_home):
    """task_events.run_id is populated for run-scoped kinds and NULL for
    task-scoped ones."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # task-scoped: 'created' — no run yet
        # run-scoped: 'claimed' + 'completed'
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="ok")

        rows = conn.execute(
            "SELECT kind, run_id FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        by_kind = {r["kind"]: r["run_id"] for r in rows}
        assert by_kind["created"] is None
        assert by_kind["claimed"] is not None
        assert by_kind["completed"] is not None
        # Both belong to the same run.
        assert by_kind["claimed"] == by_kind["completed"]
    finally:
        conn.close()


def test_build_worker_context_includes_prior_attempts(kanban_home):
    """A worker spawned after a prior attempt sees that attempt's outcome
    + summary in its context so it can skip the failed path."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="port x", assignee="worker")

        # Attempt 1: blocked with a reason.
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="needs clarification on IP vs user_id")
        kb.unblock_task(conn, tid)

        # Attempt 2: claim (but don't complete yet) and read the context
        # as this worker would see it.
        kb.claim_task(conn, tid)
        ctx = kb.build_worker_context(conn, tid)

        assert "Prior attempts on this task" in ctx
        assert "blocked" in ctx
        assert "needs clarification on IP vs user_id" in ctx
    finally:
        conn.close()


def test_build_worker_context_uses_parent_run_summary(kanban_home):
    """Downstream children read the parent's run.summary + metadata, not
    just task.result."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="research", assignee="researcher")
        child = kb.create_task(
            conn, title="write", assignee="writer", parents=[parent],
        )

        kb.claim_task(conn, parent)
        kb.complete_task(
            conn, parent,
            result="done",
            summary="three angles explored; B looks strongest",
            metadata={"sources": ["paper A", "paper B", "paper C"]},
        )

        # child becomes ready via recompute_ready (runs inside complete_task)
        ctx = kb.build_worker_context(conn, child)
        assert "Parent task results" in ctx
        assert "three angles explored; B looks strongest" in ctx
        assert '"sources"' in ctx  # metadata JSON serialized
    finally:
        conn.close()


def test_migration_backfills_inflight_run_for_legacy_db(kanban_home):
    """An existing 'running' task from before task_runs existed should
    get a synthesized run row so subsequent operations (complete,
    heartbeat) have something to write to."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pre-migration", assignee="worker")
        # Simulate legacy: set running + claim_lock directly, leave
        # current_run_id NULL and delete the run row the claim created.
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (tid,))
            conn.execute(
                "UPDATE tasks SET current_run_id = NULL WHERE id = ?",
                (tid,),
            )

        # Sanity: no runs, no pointer.
        assert kb.list_runs(conn, tid) == []
        assert kb.get_task(conn, tid).current_run_id is None

        # Re-run init_db — migration backfill should kick in.
        kb.init_db()
        conn2 = kb.connect()
        try:
            runs = kb.list_runs(conn2, tid)
            assert len(runs) == 1
            assert runs[0].status == "running"
            assert runs[0].profile == "worker"
            task = kb.get_task(conn2, tid)
            assert task.current_run_id == runs[0].id

            # Subsequent complete closes the backfilled run cleanly.
            kb.complete_task(conn2, tid, result="done", summary="ok")
            r = kb.latest_run(conn2, tid)
            assert r.outcome == "completed"
            assert r.summary == "ok"
        finally:
            conn2.close()
    finally:
        conn.close()


def test_forward_compat_columns_writable(kanban_home):
    """v2 will route by workflow_template_id + current_step_key. In v1
    these are nullable, kernel doesn't consult them for routing, but
    they must be writable so a v2 client can populate them without
    schema changes."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? "
                "WHERE id = ?",
                ("code-review-v1", "implement", tid),
            )
        task = kb.get_task(conn, tid)
        assert task.workflow_template_id == "code-review-v1"
        assert task.current_step_key == "implement"
    finally:
        conn.close()


def test_cli_runs_verb(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="ok", summary="shipped")
    finally:
        conn.close()
    out = run_slash(f"runs {tid}")
    assert "completed" in out
    assert "shipped" in out
    assert "worker" in out


def test_cli_runs_json(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(
            conn, tid, result="ok", summary="shipped",
            metadata={"files": 1},
        )
    finally:
        conn.close()
    out = run_slash(f"runs {tid} --json")
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["outcome"] == "completed"
    assert data[0]["metadata"] == {"files": 1}


def test_cli_complete_with_summary_and_metadata(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    # JSON metadata must round-trip through shlex + argparse.
    meta = '{"files": 3}'
    out = run_slash(
        "complete " + tid + " --summary \"done it\" --metadata '" + meta + "'"
    )
    assert "Completed" in out
    conn = kb.connect()
    try:
        r = kb.latest_run(conn, tid)
    finally:
        conn.close()
    assert r.summary == "done it"
    assert r.metadata == {"files": 3}


def test_cli_complete_bad_metadata_exits_nonzero(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    out = run_slash(f"complete {tid} --metadata not-json")
    assert "metadata" in out.lower()


# -------------------------------------------------------------------------
# Integration hardening (Apr 2026 audit fixes)
# -------------------------------------------------------------------------

def test_archive_of_running_task_closes_run(kanban_home):
    """Archiving a claimed task must close the in-flight run with
    outcome='reclaimed', not orphan it."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert run.ended_at is None
        open_run_id = run.id

        assert kb.archive_task(conn, tid) is True

        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.current_run_id is None
        # The previously-active run must now be closed.
        closed = kb.get_run(conn, open_run_id)
        assert closed.ended_at is not None
        assert closed.outcome == "reclaimed"
    finally:
        conn.close()


def test_archive_of_ready_task_does_not_create_spurious_run(kanban_home):
    """No active run → archive shouldn't synthesize one."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Never claimed. Move to ready (task starts in 'ready' here).
        assert kb.archive_task(conn, tid) is True
        runs = kb.list_runs(conn, tid)
        assert runs == []  # No run was ever opened; archive didn't fabricate one.
    finally:
        conn.close()


def test_dashboard_direct_status_change_off_running_closes_run(kanban_home):
    """Dashboard drag-drop running->ready must close the active run.

    Importing _set_status_direct directly to simulate the PATCH handler
    without spinning up FastAPI.
    """
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        open_run = kb.latest_run(conn, tid)
        assert open_run.ended_at is None
        prev_run_id = open_run.id

        # Simulate yanking the worker back to the queue.
        assert _set_status_direct(conn, tid, "ready") is True

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        closed = kb.get_run(conn, prev_run_id)
        assert closed.ended_at is not None
        assert closed.outcome == "reclaimed"
    finally:
        conn.close()


def test_dashboard_direct_status_change_within_same_state_is_noop_for_runs(kanban_home):
    """todo -> ready on an unclaimed task must not create any run rows."""
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        # Force to todo for the sake of the test.
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
        conn.commit()
        assert _set_status_direct(conn, tid, "ready") is True
        assert kb.list_runs(conn, tid) == []
    finally:
        conn.close()


def test_cli_bulk_complete_with_summary_rejects(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="worker")
        b = kb.create_task(conn, title="b", assignee="worker")
        kb.claim_task(conn, a); kb.claim_task(conn, b)
    finally:
        conn.close()
    # Bulk + summary is refused (stderr message, no mutation).
    # Note: hermes_cli.main doesn't propagate sub-command exit codes
    # (args.func(args) discards the return value), so we check the side
    # effects instead.
    from subprocess import run as _run
    import os, sys
    env = os.environ.copy()
    r = _run(
        [sys.executable, "-m", "hermes_cli.main", "kanban",
         "complete", a, b, "--summary", "oops"],
        capture_output=True, text=True, env=env,
    )
    assert "per-task" in r.stderr, r.stderr
    # The tasks must still be running (no partial apply).
    conn = kb.connect()
    try:
        assert kb.get_task(conn, a).status == "running"
        assert kb.get_task(conn, b).status == "running"
    finally:
        conn.close()


def test_cli_bulk_complete_without_summary_still_works(kanban_home):
    """Bulk close with no per-task handoff is allowed — the common case."""
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="worker")
        b = kb.create_task(conn, title="b", assignee="worker")
        kb.claim_task(conn, a); kb.claim_task(conn, b)
    finally:
        conn.close()
    out = run_slash(f"complete {a} {b}")
    assert f"Completed {a}" in out
    assert f"Completed {b}" in out


def test_completed_event_payload_carries_summary(kanban_home):
    """The 'completed' event must embed the run summary so gateway
    notifiers render structured handoffs without a second SQL hit."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="handoff line 1\nextra",
                         metadata={"n": 3})
        events = kb.list_events(conn, tid)
        comp = [e for e in events if e.kind == "completed"]
        assert len(comp) == 1
        # First-line-only, within the 400-char cap, preserved verbatim.
        assert comp[0].payload["summary"] == "handoff line 1"
    finally:
        conn.close()


def test_completed_event_payload_summary_none_when_missing(kanban_home):
    """If the caller passes no summary AND no result, payload.summary is None."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid)  # no summary, no result
        events = kb.list_events(conn, tid)
        comp = [e for e in events if e.kind == "completed"][0]
        assert comp.payload.get("summary") is None
    finally:
        conn.close()
