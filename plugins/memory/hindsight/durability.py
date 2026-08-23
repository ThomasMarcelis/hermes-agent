"""Hindsight delivery ledger and bounded remote settlement.

Each ledger keeps immutable session/document ownership until every accepted range
is durable, independently of subsequent provider session switches.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)
_MAX_AUTOMATIC_SETTLEMENT_RETRIES = 2
_SETTLEMENT_RETRY_BASE_DELAY_S = 0.05
_RETAIN_OP_PREFETCH_POLL_CAP = 4
_RETAIN_OP_DRAIN_POLL_CAP = 4

@dataclass
class _DeliveryState:
    """Durability ledger for one immutable session/document ownership."""

    turns: list[str]
    bank_id: str
    session_id: str
    parent_session_id: str
    document_id: str
    update_mode: str | None
    retain_async: bool
    context: str
    metadata: Dict[str, str]
    tags: tuple[str, ...]
    committed: int = 0
    queued: int = 0
    in_flight: tuple[int, int] | None = None
    job_queued: bool = False
    remote_pending: bool = False
    force_settle: bool = False
    delivery_due: bool = False
    failed: bool = False
    invalidated: bool = False
    automatic_retries: int = 0
    retry_scheduled: bool = False


@dataclass(frozen=True)
class _RetainOpOwnership:
    """Immutable ownership for one accepted remote operation group."""

    bank_id: str
    operation_ids: frozenset[str]
    on_completed: Optional[Callable[[], None]]
    on_failed: Optional[Callable[[], None]]


class HindsightDurabilityMixin:
    def _track_retain_ops(
        self,
        retain_response,
        bank_id: str,
        *,
        on_completed: Optional[Callable[[], None]] = None,
        on_failed: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Record server-side async operation id(s) from an aretain_batch reply.

        Async retains return ``operation_id`` / ``operation_ids`` that stay
        ``pending`` on the server until the write is durable and recall-visible.
        The bank_id is captured alongside so completion can be polled with the
        same bank the write targeted.
        """
        ids: list[str] = []
        single = getattr(retain_response, "operation_id", None)
        if isinstance(single, (str, int)) and str(single):
            ids.append(str(single))
        multiple = getattr(retain_response, "operation_ids", None)
        if isinstance(multiple, (list, tuple, set)):
            ids.extend(str(op) for op in multiple if op)
        if not ids:
            # Server didn't hand back an op id (older API, or it completed
            # synchronously). Nothing to poll — local queue drain is the only
            # available signal in that case.
            return False
        with self._pending_retain_ops_lock:
            self._retain_op_group_sequence += 1
            group_id = self._retain_op_group_sequence
            unique_ids = frozenset(ids)
            self._retain_op_groups[group_id] = _RetainOpOwnership(
                bank_id=bank_id,
                operation_ids=unique_ids,
                on_completed=on_completed,
                on_failed=on_failed,
            )
            self._retain_op_remaining[group_id] = set(unique_ids)
            for op_id in unique_ids:
                self._retain_op_records[op_id] = group_id
                self._retain_op_poll_state[op_id] = {
                    "attempts": 0,
                    "next_poll_at": 0.0,
                }
            self._pending_retain_ops.update(unique_ids)
        return True


    def _retain_op_status(
        self,
        bank_id: str,
        op_id: str,
        *,
        timeout: float | None = None,
    ) -> str:
        """Return ``completed``, ``failed``, or ``pending`` for one op."""
        try:
            resp = self._run_hindsight_operation(
                lambda client: client.operations.get_operation_status(
                    bank_id=bank_id, operation_id=op_id
                ),
                timeout=timeout,
            )
        except Exception as exc:
            if (
                type(exc).__name__ == "NotFoundException"
                or getattr(exc, "status", None) == 404
                or getattr(exc, "status_code", None) == 404
            ):
                return "completed"
            logger.debug("Retain operation status check failed for %s: %s", op_id, exc)
            return "pending"
        status = str(getattr(resp, "status", "") or "").lower()
        return status if status in {"completed", "failed"} else "pending"


    def _settle_retain_op(self, op_id: str, status: str) -> None:
        """Settle an accepted op through its immutable group ownership."""
        callback: Optional[Callable[[], None]] = None
        with self._pending_retain_ops_lock:
            group_id = self._retain_op_records.pop(op_id, None)
            self._pending_retain_ops.discard(op_id)
            self._retain_op_poll_state.pop(op_id, None)
            if group_id is None:
                return
            ownership = self._retain_op_groups.get(group_id)
            remaining = self._retain_op_remaining.get(group_id)
            if ownership is None or remaining is None:
                return
            if status == "failed":
                for sibling in list(remaining):
                    self._pending_retain_ops.discard(sibling)
                    self._retain_op_records.pop(sibling, None)
                    self._retain_op_poll_state.pop(sibling, None)
                self._retain_op_groups.pop(group_id, None)
                self._retain_op_remaining.pop(group_id, None)
                callback = ownership.on_failed
            else:
                remaining.discard(op_id)
                if not remaining:
                    self._retain_op_groups.pop(group_id, None)
                    self._retain_op_remaining.pop(group_id, None)
                    callback = ownership.on_completed
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                logger.warning("Hindsight retain settlement callback failed: %s", exc)


    def _wait_for_retains_drained(self, timeout: float, *, purpose: str = "prefetch") -> bool:
        """Block up to *timeout* seconds for the just-completed turn's retain to
        become recall-visible on the server.

        Used by the background prefetch so the next turn's recall observes the
        just-completed turn's write instead of racing ahead of it. Runs only on
        the background prefetch thread — never on the reply path.

        Two ordered barriers, both bounded by the shared *timeout* budget:

        1. Local writer queue drains (the retain call has been *dispatched* to
           the server). Polls ``unfinished_tasks`` rather than ``queue.join()``
           so a wedged write can't hang the prefetch.
        2. Server-side async operations complete. With ``retain_async=True`` the
           dispatched call returns on *acceptance*, not durability, so draining
           the local queue alone is NOT a read-after-write signal. We poll
           ``get_operation_status`` for the tracked op id(s) until the server
           reports completion (an explicit read-after-write condition).

        Returns True if both barriers cleared within the budget, False on
        timeout/shutdown.
        """
        deadline = None if timeout <= 0 else time.monotonic() + timeout

        def _expired() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        while True:
            # Barrier 1: local queue drain (retain dispatched to the server).
            while self._retain_queue.unfinished_tasks > 0:
                if self._shutting_down.is_set() and purpose != "shutdown":
                    return False
                if _expired():
                    logger.debug(
                        "%s: retain drain timed out after %.1fs (%d pending)",
                        purpose, timeout, self._retain_queue.unfinished_tasks,
                    )
                    return False
                time.sleep(min(0.05, max(0.0, (deadline or time.monotonic() + 0.05) - time.monotonic())))

            # Completion/failure callbacks may enqueue a remainder or retry.
            if not self._wait_for_server_retain_ops(deadline, timeout, purpose=purpose):
                return False
            if self._retain_queue.unfinished_tasks == 0:
                return True


    def _wait_for_server_retain_ops(
        self,
        deadline: float | None,
        timeout: float,
        *,
        purpose: str = "prefetch",
    ) -> bool:
        """Poll tracked async retain ops until complete or the deadline passes.

        *deadline* is a ``time.monotonic()`` value (None = no bound). Completed
        ops are removed from the pending set as they finish so a later prefetch
        doesn't re-poll them.

        Unresolved accepted operations stay tracked. Per-call poll caps plus a
        persistent exponential ``next_poll_at`` prevent successive prefetches
        from re-burning the full wait budget while preserving a later
        lifecycle/shutdown chance to prove durability.
        """
        per_op_polls: dict[str, int] = {}
        poll_cap = (
            _RETAIN_OP_PREFETCH_POLL_CAP
            if purpose == "prefetch"
            else _RETAIN_OP_DRAIN_POLL_CAP
        )
        while True:
            with self._pending_retain_ops_lock:
                pending = []
                for op_id in self._pending_retain_ops:
                    group_id = self._retain_op_records.get(op_id)
                    ownership = self._retain_op_groups.get(group_id) if group_id is not None else None
                    if ownership is not None:
                        state = self._retain_op_poll_state.get(op_id, {})
                        pending.append(
                            (
                                op_id,
                                ownership.bank_id,
                                float(state.get("next_poll_at", 0.0)),
                            )
                        )
            if not pending:
                return True
            if self._shutting_down.is_set() and purpose != "shutdown":
                return False
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                logger.warning(
                    "%s: server retain visibility timed out after %.1fs; "
                    "keeping %d unresolved op(s) retryable",
                    purpose, timeout, len(pending),
                )
                return False

            due = [
                (op_id, bank_id)
                for op_id, bank_id, next_poll_at in pending
                if per_op_polls.get(op_id, 0) < poll_cap
                and (purpose != "prefetch" or next_poll_at <= now)
            ]
            if not due:
                if purpose == "prefetch":
                    next_due = min(next_poll_at for _, _, next_poll_at in pending)
                    capped = all(per_op_polls.get(op_id, 0) >= poll_cap for op_id, _, _ in pending)
                    if capped:
                        logger.warning(
                            "Prefetch: retaining %d unresolved operation(s) after bounded polling",
                            len(pending),
                        )
                        return False
                    sleep_for = max(0.0, next_due - now)
                    if deadline is not None and now + sleep_for >= deadline:
                        return False
                    time.sleep(min(sleep_for, self._RETAIN_OP_POLL_INTERVAL_S))
                    continue
                due = [
                    (op_id, bank_id)
                    for op_id, bank_id, _ in pending
                    if per_op_polls.get(op_id, 0) < poll_cap
                ]
                if not due:
                    return False

            for op_id, bank_id in due:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return False
                per_op_polls[op_id] = per_op_polls.get(op_id, 0) + 1
                status = self._retain_op_status(bank_id, op_id, timeout=remaining)
                if status != "pending":
                    self._settle_retain_op(op_id, status)
                    continue
                with self._pending_retain_ops_lock:
                    poll_state = self._retain_op_poll_state.get(op_id)
                    if poll_state is not None:
                        attempts = int(poll_state.get("attempts", 0)) + 1
                        poll_state["attempts"] = attempts
                        poll_state["next_poll_at"] = time.monotonic() + min(
                            self._RETAIN_OP_POLL_INTERVAL_S * (2 ** max(0, attempts - 1)),
                            30.0,
                        )

            if purpose != "prefetch":
                with self._pending_retain_ops_lock:
                    if not self._pending_retain_ops:
                        return True
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(min(self._RETAIN_OP_POLL_INTERVAL_S, max(0.0, (deadline or time.monotonic() + self._RETAIN_OP_POLL_INTERVAL_S) - time.monotonic())))


    def _current_delivery_state(self) -> _DeliveryState:
        """Return the ledger bound to the active session buffer."""
        with self._delivery_lock:
            state = self._active_delivery_state
            if state is not None and state.turns is self._session_turns:
                # Completed states are pruned to avoid retaining every prior
                # transcript forever.  Re-admit the active state when new
                # below-threshold turns later make it relevant again.
                if not any(candidate is state for candidate in self._delivery_states):
                    self._delivery_states.append(state)
                return state

            document_id, update_mode = self._resolve_retain_target(self._document_id)
            lineage_tags: list[str] = []
            if self._session_id:
                lineage_tags.append(f"session:{self._session_id}")
            if self._parent_session_id:
                lineage_tags.append(f"parent:{self._parent_session_id}")
            state = _DeliveryState(
                turns=self._session_turns,
                bank_id=self._bank_id,
                session_id=self._session_id,
                parent_session_id=self._parent_session_id,
                document_id=document_id,
                update_mode=update_mode,
                retain_async=self._retain_async,
                context=self._retain_context,
                metadata=self._build_metadata(message_count=0, turn_index=self._turn_index),
                tags=tuple(lineage_tags),
            )
            self._active_delivery_state = state
            self._delivery_states.append(state)
            return state


    def _sync_active_delivery_watermarks(self, state: _DeliveryState) -> None:
        if self._active_delivery_state is state:
            self._last_retained_turn_count = state.committed
            self._queued_retained_turn_count = state.queued


    def _mark_delivery_failed(self, state: _DeliveryState) -> None:
        with self._delivery_lock:
            if state.invalidated:
                return
            state.queued = state.committed
            state.in_flight = None
            state.job_queued = False
            state.remote_pending = False
            state.failed = True
            self._sync_active_delivery_watermarks(state)


    def _schedule_delivery_retry(self, state: _DeliveryState) -> bool:
        """Schedule one bounded exponentially-delayed retry for *state*."""
        if self._shutting_down.is_set():
            return False
        with self._delivery_lock:
            if (
                state.invalidated
                or state.in_flight is not None
                or state.retry_scheduled
                or state.automatic_retries >= _MAX_AUTOMATIC_SETTLEMENT_RETRIES
            ):
                return False
            state.automatic_retries += 1
            state.retry_scheduled = True
            retry_number = state.automatic_retries
        delay = _SETTLEMENT_RETRY_BASE_DELAY_S * (2 ** (retry_number - 1))

        def _retry() -> None:
            time.sleep(delay)
            with self._delivery_lock:
                state.retry_scheduled = False
            self._queue_delivery(state, force=state.force_settle, retry=True)

        self._enqueue_retain(_retry)
        return True


    def _complete_delivery(self, state: _DeliveryState, end: int) -> None:
        queue_more = False
        with self._delivery_lock:
            if state.invalidated:
                return
            state.committed = max(state.committed, end)
            state.queued = max(state.queued, end)
            state.in_flight = None
            state.job_queued = False
            state.remote_pending = False
            state.failed = False
            state.automatic_retries = 0
            queue_more = (
                state.committed < len(state.turns)
                and (state.force_settle or state.delivery_due)
            )
            if not queue_more and state.committed >= len(state.turns):
                state.force_settle = False
                self._delivery_states = [
                    candidate for candidate in self._delivery_states
                    if candidate is not state
                ]
            self._sync_active_delivery_watermarks(state)
        if queue_more:
            self._queue_delivery(state, force=state.force_settle, retry=True)


    def _queue_delivery(
        self,
        state: _DeliveryState,
        *,
        force: bool,
        retry: bool = False,
    ) -> bool:
        """Queue one immutable append range or legacy full-document write."""
        if self._shutting_down.is_set():
            return False
        with self._delivery_lock:
            if state.invalidated:
                return False
            # Once a failed state has exhausted its retry budget, keep its
            # immutable ledger entry unresolved and report the failed drain.
            # Re-admitting it on every later prefetch/switch/shutdown barrier
            # would be an unbounded "re-burn" and, in overwrite mode, could
            # repeatedly rewrite the same document forever.
            if (
                state.failed
                and state.automatic_retries >= _MAX_AUTOMATIC_SETTLEMENT_RETRIES
                and not retry
            ):
                return False
            # A delayed retry already owns the next admission.  Do not let a
            # concurrent lifecycle drain enqueue the same range beside it.
            if state.retry_scheduled and not retry:
                return False
            if force:
                state.force_settle = True
            else:
                state.delivery_due = True
            if state.in_flight is not None or state.job_queued or state.remote_pending:
                return False
            end = len(state.turns)
            if end <= state.committed:
                state.delivery_due = False
                state.force_settle = False
                self._sync_active_delivery_watermarks(state)
                return False
            start = state.committed if state.update_mode == "append" else 0
            turns = list(state.turns[start:end]) if state.update_mode == "append" else list(state.turns[:end])
            if not turns:
                return False
            state.in_flight = (start, end)
            state.job_queued = True
            state.remote_pending = False
            state.queued = end
            state.failed = False
            state.delivery_due = False
            if not any(candidate is state for candidate in self._delivery_states):
                self._delivery_states.append(state)
            self._sync_active_delivery_watermarks(state)

        content = "[" + ",".join(turns) + "]"
        metadata = dict(state.metadata)
        metadata["retained_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        metadata["turn_index"] = str(end)
        metadata["message_count"] = str(len(turns) * 2)
        expected_range = (start, end)

        def _deliver() -> None:
            with self._delivery_lock:
                owns_range = (
                    not state.invalidated
                    and state.in_flight == expected_range
                    and end <= len(state.turns)
                    and (
                        list(state.turns[start:end]) == turns
                        if state.update_mode == "append"
                        else list(state.turns[:end]) == turns
                    )
                )
                if not owns_range:
                    if state.in_flight == expected_range:
                        state.in_flight = None
                        state.job_queued = False
                        state.queued = state.committed
                    return

            item = self._build_retain_kwargs(
                content,
                context=state.context,
                metadata=metadata,
                tags=list(state.tags) or None,
            )
            if state.update_mode is not None:
                item["update_mode"] = state.update_mode
            logger.debug(
                "Hindsight retain: bank=%s session=%s doc=%s mode=%s "
                "range=%d:%d async=%s",
                state.bank_id,
                state.session_id,
                state.document_id,
                state.update_mode,
                start,
                end,
                state.retain_async,
            )
            try:
                resp = self._retain_batch(
                    item, bank_id=state.bank_id, document_id=state.document_id,
                    retain_async=state.retain_async,
                )
            except Exception:
                self._mark_delivery_failed(state)
                if state.force_settle:
                    self._schedule_delivery_retry(state)
                raise

            if state.retain_async:
                def _completed() -> None:
                    self._complete_delivery(state, end)

                def _failed() -> None:
                    self._mark_delivery_failed(state)
                    if not self._schedule_delivery_retry(state):
                        logger.warning(
                            "Hindsight remote retain failed after bounded retries; "
                            "session=%s range=%d:%d remains retryable",
                            state.session_id,
                            start,
                            end,
                        )

                with self._delivery_lock:
                    state.job_queued = False
                    state.remote_pending = True
                if self._track_retain_ops(
                    resp,
                    state.bank_id,
                    on_completed=_completed,
                    on_failed=_failed,
                ):
                    logger.debug("Hindsight retain accepted; awaiting remote durability")
                    return
            self._complete_delivery(state, end)

        self._enqueue_retain(_deliver)
        return True


    def _force_settle_delivery_states(self) -> None:
        """Admit all retryable/partial suffixes without duplicating in-flight ranges."""
        if self._session_turns:
            self._current_delivery_state()
        with self._delivery_lock:
            states = list(self._delivery_states)
        for state in states:
            if not state.invalidated and state.committed < len(state.turns):
                self._queue_delivery(state, force=True)


    def _durability_state_clear(self) -> bool:
        with self._pending_retain_ops_lock:
            if self._pending_retain_ops:
                return False
        if self._retain_queue.unfinished_tasks:
            return False
        with self._delivery_lock:
            return not any(
                not state.invalidated and state.committed < len(state.turns)
                for state in self._delivery_states
            )


    def _settle_pending_until(self, deadline: float | None, *, purpose: str) -> bool:
        timeout = 0.0 if deadline is None else max(0.0, deadline - time.monotonic())
        for attempt in range(_MAX_AUTOMATIC_SETTLEMENT_RETRIES + 1):
            self._force_settle_delivery_states()
            while self._retain_queue.unfinished_tasks:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)

            with self._pending_retain_ops_lock:
                has_remote = bool(self._pending_retain_ops)
            if has_remote and not self._wait_for_server_retain_ops(
                deadline,
                timeout,
                purpose=purpose,
            ):
                return False
            if self._durability_state_clear():
                return True
            if attempt < _MAX_AUTOMATIC_SETTLEMENT_RETRIES:
                delay = _SETTLEMENT_RETRY_BASE_DELAY_S * (2 ** attempt)
                if deadline is not None and time.monotonic() + delay >= deadline:
                    return False
                time.sleep(delay)
        return self._durability_state_clear()


    def drain_pending(self, timeout: float | None = None) -> bool:
        """Extend the durability barrier through local and accepted remote work."""
        budget = self._prefetch_retain_drain_timeout if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + budget
        ok = self._settle_pending_until(deadline, purpose="drain")
        self._last_drain_ok = ok
        if not ok:
            logger.warning(
                "Hindsight durability drain exhausted %.2fs; unresolved accepted "
                "operations and retryable suffixes were retained",
                budget,
            )
        return ok
