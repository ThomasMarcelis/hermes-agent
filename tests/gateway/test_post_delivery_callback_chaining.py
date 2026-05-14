"""Tests for ``BasePlatformAdapter.register_post_delivery_callback`` chaining.

When two features want to run after the final response lands on the same
session (e.g. background-review release + temporary-progress cleanup), the
registration API chains them rather than clobbering. Per-callback
exceptions are swallowed so one bad callback can't sabotage the others.
Stale-generation registrations are rejected.
"""
import asyncio
import inspect

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _MinAdapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.fixture
def adapter():
    return _MinAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)


class TestPostDeliveryCallbackChaining:
    def test_single_callback_fires(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        cb = adapter.pop_post_delivery_callback("s")
        cb()
        assert fired == ["A"]

    def test_two_callbacks_chain_in_order(self, adapter):
        fired = []
        adapter.register_post_delivery_callback("s", lambda: fired.append("A"))
        adapter.register_post_delivery_callback("s", lambda: fired.append("B"))
        cb = adapter.pop_post_delivery_callback("s")
        cb()
        assert fired == ["A", "B"]

    def test_three_callbacks_chain_in_order(self, adapter):
        """Chain composes over an already-chained callback."""
        fired = []
        for label in ("A", "B", "C"):
            adapter.register_post_delivery_callback(
                "s", lambda x=label: fired.append(x)
            )
        cb = adapter.pop_post_delivery_callback("s")
        cb()
        assert fired == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_chained_callback_awaits_async_callbacks_in_order(self, adapter):
        """Async post-delivery callbacks must not be silently discarded.

        Goal-status delivery registers an async callback and background-review
        release can already have a sync callback in the same slot.  The chained
        callback must await the async leg before running later callbacks so the
        visible status line is actually sent and ordering stays deterministic.
        """
        fired = []

        async def first_async():
            await asyncio.sleep(0)
            fired.append("A")

        def second_sync():
            fired.append("B")

        adapter.register_post_delivery_callback("s", first_async)
        adapter.register_post_delivery_callback("s", second_sync)
        cb = adapter.pop_post_delivery_callback("s")
        result = cb()
        if inspect.isawaitable(result):
            await result
        assert fired == ["A", "B"]

    @pytest.mark.asyncio
    async def test_chained_callback_isolates_async_callback_exception(self, adapter):
        fired = []

        async def boom_async():
            await asyncio.sleep(0)
            raise RuntimeError("boom")

        adapter.register_post_delivery_callback("s", boom_async)
        adapter.register_post_delivery_callback("s", lambda: fired.append("survived"))
        cb = adapter.pop_post_delivery_callback("s")
        result = cb()
        if inspect.isawaitable(result):
            await result
        assert fired == ["survived"]

    def test_exception_in_one_callback_does_not_block_next(self, adapter):
        fired = []

        def boom():
            raise ValueError("boom")

        adapter.register_post_delivery_callback("s", boom)
        adapter.register_post_delivery_callback("s", lambda: fired.append("survived"))
        cb = adapter.pop_post_delivery_callback("s")
        cb()
        assert fired == ["survived"]

    def test_same_generation_chains(self, adapter):
        fired = []
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("A"), generation=5
        )
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("B"), generation=5
        )
        cb = adapter.pop_post_delivery_callback("s", generation=5)
        cb()
        assert fired == ["A", "B"]

    def test_stale_generation_registration_rejected(self, adapter):
        """A registration with an older generation than the existing
        entry is rejected — it doesn't clobber the newer run's slot."""
        fired = []
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("gen7"), generation=7
        )
        adapter.register_post_delivery_callback(
            "s", lambda: fired.append("stale_gen3"), generation=3
        )
        cb = adapter.pop_post_delivery_callback("s", generation=7)
        cb()
        assert fired == ["gen7"]

    def test_pop_at_wrong_generation_returns_none(self, adapter):
        adapter.register_post_delivery_callback(
            "s", lambda: None, generation=5
        )
        assert adapter.pop_post_delivery_callback("s", generation=99) is None
        # Correct generation still finds it.
        assert adapter.pop_post_delivery_callback("s", generation=5) is not None

    def test_empty_session_key_is_noop(self, adapter):
        adapter.register_post_delivery_callback("", lambda: None)
        assert adapter._post_delivery_callbacks == {}

    def test_non_callable_is_noop(self, adapter):
        adapter.register_post_delivery_callback("s", "not-callable")  # type: ignore[arg-type]
        assert adapter._post_delivery_callbacks == {}
