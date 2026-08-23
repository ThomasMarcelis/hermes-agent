"""Regression coverage for retained Discord thread-per-run cron delivery."""

from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler_delivery import _deliver_result, _open_continuable_cron_thread
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult


def test_discord_thread_per_run_reaches_standalone_delivery():
    pconfig = MagicMock(enabled=True)
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.DISCORD: pconfig}

    with (
        patch("gateway.config.load_gateway_config", return_value=mock_cfg),
        patch(
            "tools.send_message_tool._send_to_platform",
            new=AsyncMock(return_value={"success": True, "thread_id": "t1"}),
        ) as send_mock,
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("cron.scheduler._hermes_now") as now_mock,
    ):
        now_mock.return_value.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2026-07-20",
            "%H:%M": "08:39",
            "%Y-%m-%d %H:%M": "2026-07-20 08:39",
        }[fmt]
        _deliver_result(
            {
                "id": "daily",
                "name": "Daily report",
                "deliver": "discord:parent-1",
                "delivery_options": {
                    "discord": {
                        "thread_per_run": True,
                        "thread_name_template": "{job_name} {date}",
                        "thread_auto_archive_duration": 60,
                    }
                },
            },
            "Clean output only.",
        )

    assert send_mock.call_args.kwargs["discord_thread_name"] == "Daily report 2026-07-20"
    assert send_mock.call_args.kwargs["discord_thread_auto_archive_duration"] == 60


def test_open_discord_cron_thread_carries_archive_duration():
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="9001")

    def _run_now(coro, _loop):
        coro.close()
        future = MagicMock()
        future.result.return_value = "9001"
        return future

    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_run_now):
        thread_id = _open_continuable_cron_thread(
            {"id": "j1"},
            adapter,
            "123",
            MagicMock(),
            thread_name="Daily",
            thread_auto_archive_duration=60,
        )

    assert thread_id == "9001"
    adapter.create_handoff_thread.assert_called_once_with(
        "123", "Daily", auto_archive_duration=60
    )


def test_thread_per_run_and_continuable_open_only_one_thread():
    adapter = AsyncMock()
    adapter.send.return_value = SendResult(success=True)
    adapter._session_store = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True

    def _run_coro(coro, _loop):
        future = Future()
        try:
            import asyncio

            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    job = {
        "id": "daily",
        "name": "Daily",
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "123"},
        "attach_to_session": True,
        "delivery_options": {
            "discord": {
                "thread_per_run": True,
                "thread_auto_archive_duration": 60,
            }
        },
    }
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True)},
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._open_continuable_cron_thread", return_value="thread-1") as open_thread, \
         patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_run_coro), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        result = _deliver_result(
            job, "Brief", adapters={Platform.DISCORD: adapter}, loop=loop,
        )

    assert result is None
    open_thread.assert_called_once()
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["metadata"]["thread_id"] == "thread-1"
