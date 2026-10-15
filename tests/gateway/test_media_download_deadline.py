"""Bound the entire streamed download, including cancellation cleanup."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.platforms.base import cache_image_from_url, download_media_bytes_from_url


class DripBody(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        # Every read makes progress well within the I/O timeout; the full body exceeds it.
        for _ in range(80):
            yield b"x"
            await asyncio.sleep(0.05)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["deadline", "caller_cancel"])
@pytest.mark.parametrize("phase", ["body", "preflight", "redirect"])
async def test_download_deadline_and_cancellation_close_stream(monkeypatch, stop, phase):
    body = DripBody()
    response = (httpx.Response(302, headers={"location": "https://redirect.example/file"})
                if phase == "redirect" else httpx.Response(200, stream=body))
    clients = []
    safety_started = asyncio.Event()
    safety_release = asyncio.Event()

    async def safe_url(url):
        if phase == "preflight" or (phase == "redirect" and "redirect.example" in url):
            safety_started.set()
            await safety_release.wait()
        return True

    monkeypatch.setattr("tools.url_safety.async_is_safe_url", safe_url)

    def make_client(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response), **kwargs)
        clients.append(client)
        return client

    # Only the transport is synthetic: real HTTPX streaming and cleanup still run.
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", make_client)
    task = asyncio.create_task(download_media_bytes_from_url(
        "https://93.184.216.34/attachment", timeout=2, max_bytes=100,
    ))
    try:
        if stop == "caller_cancel":
            if phase == "body":
                await asyncio.wait_for(body.started.wait(), timeout=5)
            else:
                await asyncio.wait_for(safety_started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                await task
    finally:
        safety_release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    if phase == "preflight":
        assert not clients
    else:
        if phase == "body":
            assert body.closed
        assert response.is_closed
        assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("max_bytes", [3, 10])
async def test_download_keeps_body_limit_and_success_cleanup(monkeypatch, max_bytes):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"abc"
            yield b"def"

    response = httpx.Response(200, stream=Body())
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", lambda **_: client)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", AsyncMock(return_value=True))
    if max_bytes == 3:
        with pytest.raises(ValueError, match="too large"):
            await download_media_bytes_from_url("https://93.184.216.34/file", max_bytes=max_bytes)
    else:
        assert await download_media_bytes_from_url("https://93.184.216.34/file", max_bytes=max_bytes) == b"abcdef"
    assert response.is_closed
    assert client.is_closed


@pytest.mark.asyncio
async def test_public_image_cache_has_one_total_slow_drip_deadline(monkeypatch):
    body = DripBody()
    response = httpx.Response(200, stream=body)
    clients = []

    def make_client(**kwargs):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: response), **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", make_client)
    monkeypatch.setattr("tools.url_safety.async_is_safe_url", AsyncMock(return_value=True))

    with pytest.raises(TimeoutError):
        await cache_image_from_url(
            "https://93.184.216.34/image.png", retries=2, timeout=2.0,
        )

    assert body.closed
    assert response.is_closed
    assert clients[0].is_closed
