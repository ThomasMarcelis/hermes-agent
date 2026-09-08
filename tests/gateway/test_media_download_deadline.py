"""Bound the entire streamed download, including cancellation cleanup."""

import asyncio
import socket
import threading

import httpx
import pytest

from gateway.platforms.base import download_media_bytes_from_url


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    # Keep production SSRF validation active without querying an external resolver.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
    ])


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
    dns_started, dns_release, dns_finished = (threading.Event() for _ in range(3))
    public_resolver = socket.getaddrinfo

    def resolve(host, *args, **kwargs):
        if phase == "preflight" or (phase == "redirect" and host == "redirect.example"):
            dns_started.set()
            try:
                dns_release.wait(timeout=5)
            finally:
                dns_finished.set()
        return public_resolver(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)

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
                assert await asyncio.to_thread(dns_started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                await task
    finally:
        dns_release.set()
        if dns_started.is_set():
            assert await asyncio.to_thread(dns_finished.wait, 5)
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
    if max_bytes == 3:
        with pytest.raises(ValueError, match="too large"):
            await download_media_bytes_from_url("https://93.184.216.34/file", max_bytes=max_bytes)
    else:
        assert await download_media_bytes_from_url("https://93.184.216.34/file", max_bytes=max_bytes) == b"abcdef"
    assert response.is_closed
    assert client.is_closed
