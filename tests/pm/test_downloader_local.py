"""Pinned local tool artifacts use the normal verified publication contract."""

from __future__ import annotations

import hashlib

import pytest

from pm.downloader import Download, DownloadError, DownloadPaused, HashError, Source


@pytest.mark.parametrize("authority", ["", "localhost"])
def test_local_artifact_is_verified_and_published(tmp_path, authority):
    payload = b"pinned runtime artifact"
    artifact = tmp_path / "runtime with spaces.tar.gz"
    artifact.write_bytes(payload)
    destination = tmp_path / "store" / "runtime.tar.gz"
    url = artifact.as_uri().replace("file://", f"file://{authority}", 1)
    source = Source(url, destination, hashlib.sha256(payload).hexdigest())
    progress = []

    result = Download([source], partials_dir=tmp_path / "partials").run(
        progress=lambda done, total, ranges: progress.append((done, total, ranges)))

    assert result == [destination]
    assert destination.read_bytes() == artifact.read_bytes() == payload
    assert progress[-1] == (len(payload), len(payload), {str(destination): [(0, len(payload))]})
    assert list(destination.parent.iterdir()) == [destination]


def test_local_hash_mismatch_preserves_source_and_previous_destination(tmp_path):
    artifact = tmp_path / "runtime.tar.gz"
    artifact.write_bytes(b"incorrect artifact")
    destination = tmp_path / "store" / "runtime.tar.gz"
    destination.parent.mkdir()
    destination.write_bytes(b"previous complete artifact")
    partials = tmp_path / "partials"
    source = Source(artifact.as_uri(), destination, hashlib.sha256(b"expected artifact").hexdigest())

    with pytest.raises(HashError):
        Download([source], partials_dir=partials).run()

    assert artifact.read_bytes() == b"incorrect artifact"
    assert destination.read_bytes() == b"previous complete artifact"
    assert list(destination.parent.iterdir()) == [destination]
    assert not list(partials.rglob("*.part"))


@pytest.mark.parametrize("digest", ["", "abc123", "z" * 64])
def test_local_artifact_requires_full_sha256_even_when_destination_exists(tmp_path, digest):
    artifact = tmp_path / "runtime.tar.gz"
    artifact.write_bytes(b"runtime")
    with pytest.raises(ValueError, match="SHA256"):
        Download([Source(artifact.as_uri(), artifact, digest)],
                 partials_dir=tmp_path / "partials").run()


@pytest.mark.parametrize("url", [
    "file://server/share/runtime.tar.gz",
    "file://localhost:123/runtime.tar.gz",
    "file://user@localhost/runtime.tar.gz",
    "file:relative/runtime.tar.gz",
    "file:////server/share/runtime.tar.gz",
    "file:///%2Fserver/share/runtime.tar.gz",
    "file:///%5Cserver/share/runtime.tar.gz",
    "file:///runtime.tar.gz?alternative",
    "file:///runtime.tar.gz#fragment",
])
def test_local_artifact_rejects_nonlocal_or_ambiguous_urls(tmp_path, url):
    destination = tmp_path / "runtime.tar.gz"
    with pytest.raises(ValueError, match="local file"):
        Download([Source(url, destination, "0" * 64)],
                 partials_dir=tmp_path / "partials").run()
    assert not destination.exists()


def test_local_artifact_can_already_be_its_destination(tmp_path):
    artifact = tmp_path / "runtime.tar.gz"
    payload = b"runtime"
    artifact.write_bytes(payload)
    source = Source(artifact.as_uri(), artifact, hashlib.sha256(payload).hexdigest())
    assert Download([source], partials_dir=tmp_path / "partials").run() == [artifact]
    assert artifact.read_bytes() == payload


def test_local_hash_mismatch_cannot_delete_its_own_source(tmp_path):
    artifact = tmp_path / "runtime.tar.gz"
    artifact.write_bytes(b"incorrect artifact")
    with pytest.raises(HashError):
        Download([Source(artifact.as_uri(), artifact, "0" * 64)],
                 partials_dir=tmp_path / "partials").run()
    assert artifact.read_bytes() == b"incorrect artifact"


def test_local_artifact_rejects_directory(tmp_path):
    destination = tmp_path / "runtime.tar.gz"
    with pytest.raises(DownloadError, match="regular file"):
        Download([Source(tmp_path.as_uri(), destination, "0" * 64)],
                 partials_dir=tmp_path / "partials").run()
    assert not destination.exists()


def test_pausing_local_copy_preserves_destination_and_cleans_staging(tmp_path):
    payload = b"runtime" * (1 << 20)
    artifact = tmp_path / "runtime.tar.gz"
    artifact.write_bytes(payload)
    destination = tmp_path / "store" / "runtime.tar.gz"
    destination.parent.mkdir()
    destination.write_bytes(b"previous runtime")
    partials = tmp_path / "partials"
    download = Download([Source(artifact.as_uri(), destination, hashlib.sha256(payload).hexdigest())],
                        partials_dir=partials)

    def pause_after_copy_starts(done, total, ranges):
        if done:
            download.pause()

    with pytest.raises(DownloadPaused):
        download.run(progress=pause_after_copy_starts)

    assert destination.read_bytes() == b"previous runtime"
    assert artifact.read_bytes() == payload
    assert list(destination.parent.iterdir()) == [destination]
    assert not list(partials.rglob("*.part"))


def test_failed_local_publication_keeps_previous_destination(tmp_path, monkeypatch):
    import shutil

    artifact = tmp_path / "runtime.tar.gz"
    payload = b"complete new artifact"
    artifact.write_bytes(payload)
    destination = tmp_path / "store" / "runtime.tar.gz"
    destination.parent.mkdir()
    destination.write_bytes(b"previous runtime")
    partials = tmp_path / "partials"

    def interrupted_copy(incoming, target, *args):
        target.write(incoming.read(4))
        raise OSError("destination full")

    monkeypatch.setattr(shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="destination full"):
        Download([Source(artifact.as_uri(), destination, hashlib.sha256(payload).hexdigest())],
                 partials_dir=partials).run()

    assert destination.read_bytes() == b"previous runtime"
    assert artifact.read_bytes() == payload
    assert list(destination.parent.iterdir()) == [destination]
    assert not list(partials.rglob("*.part"))


def test_https_cannot_redirect_to_local_artifact(tmp_path):
    import urllib.request

    from pm.downloader import _HttpsRedirectHandler

    request = urllib.request.Request("https://example.invalid/runtime.tar.gz")
    with pytest.raises(DownloadError, match="redirect to non-https"):
        _HttpsRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, (tmp_path / "runtime.tar.gz").as_uri())
