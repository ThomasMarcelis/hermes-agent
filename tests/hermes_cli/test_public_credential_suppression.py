"""The public suppression lookup reads the bound profile on every call."""

import json

from hermes_cli.auth import is_source_suppressed
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_suppression_is_profile_scoped_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    launch = tmp_path / "launch"
    served = tmp_path / "served"
    launch.mkdir()
    served.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    store = launch / "auth.json"
    store.write_text(json.dumps({"providers": {}, "suppressed_sources": {"example": ["env:EXAMPLE_API_KEY"]}}))
    before = store.read_bytes()
    assert is_source_suppressed("example", "env:EXAMPLE_API_KEY")
    token = set_hermes_home_override(served)
    try:
        assert not is_source_suppressed("example", "env:EXAMPLE_API_KEY")
        assert not (served / "auth.json").exists()
        (served / "auth.json").write_text(json.dumps({"providers": {}, "suppressed_sources": {"example": ["manual"]}}))
        assert is_source_suppressed("example", "manual")
        assert not is_source_suppressed("example", "env:EXAMPLE_API_KEY")
    finally:
        reset_hermes_home_override(token)
    assert is_source_suppressed("example", "env:EXAMPLE_API_KEY")
    assert not is_source_suppressed("example", "manual")
    assert store.read_bytes() == before
