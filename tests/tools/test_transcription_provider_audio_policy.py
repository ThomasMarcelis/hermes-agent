"""Native provider upload/preprocessing opt-ins preserve the built-in pipeline order."""
from __future__ import annotations

import json
import struct
import subprocess
import wave
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli.plugins import _reset_plugin_managers_for_tests
from tools.transcription_tools import transcribe_audio

_PLUGIN = '''
import json
from pathlib import Path
from agent.transcription_provider import TranscriptionProvider
from hermes_cli.config import load_config

def config():
    return load_config()["stt"]["test-asr"]

def record(event, path):
    with open(config()["trace"], "a") as stream:
        stream.write(json.dumps({"event": event, "path": path,
                                "bytes": Path(path).stat().st_size}) + "\\n")

class Provider(TranscriptionProvider):
    @property
    def name(self): return "test-asr"
    def transcribe(self, file_path, **kw):
        record("upload", file_path)
        return {"success": True, "transcript": "synthetic segment", "provider": self.name}
    CAPABILITIES

def hints(file_path, **kw):
    record("hook", file_path)
    return {"language": "nld"}

def register(ctx):
    ctx.register_transcription_provider(Provider())
    ctx.register_hook("pre_transcription", hints)
'''
_OPT_IN = '''
    @property
    def max_upload_bytes(self): return config()["cap"]
    @property
    def cloud_preprocessing(self): return config()["preprocessing"]
'''


def _tone(path: Path, seconds: int = 20) -> None:
    frames = b"".join(struct.pack("<hh", 6000, -6000) * 8000 if n % 2 == 0
                      else b"\0\0" * 16000 for n in range(seconds))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(frames)


@pytest.fixture
def audio_provider(tmp_path, monkeypatch):
    home = tmp_path / "home"
    profile = tmp_path / "profile"
    home.mkdir()
    plugin = profile / "plugins" / "test-asr"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    _reset_plugin_managers_for_tests()
    audio = tmp_path / "synthetic.wav"
    _tone(audio)
    trace = tmp_path / "trace.jsonl"
    (plugin / "plugin.yaml").write_text(
        "name: test-asr\nversion: 1.0.0\ndescription: Synthetic ASR\nprovides_hooks: [pre_transcription]\n")

    def run(*, opt_in=True, trim=True, cap=1_000_000, path=audio, command=False, preprocessing=True, provider="test-asr"):
        (plugin / "__init__.py").write_text(_PLUGIN.replace("CAPABILITIES", _OPT_IN if opt_in else ""))
        cfg = {"plugins": {"enabled": ["test-asr"]}, "stt": {
            "provider": provider, "cloud_trim_silence": trim,
            "test-asr": {"cap": cap, "trace": str(trace), "preprocessing": preprocessing}}}
        if command:
            cfg["stt"]["providers"] = {"test-asr": {"type": "command", "command": "missing-synthetic-command"}}
        (profile / "config.yaml").write_text(yaml.safe_dump(cfg))
        result = transcribe_audio(str(path))
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, events

    yield run, audio
    _reset_plugin_managers_for_tests()


@pytest.mark.parametrize("trim", [True, False])
def test_caf_conversion_then_opted_in_trim_before_hook_and_upload(audio_provider, tmp_path, trim):
    run, audio = audio_provider
    caf = tmp_path / "synthetic.caf"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio), str(caf)], check=True)
    result, events = run(trim=trim, path=caf)
    assert result["success"]
    assert [e["event"] for e in events] == ["hook", "upload"]
    assert events[0]["path"] == events[1]["path"]
    prepared = Path(events[0]["path"])
    assert prepared.suffix == (".m4a" if trim else ".wav")
    assert ("-trimmed" in prepared.name) == trim
    assert not prepared.exists()  # Both CAF conversion and trim have request-owned cleanup.
    assert caf.exists() and audio.exists()


def test_ordinary_plugin_keeps_original_audio_even_with_cloud_trim_enabled(audio_provider):
    run, audio = audio_provider
    result, events = run(opt_in=False, trim=True, cap=1)
    assert result["success"]
    assert [e["event"] for e in events] == ["hook", "upload"]
    assert all(e["path"] == str(audio) for e in events)
    assert events[1]["bytes"] == audio.stat().st_size
    assert audio.exists()


@pytest.mark.parametrize("over", [False, True])
def test_opted_in_cap_boundary_reencodes_only_when_exceeded(audio_provider, over):
    run, audio = audio_provider
    cap = audio.stat().st_size - int(over)
    result, events = run(trim=False, cap=cap)
    assert result["success"]
    assert events[0]["event"] == "hook" and events[0]["path"] == str(audio)
    upload = Path(events[1]["path"])
    assert upload.suffix == (".m4a" if over else ".wav")
    assert events[1]["bytes"] <= cap
    assert upload.exists() == (not over)


def test_opted_in_trim_precedes_chunking_and_hook_runs_once(audio_provider):
    run, audio = audio_provider
    result, events = run(trim=True, cap=25_000)
    assert result["success"], result
    assert result["segments"] >= 2
    assert events[0]["event"] == "hook" and "-trimmed" in events[0]["path"]
    assert [e["event"] for e in events[1:]] == ["upload"] * result["segments"]
    assert all(e["bytes"] <= 25_000 for e in events[1:])
    assert all(not Path(e["path"]).exists() for e in events)
    assert result["transcript"] == " ".join(["synthetic segment"] * result["segments"])
    assert audio.exists()


def test_command_precedence_does_not_inherit_plugin_capabilities(audio_provider):
    run, audio = audio_provider
    result, events = run(trim=True, cap=1, command=True)
    assert not result["success"]
    assert "File too large" not in result["error"]
    assert all(e["path"] == str(audio) for e in events)
    assert not any(e["event"] == "upload" for e in events)


@pytest.mark.parametrize("cap", [0, -1, True, "large"])
def test_invalid_plugin_cap_fails_closed_before_preprocessing(audio_provider, cap):
    run, _audio = audio_provider
    result, events = run(trim=True, cap=cap)
    assert not result["success"]
    assert "max_upload_bytes" in result["error"]
    assert events == []


@pytest.mark.parametrize("preprocessing", ["true", 1, None])
def test_invalid_preprocessing_capability_refuses_upload(audio_provider, preprocessing):
    run, _audio = audio_provider
    result, events = run(preprocessing=preprocessing)
    assert not result["success"]
    assert "cloud_preprocessing" in result["error"]
    assert events == []


def test_policy_and_all_uploads_use_one_provider_instance(audio_provider, monkeypatch):
    from agent.transcription_provider import TranscriptionProvider
    from agent.transcription_registry import register_provider
    from tools import transcription_tools as stt

    class Replacement(TranscriptionProvider):
        @property
        def name(self):
            return 'test-asr'
        def transcribe(self, file_path, **kwargs):
            return {'success': True, 'transcript': 'replacement must wait for next recording'}

    original = stt._resolve_dispatch_hints
    def reload_at_hook(*args):
        hints = original(*args)
        from hermes_constants import get_hermes_home
        register_provider(Replacement(), scope=str(get_hermes_home()))
        return hints
    monkeypatch.setattr(stt, '_resolve_dispatch_hints', reload_at_hook)
    run, _audio = audio_provider
    result, events = run(trim=True, cap=25_000)
    assert result['success'] and result['segments'] >= 2
    assert result['transcript'] == ' '.join(['synthetic segment'] * result['segments'])
    assert len([e for e in events if e['event'] == 'upload']) == result['segments']


def test_failed_policy_discovery_never_recovers_into_unchecked_upload(audio_provider, monkeypatch):
    from hermes_cli import plugins
    actual = plugins._ensure_plugins_discovered
    attempted = []
    def fail_once(*args, **kwargs):
        attempted.append(kwargs.get('force', False))
        if len(attempted) == 1:
            raise RuntimeError('synthetic discovery failure')
        return actual(*args, **kwargs)
    monkeypatch.setattr(plugins, '_ensure_plugins_discovered', fail_once)
    run, _audio = audio_provider
    result, events = run()
    assert not result['success']
    assert not any(e['event'] == 'upload' for e in events)


def test_unregistered_name_forces_only_one_discovery_refresh(audio_provider, monkeypatch):
    from hermes_cli import plugins
    actual = plugins._ensure_plugins_discovered
    refreshes = []
    def discover_without_provider(*args, **kwargs):
        actual(*args, **kwargs)
        if kwargs.get('force'):
            refreshes.append(True)
    monkeypatch.setattr(plugins, '_ensure_plugins_discovered', discover_without_provider)
    run, _audio = audio_provider
    result, events = run(provider="unknown-asr")
    assert result['error_type'] == 'provider_not_registered'
    assert len(refreshes) == 1
    assert not any(e['event'] == 'upload' for e in events)
