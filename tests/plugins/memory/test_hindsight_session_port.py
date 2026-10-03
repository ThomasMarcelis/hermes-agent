"""Real config/lifecycle integration for the owned Hindsight session port."""
from agent.memory_manager import MemoryManager
from hermes_constants import get_hermes_home
from plugins.memory.hindsight import HindsightMemoryProvider


def test_hindsight_local_config_and_manager_session_boundaries(monkeypatch):
    # Exercise the real profile config reader and manager/provider methods while
    # avoiding dependency installation; disabled recall/retention needs no SDK.
    monkeypatch.setattr('plugins.memory.hindsight._maybe_upgrade_client', lambda: None)
    provider = HindsightMemoryProvider()
    provider.save_config({
        'mode': 'local_external',
        'api_url': 'http://127.0.0.1:1',
        'auto_retain': False,
        'auto_recall': False,
        'expose_retain_tool': False,
        'bank_id': 'fallback',
        'bank_id_template': 'owned-{session}',
        'recall_tags': '',
    }, str(get_hermes_home()))
    assert (get_hermes_home() / 'hindsight' / 'config.json').is_file()
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id='child', parent_session_id='parent', platform='cli')
    assert provider._bank_id == 'owned-child'
    assert provider._parent_session_id == 'parent'
    assert all(schema['name'] != 'hindsight_retain' for schema in manager.get_all_tool_schemas())
    manager.queue_session_switch('root', parent_session_id='', reason='resume')
    assert manager.flush_pending(timeout=2)
    assert provider._session_id == 'root'
    assert provider._parent_session_id == ''
    assert provider._bank_id == 'owned-root'
    assert provider._client is None
    manager.shutdown_all()


def test_fork_provider_discovery_and_client_factory_are_profile_scoped(tmp_path, monkeypatch):
    """A→B→A discovers the durable fork and builds each SDK client with its own credentials."""
    import json
    import sys
    from pathlib import Path
    from types import ModuleType

    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from plugins.memory import find_provider_dir, load_memory_provider

    calls = []
    sdk = ModuleType("hindsight_client")
    sdk.Hindsight = lambda **kwargs: calls.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "hindsight_client", sdk)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr("plugins.memory.hindsight._maybe_upgrade_client", lambda: None)
    homes = [tmp_path / name for name in ("a", "b")]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("memory:\n  provider: hindsight\n")
        directory = home / "hindsight"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({
            "mode": "local_external", "api_url": f"http://{home.name}.invalid",
            "auto_retain": False, "auto_recall": False,
        }))
    for home in (homes[0], homes[1], homes[0]):
        with _profile_runtime_scope(home, {"HINDSIGHT_API_KEY": f"key-{home.name}"}):
            assert find_provider_dir("hindsight").resolve() == (
                Path(__file__).resolve().parents[3]
                / "plugins" / "memory" / "hindsight"
            )
            provider = load_memory_provider("hindsight", register_skills=False)
            assert isinstance(provider, HindsightMemoryProvider)
            provider.initialize(session_id=f"session-{home.name}", platform="cli")
            provider._new_cloud_client()
            provider.shutdown()
    assert [call["api_key"] for call in calls] == ["key-a", "key-b", "key-a"]
    assert [call["base_url"] for call in calls] == [
        "http://a.invalid", "http://b.invalid", "http://a.invalid",
    ]
