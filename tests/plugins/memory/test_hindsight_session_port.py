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
