"""Discovered OAuth plugins must use the same lease for wire, cache and recovery."""
import pytest


@pytest.mark.parametrize('auth_type', ['oauth_external', 'oauth_device_code'])
def test_discovered_oauth_plugin_consumes_one_frozen_lease(tmp_path, monkeypatch, auth_type):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    home = tmp_path / 'hermes'
    plugin = home / 'plugins/model-providers/frozen-fixture'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: frozen-fixture\nkind: model-provider\nversion: 1.0.0\n')
    (plugin / '__init__.py').write_text(f'''
from types import SimpleNamespace
from providers import register_provider
from providers.base import ProviderProfile
class Fixture(ProviderProfile):
    def create_client(self, **kwargs):
        return SimpleNamespace(**kwargs)
register_provider(Fixture(name="frozen-fixture", auth_type={auth_type!r}, base_url="https://fixture.invalid/v1"))
''')
    (home / 'config.yaml').write_text('credential_pool_strategies:\n  frozen-fixture: least_used\n')
    from providers import get_provider_profile
    from hermes_cli.auth import write_credential_pool, read_credential_pool
    from agent import auxiliary_client as aux
    from agent.auxiliary_pool import client_pool_failure_identity
    assert get_provider_profile('frozen-fixture').auth_type == auth_type
    write_credential_pool('frozen-fixture', [
        dict(id=key, label=key, auth_type='oauth', source='manual', priority=1,
             access_token='fixture-' + key, base_url='https://' + key + '.invalid/v1', request_count=0)
        for key in ('a', 'b')
    ])
    monkeypatch.setattr(aux, '_client_cache', {})
    client, model = aux._get_cached_client('frozen-fixture', 'fixture-model')
    assert model == 'fixture-model'
    identity = client_pool_failure_identity(client)
    assert client.api_key == identity['failed_api_key'] == 'fixture-' + identity['failed_entry_id']
    assert client.base_url == 'https://' + identity['failed_entry_id'] + '.invalid/v1'
    assert sum(row.get('request_count', 0) for row in read_credential_pool('frozen-fixture')) == 1
    assert next(iter(aux._client_cache)).pool_hint == client._hermes_pool_cache_revision
    # Explicit runtime credentials must not acquire or blame another pooled row.
    explicit, _ = aux._get_cached_client('frozen-fixture', 'fixture-model',
                                        api_key='explicit-fixture', base_url='https://explicit.invalid/v1')
    assert explicit.api_key == 'explicit-fixture' and explicit.base_url == 'https://explicit.invalid/v1'
    assert client_pool_failure_identity(explicit)['failed_entry_id'] == ''
    assert sum(row.get('request_count', 0) for row in read_credential_pool('frozen-fixture')) == 1

    # A response failure must bench the row that supplied the wire token.
    from openai import RateLimitError
    import httpx
    from agent.credential_pool import STATUS_EXHAUSTED
    error = RateLimitError('fixture quota exhausted', response=httpx.Response(
        429, request=httpx.Request('POST', client.base_url)), body={'error': {'code': 'insufficient_quota'}})
    assert aux._recover_provider_pool('frozen-fixture', error, **identity)
    rows = {row['id']: row for row in read_credential_pool('frozen-fixture')}
    assert rows[identity['failed_entry_id']]['last_status'] == STATUS_EXHAUSTED
    other = next(row for key, row in rows.items() if key != identity['failed_entry_id'])
    assert other.get('last_status') != STATUS_EXHAUSTED
