"""A discovered declaration controls native and auxiliary Responses without changing auth."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('levels', [(), ('low', 'high', 'ultra')])
def test_discovered_endpoint_declaration_reaches_both_response_paths(tmp_path, monkeypatch, levels):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    plugin = tmp_path / 'hermes/plugins/model-providers/declared-route'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: declared-route\nkind: model-provider\nversion: 1.0.0\n')
    (plugin / '__init__.py').write_text(f'''
from providers import register_provider
from providers.base import ProviderProfile
class Declared(ProviderProfile):
    def matches_endpoint(self, base_url):
        return base_url.rstrip('/') == 'https://declared.invalid/backend-api/codex'
    def supported_reasoning_efforts(self, model):
        return {levels!r}
    def get_model_context_length(self, model):
        return 123456
register_provider(Declared(name='declared-route'))
''')
    from agent.auxiliary_client import _CodexCompletionsAdapter
    from agent.model_metadata import get_model_context_length
    from agent.transports.codex import ResponsesApiTransport
    from providers import get_provider_profile
    endpoint = 'https://declared.invalid/backend-api/codex'
    assert get_provider_profile('declared-route') is not None
    # The frozen pre-extension signature must continue to work.
    assert get_model_context_length('fixture-model', endpoint, api_key='fixture-key', provider='custom:fixture') == 123456
    primary = ResponsesApiTransport().build_kwargs(
        model='fixture-model', messages=[{'role': 'user', 'content': 'fixture'}], tools=[],
        provider='custom:fixture', base_url=endpoint, is_codex_backend=True, reasoning_config={'effort': 'ultra'})
    auxiliary, _, _ = _CodexCompletionsAdapter(SimpleNamespace(base_url=endpoint), 'fixture-model')._build_responses_kwargs({
        'messages': [{'role': 'user', 'content': 'fixture'}], 'extra_body': {'reasoning': {'effort': 'ultra'}}})
    if levels:
        assert primary['reasoning']['effort'] == auxiliary['reasoning']['effort'] == 'ultra'
    else:
        assert 'reasoning' not in primary and 'reasoning' not in auxiliary


def test_discovered_compression_declaration_gates_and_encodes_requests(tmp_path, monkeypatch):
    import httpx
    zstd = pytest.importorskip('compression.zstd')
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    plugin = tmp_path / 'hermes/plugins/model-providers/encoding-fixture'
    plugin.mkdir(parents=True)
    (plugin / '__init__.py').write_text('''
from providers import register_provider
from providers.base import ProviderProfile
class Encoding(ProviderProfile):
    def matches_endpoint(self, base_url):
        return base_url == 'https://encoding.invalid/v1'
    def request_compression(self, base_url):
        return 'zstd'
register_provider(Encoding(name='encoding-fixture'))
''')
    from agent.codex_headers import codex_request_compression_hooks
    hooks = codex_request_compression_hooks('https://encoding.invalid/v1')
    assert hooks and codex_request_compression_hooks('https://unrelated.invalid/v1') is None
    for method, path, headers, encoded in [('POST','/responses',{},True), ('GET','/responses',{},False),
            ('POST','/responses/compact',{},False), ('POST','/responses',{'Content-Encoding':'gzip'},False)]:
        request = httpx.Request(method, 'https://encoding.invalid/v1'+path, headers=headers, content=b'{"fixture":1}')
        hooks['request'][0](request)
        assert (zstd.decompress(request.content) if encoded else request.content) == b'{"fixture":1}'
    import sys
    monkeypatch.setitem(sys.modules, 'compression', None)
    assert codex_request_compression_hooks('https://encoding.invalid/v1') is None


def test_broken_declaration_reports_policy_fault_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    plugin = tmp_path / 'hermes/plugins/model-providers/broken-fixture'
    plugin.mkdir(parents=True)
    (plugin / '__init__.py').write_text('''
from providers import register_provider
from providers.base import ProviderProfile
class Broken(ProviderProfile):
    def matches_endpoint(self, base_url):
        return base_url == 'https://broken.invalid/v1'
    def get_model_context_length(self, model):
        raise ImportError('fixture changed helper')
    def supported_reasoning_efforts(self, model, *, base_url):
        raise ValueError('fixture declaration fault')
register_provider(Broken(name='broken-fixture'))
''')
    from providers.routes import ProviderDeclarationError
    from agent.model_metadata import get_model_context_length
    from agent.reasoning_effort import route_supported_efforts
    with pytest.raises(ProviderDeclarationError, match='get_model_context_length'):
        get_model_context_length('fixture', 'https://broken.invalid/v1')
    with pytest.raises(ProviderDeclarationError, match='supported_reasoning_efforts'):
        route_supported_efforts('custom:fixture', 'fixture', base_url='https://broken.invalid/v1')
