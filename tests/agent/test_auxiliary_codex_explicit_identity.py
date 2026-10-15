"""An explicit native runtime credential must not trigger another pool selection."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('raw', [False, True])
def test_explicit_native_codex_key_and_endpoint_are_used_together(monkeypatch, raw):
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux, '_select_pool_entry', lambda provider: pytest.fail('explicit credential reselected'))
    monkeypatch.setattr(aux, '_create_openai_client', lambda **kwargs: SimpleNamespace(**kwargs))
    client, model = aux.resolve_provider_client(
        'openai-codex', 'gpt-6-astra', raw_codex=raw,
        explicit_api_key='fixture-runtime-key', explicit_base_url='https://runtime.invalid/backend-api/codex')
    assert client.api_key == 'fixture-runtime-key'
    assert str(client.base_url).rstrip('/') == 'https://runtime.invalid/backend-api/codex'
    assert model == 'gpt-6-astra'


@pytest.mark.parametrize('raw', [False, True])
def test_destination_only_override_cannot_retarget_selected_codex_key(monkeypatch, raw):
    from agent import auxiliary_client as aux
    from agent.auxiliary_pool import SelectedPoolContext
    selected = SelectedPoolContext(provider='openai-codex', pool_present=True,
        entry_id='fixture-row', api_key='fixture-pooled', base_url='https://selected.invalid/backend-api/codex')
    monkeypatch.setattr(aux, '_create_openai_client', lambda **kwargs: SimpleNamespace(**kwargs))
    client, _ = aux.resolve_provider_client('openai-codex', 'gpt-6-astra', raw_codex=raw,
        explicit_base_url='https://unrelated.invalid/v1', selected_pool=selected)
    assert client.api_key == 'fixture-pooled'
    assert str(client.base_url).rstrip('/') == 'https://selected.invalid/backend-api/codex'


def test_keyless_override_cannot_change_sdk_wire_destination(monkeypatch):
    import httpx
    from openai import OpenAI
    from agent import auxiliary_client as aux
    from agent.auxiliary_pool import SelectedPoolContext
    seen = []

    def receive(request):
        seen.append((str(request.url), request.headers['authorization']))
        return httpx.Response(200, json={'id': 'fixture', 'object': 'response', 'output': []})

    def client(**kwargs):
        return OpenAI(api_key=kwargs['api_key'], base_url=kwargs['base_url'],
                      http_client=httpx.Client(transport=httpx.MockTransport(receive)))

    monkeypatch.setattr(aux, '_create_openai_client', client)
    selection = SelectedPoolContext(provider='openai-codex', pool_present=True,
        entry_id='fixture-row', api_key='fixture-key', base_url='https://selected.invalid/backend-api/codex')
    sdk, _ = aux.resolve_provider_client('openai-codex', 'gpt-6-astra', raw_codex=True,
        explicit_base_url='https://unrelated.invalid/v1', selected_pool=selection)
    try:
        sdk.responses.create(model='gpt-6-astra', input='fixture')
        assert seen == [('https://selected.invalid/backend-api/codex/responses', 'Bearer fixture-key')]
    finally:
        sdk.close()
