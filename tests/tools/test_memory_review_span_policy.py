"""Upstream span patches preserve facts while honoring the owned review opt-in."""
import json

import pytest

from tools.memory_tool import MemoryStore, memory_tool
from tools.skill_provenance import reset_current_write_origin, set_current_write_origin


@pytest.mark.parametrize('batch', [False, True])
def test_opted_in_review_can_patch_a_span_without_replacing_a_bulk_entry(tmp_path, monkeypatch, batch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text(
        'memory:\n  background_review_allow_destructive: true\n  write_approval: false\n', encoding='utf-8')
    store = MemoryStore()
    original = 'Name and timezone.\n\nPrefers terse replies.\n\nCareer background and salary floor.'
    assert store.add('user', original)['success']
    op = {'action': 'patch', 'old_text': 'terse', 'content': 'concise'}
    token = set_current_write_origin('background_review')
    try:
        if batch:
            response = memory_tool(target='user', store=store, operations=[op])
        else:
            response = memory_tool(target='user', store=store, action='patch',
                                   old_text=op['old_text'], content=op['content'])
        result = json.loads(response)
    finally:
        reset_current_write_origin(token)
    assert result['success'] is True
    assert not result.get('staged')
    assert store._entries_for('user') == [original.replace('terse', 'concise')]


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('original', [
    'Name and timezone.\n\nCareer background and salary floor.',
    'x' * 1000,
])
def test_opted_in_review_still_stages_a_whole_bulk_entry_patch(tmp_path, monkeypatch, batch, original):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text(
        'memory:\n  background_review_allow_destructive: true\n  write_approval: false\n', encoding='utf-8')
    store = MemoryStore()
    assert store.add('user', original)['success']
    op = {'action': 'patch', 'old_text': original, 'content': 'Only one fact remains.'}
    token = set_current_write_origin('background_review')
    try:
        if batch:
            response = memory_tool(target='user', store=store, operations=[op])
        else:
            response = memory_tool(target='user', store=store, action='patch',
                                   old_text=op['old_text'], content=op['content'])
        result = json.loads(response)
    finally:
        reset_current_write_origin(token)
    assert result['staged'] is True
    assert store._entries_for('user') == [original]
