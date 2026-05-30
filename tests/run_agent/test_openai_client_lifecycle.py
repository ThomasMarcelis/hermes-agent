import sys
import threading
import time
import types
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


class FakeRequestClient:
    def __init__(self, responder):
        self._responder = responder
        self._client = SimpleNamespace(is_closed=False)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.responses = SimpleNamespace()
        self.close_calls = 0

    def _create(self, **kwargs):
        return self._responder(**kwargs)

    def close(self):
        self.close_calls += 1
        self._client.is_closed = True


class FakeSharedClient(FakeRequestClient):
    pass


class OpenAIFactory:
    def __init__(self, clients):
        self._clients = list(clients)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._clients:
            raise AssertionError("OpenAI factory exhausted")
        return self._clients.pop(0)


def _build_agent(shared_client=None):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.model = "gpt-5-codex"
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._client_lock = threading.RLock()
    agent._client_kwargs = {"api_key": "***", "base_url": agent.base_url}
    agent.client = shared_client or FakeSharedClient(lambda **kwargs: {"shared": True})
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.reasoning_callback = None
    agent.status_callback = None
    return agent


def _connection_error():
    return APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )


def test_retry_after_api_connection_error_recreates_request_client(monkeypatch):
    first_request = FakeRequestClient(lambda **kwargs: (_ for _ in ()).throw(_connection_error()))
    second_request = FakeRequestClient(lambda **kwargs: {"ok": True})
    factory = OpenAIFactory([first_request, second_request])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()

    with pytest.raises(APIConnectionError):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    result = agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert result == {"ok": True}
    assert len(factory.calls) == 2
    assert first_request.close_calls >= 1
    assert second_request.close_calls >= 1


def test_stale_non_stream_close_is_single_owner(monkeypatch):
    def slow_responder(**kwargs):
        time.sleep(0.1)
        raise _connection_error()

    request_client = FakeRequestClient(slow_responder)
    factory = OpenAIFactory([request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()
    agent._compute_non_stream_stale_timeout = lambda api_payload: 0.01

    with pytest.raises(APIConnectionError):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert request_client.close_calls == 1


def test_closed_shared_client_is_recreated_before_request(monkeypatch):
    stale_shared = FakeSharedClient(lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale shared client used")))
    stale_shared._client.is_closed = True

    replacement_shared = FakeSharedClient(lambda **kwargs: {"replacement": True})
    request_client = FakeRequestClient(lambda **kwargs: {"ok": "fresh-request-client"})
    factory = OpenAIFactory([replacement_shared, request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent(shared_client=stale_shared)
    result = agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert result == {"ok": "fresh-request-client"}
    assert agent.client is replacement_shared
    assert stale_shared.close_calls >= 1
    assert replacement_shared.close_calls == 0
    assert len(factory.calls) == 2


def test_concurrent_requests_do_not_break_each_other_when_one_client_closes(monkeypatch):
    first_started = threading.Event()
    first_closed = threading.Event()

    def first_responder(**kwargs):
        first_started.set()
        first_client.close()
        first_closed.set()
        raise _connection_error()

    def second_responder(**kwargs):
        assert first_started.wait(timeout=2)
        assert first_closed.wait(timeout=2)
        return {"ok": "second"}

    first_client = FakeRequestClient(first_responder)
    second_client = FakeRequestClient(second_responder)
    factory = OpenAIFactory([first_client, second_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent()
    results = {}

    def run_call(name):
        try:
            results[name] = agent._interruptible_api_call({"model": agent.model, "messages": []})
        except Exception as exc:  # noqa: BLE001 - asserting exact type below
            results[name] = exc

    thread_one = threading.Thread(target=run_call, args=("first",), daemon=True)
    thread_two = threading.Thread(target=run_call, args=("second",), daemon=True)
    thread_one.start()
    thread_two.start()
    thread_one.join(timeout=5)
    thread_two.join(timeout=5)

    values = list(results.values())
    assert sum(isinstance(value, APIConnectionError) for value in values) == 1
    assert values.count({"ok": "second"}) == 1
    assert len(factory.calls) == 2



def test_streaming_call_recreates_closed_shared_client_before_request(monkeypatch):
    chunks = iter([
        SimpleNamespace(
            model="gpt-5-codex",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello", tool_calls=None), finish_reason=None)],
        ),
        SimpleNamespace(
            model="gpt-5-codex",
            choices=[SimpleNamespace(delta=SimpleNamespace(content=" world", tool_calls=None), finish_reason="stop")],
        ),
    ])

    stale_shared = FakeSharedClient(lambda **kwargs: (_ for _ in ()).throw(AssertionError("stale shared client used")))
    stale_shared._client.is_closed = True

    replacement_shared = FakeSharedClient(lambda **kwargs: {"replacement": True})
    request_client = FakeRequestClient(lambda **kwargs: chunks)
    factory = OpenAIFactory([replacement_shared, request_client])
    monkeypatch.setattr(run_agent, "OpenAI", factory)

    agent = _build_agent(shared_client=stale_shared)
    agent.stream_delta_callback = lambda _delta: None
    # Force chat_completions mode so the streaming path uses
    # chat.completions.create(stream=True) instead of Codex responses.stream()
    agent.api_mode = "chat_completions"
    response = agent._interruptible_streaming_api_call({"model": agent.model, "messages": []})

    assert response.choices[0].message.content == "Hello world"
    assert agent.client is replacement_shared
    assert stale_shared.close_calls >= 1
    assert request_client.close_calls >= 1
    assert len(factory.calls) == 2


def test_codex_stream_progress_prevents_stale_kill_while_events_arrive(monkeypatch):
    """Codex Responses is streaming internally; stale detection tracks events."""
    agent = _build_agent()
    agent.api_mode = "codex_responses"
    agent.status_callback = None
    agent._print_fn = None
    agent.tool_progress_callback = None
    agent._executing_tools = False
    agent.suppress_status_output = True
    agent._has_stream_consumers = lambda: False
    agent._compute_non_stream_stale_timeout = lambda _messages=None: 0.25
    close_reasons = []

    def _close_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _abort_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _run_codex_stream(api_kwargs, client=None, on_first_delta=None, on_stream_event=None):
        time.sleep(0.20)
        if on_stream_event is not None:
            on_stream_event(SimpleNamespace(type="response.created"))
        time.sleep(0.22)
        return {"ok": "codex-result"}

    agent._close_request_openai_client = _close_request_openai_client
    agent._abort_request_openai_client = _abort_request_openai_client
    agent._run_codex_stream = _run_codex_stream
    monkeypatch.setattr(run_agent, "OpenAI", OpenAIFactory([FakeRequestClient(lambda **kwargs: None)]))

    result = agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert result == {"ok": "codex-result"}
    assert "stale_call_kill" not in close_reasons
    assert close_reasons == ["request_complete"]


def test_codex_stall_status_names_stream_first_event_not_non_streaming(monkeypatch):
    """A Codex first-event stall should be diagnosed as a stream stall."""
    agent = _build_agent()
    agent.api_mode = "codex_responses"
    agent.status_callback = None
    agent._print_fn = None
    agent.tool_progress_callback = None
    agent._executing_tools = False
    agent.suppress_status_output = True
    agent._has_stream_consumers = lambda: False
    agent._compute_non_stream_stale_timeout = lambda _messages=None: 0.05
    activities = []
    close_reasons = []
    close_event = threading.Event()

    def _close_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _abort_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()
        close_event.set()

    def _run_codex_stream(api_kwargs, client=None, on_first_delta=None, on_stream_event=None):
        close_event.wait(timeout=5)
        raise TimeoutError("provider socket closed before first Codex event")

    agent._touch_activity = activities.append
    agent._close_request_openai_client = _close_request_openai_client
    agent._abort_request_openai_client = _abort_request_openai_client
    agent._run_codex_stream = _run_codex_stream
    monkeypatch.setattr(run_agent, "OpenAI", OpenAIFactory([FakeRequestClient(lambda **kwargs: None)]))

    with pytest.raises(TimeoutError, match="before first Codex event"):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    joined = "\n".join(activities)
    assert "Codex stream first event" in joined
    assert "non-streaming" not in joined
    assert "stale_call_kill" in close_reasons


def test_codex_first_event_guard_is_separate_from_total_stale_timeout(monkeypatch):
    """High total stale windows must not disable no-first-event recovery."""
    monkeypatch.setenv("HERMES_CODEX_STREAM_FIRST_EVENT_TIMEOUT", "0.05")
    agent = _build_agent()
    agent.api_mode = "codex_responses"
    agent.status_callback = None
    agent._print_fn = None
    agent.tool_progress_callback = None
    agent._executing_tools = False
    agent.suppress_status_output = True
    agent._has_stream_consumers = lambda: False
    agent._compute_non_stream_stale_timeout = lambda _messages=None: 21600.0
    close_reasons = []
    close_event = threading.Event()

    def _close_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()

    def _abort_request_openai_client(client, *, reason):
        close_reasons.append(reason)
        client.close()
        close_event.set()

    def _run_codex_stream(api_kwargs, client=None, on_first_delta=None, on_stream_event=None):
        if close_event.wait(timeout=0.50):
            raise TimeoutError("provider socket closed before first Codex event")
        return {"unexpected": "hung until provider eventually replied"}

    agent._close_request_openai_client = _close_request_openai_client
    agent._abort_request_openai_client = _abort_request_openai_client
    agent._run_codex_stream = _run_codex_stream
    monkeypatch.setattr(run_agent, "OpenAI", OpenAIFactory([FakeRequestClient(lambda **kwargs: None)]))

    with pytest.raises(TimeoutError, match="before first Codex event"):
        agent._interruptible_api_call({"model": agent.model, "messages": []})

    assert "codex_ttfb_kill" in close_reasons
