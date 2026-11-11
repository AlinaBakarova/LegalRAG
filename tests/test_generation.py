from __future__ import annotations

import json

import httpx
import pytest

from rag.generation import Budget, BudgetError, LLMClient, answer_question, messages_for, pack_context
from rag.retrieval import BM25, Retriever, lexical_tokens, make_chunks

from conftest import CharacterCounter


def test_budget_reserves_fixed_output_and_all_thinking(settings):
    config, runtime, _ = settings
    budget = Budget(runtime, config.token_margin)
    with pytest.raises(BudgetError):
        budget.begin(8192)
    assert budget.calls == 0
    for _ in range(4):
        budget.begin(100)
        budget.finish({"prompt_tokens": 100, "completion_tokens": 1024, "completion_tokens_details": {"reasoning_tokens": 900}})
    assert budget.output_tokens == 4096
    with pytest.raises(BudgetError):
        budget.begin(100)
    assert budget.calls == 4


def test_cumulative_input_budget(settings):
    config, runtime, _ = settings
    budget = Budget(runtime, config.token_margin)
    for _ in range(2):
        budget.begin(6500)
        budget.finish({"prompt_tokens": 6500, "completion_tokens": 50})
    assert not budget.can_fit(4000)


def test_http_headers_payload_no_retry_and_usage(settings, question):
    config, runtime, _ = settings
    counter = CharacterCounter()
    messages = messages_for(question, [])
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"model": "serving-alias", "usage": {
            "prompt_tokens": counter.count(messages), "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 7}}, "choices": [{
                "message": {"content": "{}"}, "finish_reason": "stop"}]})

    client = LLMClient(runtime, "https://allowed.example/v1", "assigned-alias", "test-secret", "test-submission",
                       httpx.Client(transport=httpx.MockTransport(handler)))
    budget = Budget(runtime, config.token_margin)
    assert client.complete(question.id, messages, counter, budget) == "{}"
    request = captured[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://allowed.example/v1/chat/completions"
    assert request.headers["X-Question-ID"] == question.id
    assert request.headers["X-Submission-ID"] == "test-submission"
    for k, v in runtime.llm.generation.items():
        assert payload[k] == v
    assert payload["chat_template_kwargs"] == runtime.llm.chat_template_kwargs
    assert budget.calls == 1 and budget.output_tokens == 20
    assert budget.requests[0]["reasoning_tokens"] == 7


def test_missing_usage_is_never_zero_or_estimated(settings):
    config, runtime, _ = settings
    budget = Budget(runtime, config.token_margin)
    budget.begin(100)
    with pytest.raises(ValueError, match="unverifiable"):
        budget.finish(None)
    assert not budget.usage_verified
    assert not budget.can_fit(100)


def test_context_token_packing(settings, blocks, question):
    config, runtime, _ = settings
    chunks = make_chunks(blocks, 170, 30)
    retriever = Retriever(chunks, BM25([lexical_tokens(c.text) for c in chunks]), config)
    counter, budget = CharacterCounter(), Budget(runtime, config.token_margin)
    fragments, selected = pack_context(question, retriever, list(range(len(chunks))), True, counter, budget, 8)
    assert fragments and len(fragments) <= 8
    assert budget.can_fit(counter.count(messages_for(question, fragments)))
    assert {f["block_id"] for f in fragments} <= {b.block_id for b in blocks}


def test_repair_unknown_citation_without_changing_runtime(settings, blocks, question):
    config, runtime, _ = settings
    counter = CharacterCounter()
    chunks = make_chunks(blocks, 170, 30)
    retriever = Retriever(chunks, BM25([lexical_tokens(c.text) for c in chunks]), config)
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        response = {"answer_ids": ["choice-z"], "abstain": False, "rationale": "Ограниченная целая функция постоянна.",
                    "citations": ["invented"] if len(captured) == 1 else ["original-α"]}
        return httpx.Response(200, json={"usage": {"prompt_tokens": counter.count(body["messages"]), "completion_tokens": 70},
                                       "choices": [{"message": {"content": json.dumps(response)}, "finish_reason": "stop"}]})

    client = LLMClient(runtime, "https://allowed.example/v1", "assigned", "test-key", "test-submission",
                       httpx.Client(transport=httpx.MockTransport(handler)))
    answer, trace = answer_question(question, "no_dense", retriever, counter, client, config, runtime)
    assert answer["citations"] == ["original-α"]
    assert trace["valid"] and trace["usage_verified"] and trace["llm_calls"] == 2
    assert trace["output_tokens"] == 140
    assert any(d["action"] == "repair_json" for d in trace["decisions"])
    assert all(body["max_tokens"] == runtime.llm.generation["max_tokens"] for body in captured)


def test_simple_never_repairs_and_transport_error_not_abstention_success(settings, blocks, question):
    config, runtime, _ = settings
    chunks = make_chunks(blocks, 170, 30)
    retriever = Retriever(chunks, BM25([lexical_tokens(c.text) for c in chunks]), config)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503)

    client = LLMClient(runtime, "https://allowed.example/v1", "assigned", "test-key", "test-submission",
                       httpx.Client(transport=httpx.MockTransport(handler)))
    answer, trace = answer_question(question, "simple", retriever, CharacterCounter(), client, config, runtime)
    assert len(requests) == 1
    assert answer["abstain"] and not trace["valid"] and not trace["usage_verified"]
    assert trace["input_tokens"] is None and trace["output_tokens"] is None

