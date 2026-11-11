from __future__ import annotations

import json

import httpx

from build_index import build
from predict import predict
from rag.generation import LLMClient
from rag.io import read_jsonl, write_jsonl

from conftest import CharacterCounter


def test_prediction_one_line_per_input_and_no_state_carry(tmp_path, settings, blocks, question, monkeypatch):
    config, runtime, root = settings
    corpus, index = tmp_path / "corpus.jsonl", tmp_path / "index"
    write_jsonl(corpus, blocks)
    build(corpus, index, root / "config.json", lexical_only=True)
    other = question.model_copy(deep=True)
    other.id = "new-question-different-order"
    other.options.reverse()
    for i, option in enumerate(other.options):
        option.id = f"unseen-option-{i}"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [question, other])
    counter, requests = CharacterCounter(), []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        requests.append((request.headers["X-Question-ID"], payload))
        answer = {"answer_ids": [payload["options"][0]["id"]], "abstain": False, "rationale": "Тест формата с имитацией API.",
                  "citations": [payload["source_fragments"][0]["block_id"]]}
        return httpx.Response(200, json={"usage": {"prompt_tokens": counter.count(body["messages"]), "completion_tokens": 40},
                                       "choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]})

    client = LLMClient(runtime, "https://test.invalid/v1", "assigned", "fake-test-key", "test-submission",
                       httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("rag.models.load_counter", lambda *args: counter)
    monkeypatch.setattr(LLMClient, "from_env", lambda *args: client)
    answers, trace = tmp_path / "answers.jsonl", tmp_path / "trace.jsonl"
    assert predict(index, questions, answers, trace, root / "config.json", "simple") == 0
    output, traces = read_jsonl(answers), read_jsonl(trace)
    assert [a["id"] for a in output] == [question.id, other.id]
    assert output[1]["answer_ids"] == [other.options[0].id]
    assert [r[0] for r in requests] == [question.id, other.id]
    assert len(traces) == 2 and all(t["llm_calls"] == 1 and t["usage_verified"] for t in traces)
    assert question.id not in json.dumps(requests[1][1])

