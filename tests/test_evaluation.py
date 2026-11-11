from __future__ import annotations

import pytest

from evaluate import citation_f1, question_score, trace_check
from rag.schemas import Answer, Gold


def test_scoring_exact_sets_and_best_evidence():
    gold = Gold(id="g", category="comparison", answer_ids=["B", "C"], abstain=False,
                evidence_sets=[["b1"], ["b2", "b3"]], rationale="Обоснование")
    answer = Answer(id="g", answer_ids=["C", "B"], abstain=False, rationale="Вывод", citations=["b2", "b3"])
    assert question_score(answer, gold) == (1, 1.0)
    extra = answer.model_copy(update={"citations": ["b2", "b3", "b4"]})
    assert question_score(extra, gold)[1] == pytest.approx(0.96)
    wrong = answer.model_copy(update={"answer_ids": ["B"]})
    assert question_score(wrong, gold) == (0, 0.0)
    assert citation_f1(["b1"], ["b1", "b2"]) == pytest.approx(2 / 3)


def test_technical_failure_does_not_score_correct_refusal():
    gold = Gold(id="g", category="insufficient", answer_ids=[], abstain=True, evidence_sets=[],
                rationale="Недостаточно данных", missing_information="Ориентация")
    answer = Answer(id="g", answer_ids=[], abstain=True, citations=[], rationale="Техническая ошибка")
    assert question_score(answer, gold, valid=False) == (0, 0.0)
    assert question_score(answer, gold, valid=True) == (1, 1.0)


def test_trace_usage_and_per_request_limits(settings):
    _, runtime, _ = settings
    trace = {"usage_verified": True, "llm_calls": 1, "input_tokens": 100, "output_tokens": 200,
             "requests": [{"input_tokens": 100, "output_tokens": 200, "max_output_tokens": 1024}]}
    assert trace_check(trace, runtime)
    trace["requests"][0]["input_tokens"] = 8000
    trace["input_tokens"] = 8000
    assert not trace_check(trace, runtime)

