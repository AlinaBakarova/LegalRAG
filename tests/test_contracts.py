from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag.config import Runtime
from rag.io import json_loads, read_jsonl
from rag.retrieval import make_chunks
from rag.schemas import AnswerBody, Question, validate_references
from prepare_data import repair_font_encoding


def test_strict_answer_and_evidence(question):
    body = AnswerBody(answer_ids=["choice-z"], abstain=False, rationale="Основание.", citations=["original-α"])
    validate_references(body, question, {"original-α"})
    with pytest.raises(ValueError):
        validate_references(body, question, {"invented-chunk-id"})
    with pytest.raises(ValueError):
        validate_references(body.model_copy(update={"answer_ids": ["A"]}), question, {"original-α"})
    for update in [dict(answer_ids=[]), dict(citations=[]), dict(abstain="false"), dict(citations=["x"] * 2),
                   dict(answer_ids=["choice-z"] * 2), dict(rationale="x" * 1201), dict(extra="ignored")]:
        with pytest.raises(ValidationError):
            AnswerBody.model_validate({**body.model_dump(), **update})


def test_abstention_is_explicit():
    AnswerBody(answer_ids=[], abstain=True, rationale="Не задана ориентация.", citations=[])
    with pytest.raises(ValidationError):
        AnswerBody(answer_ids=["A"], abstain=True, rationale="Нет оснований", citations=[])


def test_duplicate_ids_and_json_keys(tmp_path, question):
    with pytest.raises(ValueError):
        json_loads('{"id":"one","id":"two"}')
    with pytest.raises(ValueError):
        json_loads('{"latency":NaN}')
    path = tmp_path / "questions.jsonl"
    line = json.dumps(question.model_dump(), ensure_ascii=False) + "\n"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="Duplicate ID"):
        read_jsonl(path, Question)


def test_chunk_spans_cover_source_without_mutation(blocks):
    chunks = make_chunks(blocks, 170, 40, lambda t: len(t) <= 120)
    for block in blocks:
        covered = set()
        for c in chunks:
            if c.block_id == block.block_id:
                assert c.text == block.text[c.start:c.end]
                assert c.end - c.start <= 120
                covered.update(range(c.start, c.end))
        assert covered == set(range(len(block.text)))


def test_repair_specific_font_preserves_math():
    assert repair_font_encoding("Ãîëîìîðôíàÿ ôóíêöèÿ, α × β ÷ γ ∂f/∂z") == "Голоморфная функция, α × β ÷ γ ∂f/∂z"


def test_original_source_whitespace_is_not_silently_modified():
    from rag.schemas import Block
    text = "  формула\n  ∂f/∂z̄ = 0\n"
    assert Block(block_id="source-id", page=1, section="§ 2", text=text).text == text
    from rag.retrieval import lexical_tokens
    assert "z̄" in lexical_tokens(text)


def test_runtime_disallows_other_generators_and_excess(settings):
    _, runtime, _ = settings
    data = runtime.model_dump()
    data["llm"]["repository"] = "some/other-generator"
    with pytest.raises(ValidationError):
        Runtime.model_validate(data)
    data = runtime.model_dump()
    data["limits"]["max_llm_calls"] = 5
    with pytest.raises(ValidationError):
        Runtime.model_validate(data)

