from __future__ import annotations

import argparse
import statistics

from rag.config import load_settings
from rag.io import read_jsonl, write_json
from rag.schemas import Answer, Block, CATEGORIES, ClosedBookAnswer, Gold, Question, validate_references


def citation_f1(actual, expected):
    actual, expected = set(actual), set(expected)
    return 2 * len(actual & expected) / (len(actual) + len(expected)) if actual or expected else 1.0


def question_score(answer, gold, valid=True):
    correct = valid and answer.abstain == gold.abstain and set(answer.answer_ids) == set(gold.answer_ids)
    if gold.abstain:
        return int(correct), float(correct)
    f1 = max((citation_f1(answer.citations, evidence) for evidence in gold.evidence_sets), default=0)
    return int(correct), float(correct) * (0.8 + 0.2 * f1)


def trace_check(trace, runtime):
    limits = runtime.limits
    calls, inp, out = (trace.get(k) for k in ("llm_calls", "input_tokens", "output_tokens"))
    if trace.get("usage_verified") is not True or any(type(v) is not int or v < 0 for v in (calls, inp, out)):
        return False
    if calls > limits.max_llm_calls or inp > limits.max_input_tokens or out > limits.max_output_tokens:
        return False
    requests = trace.get("requests", [])
    if len(requests) != calls:
        return False
    if any(type(r.get(k)) is not int or r[k] < 0 for r in requests for k in ("input_tokens", "output_tokens", "max_output_tokens")):
        return False
    if sum(r["input_tokens"] for r in requests) != inp or sum(r["output_tokens"] for r in requests) != out:
        return False
    return all(r["input_tokens"] + r["max_output_tokens"] <= limits.max_request_tokens
               and r["max_output_tokens"] == runtime.llm.generation["max_tokens"]
               and r["output_tokens"] <= r["max_output_tokens"] for r in requests)


def unique_rows(rows):
    mapped, duplicates = {}, set()
    for row in rows:
        qid = row.get("id")
        if not isinstance(qid, str):
            raise ValueError("Record without string id")
        if qid in mapped:
            duplicates.add(qid)
        mapped[qid] = row
    return mapped, duplicates


def evaluate(questions_path, gold_path, answers_path, trace_path, corpus_path, config_path, closed=False, reference_cost=None):
    _, runtime, _ = load_settings(config_path)
    questions = read_jsonl(questions_path, Question)
    golds = {g.id: g for g in read_jsonl(gold_path, Gold)}
    blocks = {b.block_id for b in read_jsonl(corpus_path, Block)}
    answers, duplicate_answers = unique_rows(read_jsonl(answers_path))
    traces, duplicate_traces = unique_rows(read_jsonl(trace_path))
    ids = {q.id for q in questions}
    if ids != set(golds):
        raise ValueError("Gold/question IDs differ")
    if set(answers) - ids or set(traces) - ids:
        raise ValueError("Extra answer/trace IDs")
    if reference_cost is not None and reference_cost <= 0:
        raise ValueError("Reference cost must be positive and measured on this exact dataset/runtime")
    by_category = {name: [] for name in CATEGORIES}
    accuracy, inputs, outputs, calls, costs, latencies, details = [], [], [], [], [], [], []
    efficiency_eligible = True
    for q in questions:
        gold, trace = golds[q.id], traces.get(q.id, {})
        if any(not set(e) <= blocks for e in gold.evidence_sets):
            raise ValueError("Gold contains nonexistent source blocks")
        if not set(gold.answer_ids) <= {o.id for o in q.options}:
            raise ValueError("Gold contains unknown options")
        budget_ok = q.id not in duplicate_traces and trace_check(trace, runtime)
        efficiency_eligible &= budget_ok
        valid = budget_ok and trace.get("valid") is True and q.id not in duplicate_answers
        answer = None
        try:
            schema = ClosedBookAnswer if closed else Answer
            answer = schema.model_validate(answers.get(q.id, {}))
            validate_references(answer, q, blocks)
            if not closed and not set(answer.citations) <= set(trace.get("context_block_ids", [])):
                valid = False
        except (ValueError, TypeError):
            valid = False
        exact, score = question_score(answer, gold, valid) if answer else (0, 0.0)
        accuracy.append(exact)
        by_category[gold.category].append(score)
        details.append({"id": q.id, "answer_correct": exact, "score": score, "valid": valid,
                        "budget_verified": budget_ok, "gold_category": gold.category})
        if budget_ok:
            inp, out, n_calls = (trace[k] for k in ("input_tokens", "output_tokens", "llm_calls"))
            inputs.append(inp)
            outputs.append(out)
            calls.append(n_calls)
            costs.append(inp + 4 * out + 256 * n_calls)
        latency = trace.get("elapsed_ms")
        if type(latency) in (int, float) and latency >= 0:
            latencies.append(latency)
    if not all(by_category.values()):
        raise ValueError("Q requires all four categories")
    macro = statistics.mean(statistics.mean(values) for values in by_category.values())
    result = {
        "answer_accuracy": statistics.mean(accuracy), "Q": None if closed else macro,
        "mean_input_tokens": statistics.mean(inputs) if efficiency_eligible else None,
        "mean_output_tokens": statistics.mean(outputs) if efficiency_eligible else None,
        "mean_llm_calls": statistics.mean(calls) if efficiency_eligible else None,
        "mean_cost": statistics.mean(costs) if efficiency_eligible else None,
        "median_latency_ms": statistics.median(latencies) if len(latencies) == len(questions) else None,
        "efficiency_eligible": efficiency_eligible,
        "E": None, "total_score": None,
        "reference_note": "User-created simple baseline; not the missing official C_ref" if reference_cost else "C_ref not provided",
        "category_scores": {k: statistics.mean(v) for k, v in by_category.items()}, "questions": details,
    }
    if reference_cost is not None and not closed:
        result["E"] = min(1.0, reference_cost / max(result["mean_cost"], 1)) if efficiency_eligible else 0.0
        result["total_score"] = 75 * macro + 25 * macro * result["E"]
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("questions", "gold", "answers", "trace", "corpus", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--closed-book", action="store_true")
    parser.add_argument("--reference-cost", type=float)
    args = parser.parse_args()
    result = evaluate(args.questions, args.gold, args.answers, args.trace, args.corpus, args.config, args.closed_book, args.reference_cost)
    write_json(args.output, result)


if __name__ == "__main__":
    main()

