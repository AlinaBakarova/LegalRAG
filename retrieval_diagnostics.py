"""Retrieval-only measurements; never report these as answer quality/Q."""
from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict

from rag.config import load_settings
from rag.index import load_index
from rag.io import read_jsonl, write_json
from rag.resources import configure_resources
from rag.schemas import Gold, Question


PROFILES = {
    "simple": ("bm25", False),
    "final": ("hybrid", True),
    "no_dense": ("bm25", True),
    "no_neighbors": ("hybrid", False),
}


def diagnose(retriever, questions, golds, *, mode, neighbors, context_chunks):
    if {q.id for q in questions} != set(golds):
        raise ValueError("Question/gold IDs differ")
    measurements = []
    for q in questions:
        gold = golds[q.id]
        ranked = retriever.search(q, mode)
        chosen = retriever.context_order(ranked, neighbors=neighbors)[:context_chunks]
        retrieved = list(dict.fromkeys(retriever.chunks[i].block_id for i in ranked))
        context_ids = list(dict.fromkeys(retriever.chunks[i].block_id for i in chosen))
        recall = None if gold.abstain else max(len(set(e) & set(context_ids)) / len(e) for e in gold.evidence_sets)
        measurements.append({"id": q.id, "question": q.question, "expected_evidence_sets": gold.evidence_sets,
                             "retrieved_block_ids": retrieved, "candidate_context_block_ids": context_ids,
                             "candidate_context_fragments": [asdict(retriever.chunks[i]) for i in chosen],
                             "page_evidence_recall": recall})
    values = [r["page_evidence_recall"] for r in measurements if r["page_evidence_recall"] is not None]
    return {"questions": measurements,
            "mean_page_evidence_recall": statistics.mean(values) if values else None,
            "solvable_questions": len(values),
            "full_page_evidence_questions": sum(value == 1 for value in values)}


def main():
    parser = argparse.ArgumentParser()
    for name in ("index", "questions", "gold", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="simple")
    args = parser.parse_args()
    config, runtime, root = load_settings(args.config)
    stop = configure_resources(runtime.resources)
    mode, neighbors = PROFILES[args.profile]
    retriever, manifest = load_index(args.index, config, runtime, root, hybrid=mode == "hybrid")
    golds = {g.id: g for g in read_jsonl(args.gold, Gold)}
    report = diagnose(retriever, read_jsonl(args.questions, Question), golds,
                      mode=mode, neighbors=neighbors, context_chunks=config.retrieval.context_chunks)
    write_json(args.output, {"measurement": "Retrieval-only, before generator token packing; page recall does not verify formula preservation",
               "profile": args.profile, "mode": mode, "neighbors": neighbors,
               "corpus_sha256": manifest["corpus_sha256"], "index_signature": manifest["index_signature"],
               **report})
    stop.set()


if __name__ == "__main__":
    main()
