from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from rag.config import load_settings
from rag.io import digest_json, dump_line, read_jsonl
from rag.schemas import Question


def predict(index, questions, output, trace_path, config_path, profile="final"):
    from rag.generation import LLMClient, PROFILES, answer_question
    from rag.resources import configure_resources
    from rag.models import load_counter
    from rag.index import load_index

    config, runtime, root = load_settings(config_path)
    stop = configure_resources(runtime.resources)
    rows = read_jsonl(questions, Question)
    paths = [Path(output).resolve(), Path(trace_path).resolve()]
    if len(set(paths)) != 2 or any(p in {Path(questions).resolve(), Path(config_path).resolve()} for p in paths):
        raise ValueError("Output and trace must be distinct and must not overwrite inputs")
    if any(p.exists() for p in paths):
        raise ValueError("Output files already exist; use a fresh run directory")
    hybrid = PROFILES[profile]["retrieval"] == "hybrid"
    retriever, manifest = load_index(index, config, runtime, root, hybrid)
    counter = load_counter(config, runtime, root)
    client = LLMClient.from_env(runtime)
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    temporary = [Path(str(p) + ".tmp") for p in paths]
    failures = 0
    with temporary[0].open("w", encoding="utf-8") as answers, temporary[1].open("w", encoding="utf-8") as traces:
        try:
            for q in rows:
                started = time.perf_counter()
                answer, trace = answer_question(q, profile, retriever, counter, client, config, runtime)
                trace.update(elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                             config_sha256=digest_json(config.model_dump()), runtime_sha256=digest_json(runtime.model_dump()),
                             corpus_sha256=manifest["corpus_sha256"], model=client.model)
                failures += not trace["valid"]
                answers.write(dump_line(answer))
                traces.write(dump_line(trace))
                answers.flush()
                traces.flush()
        finally:
            client.close()
            stop.set()
    for source, destination in zip(temporary, paths):
        os.replace(source, destination)
    return failures


def main():
    from rag.generation import PROFILES

    parser = argparse.ArgumentParser(description="Offline retrieval; generation uses only the configured Chat Completions endpoint.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="final")
    args = parser.parse_args()
    try:
        failures = predict(args.index, args.questions, args.output, args.trace, args.config, args.profile)
        if failures:
            print(f"{failures} technical failures; these rows must score zero, including on abstention gold", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"predict failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
