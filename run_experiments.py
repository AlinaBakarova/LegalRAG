from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from evaluate import evaluate
from rag.config import load_settings
from rag.generation import PROFILES
from rag.io import sha256_file, write_json


COLUMNS = ["name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens", "mean_llm_calls", "mean_cost", "median_latency_ms"]


def main():
    parser = argparse.ArgumentParser()
    for name in ("index", "questions", "gold", "corpus", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--export-submission", help="Copy final answers/trace and measured experiments.csv into this solution root")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiments output must be empty")
    config = Path(args.config).resolve()
    _, runtime, _ = load_settings(config)
    script = Path(__file__).with_name("predict.py")
    questions, corpus, index, gold = map(lambda p: str(Path(p).resolve()), (args.questions, args.corpus, args.index, args.gold))
    output.mkdir(parents=True, exist_ok=True)
    fixed = {"config": sha256_file(config), "runtime": sha256_file(config.parent / load_settings(config)[0].runtime_path),
             "questions": sha256_file(questions), "gold": sha256_file(gold), "corpus": sha256_file(corpus)}
    results = {}
    for profile in PROFILES:
        answers, trace = output / f"{profile}.answers.jsonl", output / f"{profile}.trace.jsonl"
        command = [sys.executable, str(script), "--index", index, "--questions", questions,
                   "--output", str(answers), "--trace", str(trace), "--config", str(config), "--profile", profile]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            write_json(output / "status.json", {"status": "failed", "profile": profile, "returncode": completed.returncode, "fixed_input_hashes": fixed})
            raise SystemExit("Experiment failed; metrics were not recorded")
        reference = results.get("simple", {}).get("mean_cost")
        result = evaluate(questions, gold, answers, trace, corpus, config, profile == "no_context", reference)
        if not result["efficiency_eligible"]:
            raise SystemExit("Unverifiable usage; experiment metrics were not recorded")
        results[profile] = result
        write_json(output / f"{profile}.metrics.json", result)
    # All five runs must finish with verifiable usage before publishing the comparison.
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({"name": name, **{k: r[k] for k in COLUMNS[1:]}} for name, r in results.items())
    write_json(output / "status.json", {"status": "complete", "fixed_input_hashes": fixed,
               "runtime_is_official": runtime.official, "cost_reference": "measured local simple baseline, not official C_ref"})
    if args.export_submission:
        target = Path(args.export_submission)
        target.mkdir(parents=True, exist_ok=True)
        for source, name in [(output / "final.answers.jsonl", "dev_answers.jsonl"),
                             (output / "final.trace.jsonl", "dev_trace.jsonl"), (output / "experiments.csv", "experiments.csv")]:
            shutil.copyfile(source, target / name)


if __name__ == "__main__":
    main()

