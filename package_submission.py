"""Allow-list packaging: generated corpus, models, caches and secrets cannot enter."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from rag.io import sha256_file, write_json


ROOT_FILES = {
    "README.md", "report.md", "requirements.txt", "requirements.in", "config.json", "runtime.json",
    "models.lock.json", ".gitignore", ".env.example", "pytest.ini", "experiments.csv", "experiments.status.json",
    "own_questions.jsonl", "own_gold.jsonl", "dev_answers.jsonl", "dev_trace.jsonl", "local_validation.json", "extraction_validation.json",
}
FORBIDDEN_SUFFIXES = {".pdf", ".faiss", ".npy", ".npz", ".safetensors", ".pt", ".pth", ".bin", ".sqlite", ".db"}


def selected_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        accepted = (len(rel.parts) == 1 and (path.name in ROOT_FILES or path.suffix == ".py")) or (
            rel.parts[0] in {"rag", "tests", "scripts", "docs"} and path.suffix in {".py", ".md", ".json", ".sh"}
        )
        if not accepted or "__pycache__" in rel.parts:
            continue
        if path.is_symlink() or path.suffix in FORBIDDEN_SUFFIXES or path.stat().st_size > 2_000_000:
            raise ValueError(f"Forbidden publication artifact: {rel}")
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if {"block_id", "page", "text"} <= set(row):
                    raise ValueError(f"Corpus row in source package: {rel}")
        files.append(path)
    required = {"build_index.py", "predict.py", "requirements.txt", "config.json", "runtime.json", "README.md", "report.md",
                "own_questions.jsonl", "own_gold.jsonl", "experiments.csv"}
    if not required <= {p.name for p in files}:
        raise ValueError("Missing required source deliverables")
    return files


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    files = selected_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo("tfkp_rag/" + path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return {"archive_sha256": sha256_file(output), "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    write_json(args.manifest, package(args.root, args.output))


if __name__ == "__main__":
    main()

