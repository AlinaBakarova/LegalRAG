from __future__ import annotations

from pathlib import Path
import zipfile

from package_submission import ROOT_FILES, package


def test_allowlist_excludes_corpus_models_and_secrets(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ROOT_FILES | {"build_index.py", "predict.py"}:
        if name in {"own_questions.jsonl", "own_gold.jsonl"}:
            (root / name).write_text('{"id":"fixture"}\n')
        elif name not in {"dev_answers.jsonl", "dev_trace.jsonl"}:
            (root / name).write_text("test fixture\n")
    for relative in ["data/corpus.jsonl", "models/model.safetensors", ".env", "index/dense.faiss"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("MUST_NOT_BE_PUBLISHED")
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    first, second = package(root, one), package(root, two)
    assert first["archive_sha256"] == second["archive_sha256"]
    with zipfile.ZipFile(one) as archive:
        assert not any("MUST_NOT_BE_PUBLISHED" in archive.read(n).decode() for n in archive.namelist())
