from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag.index import load_index
from rag.io import read_json, sha256_file, write_jsonl


def test_rebuild_from_other_directory_and_integrity(tmp_path, settings, blocks):
    config, runtime, root = settings
    corpus = tmp_path / "corpus with spaces.jsonl"
    write_jsonl(corpus, blocks)
    script = Path(__file__).resolve().parents[1] / "build_index.py"
    for index_name in ("one", "different index"):
        completed = subprocess.run([sys.executable, str(script), "--corpus", str(corpus), "--output", str(tmp_path / index_name),
                                    "--config", str(root / "config.json"), "--lexical-only"],
                                   cwd=tmp_path, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    for name in ("blocks.jsonl", "chunks.jsonl", "bm25.json"):
        assert sha256_file(tmp_path / "one" / name) == sha256_file(tmp_path / "different index" / name)
    manifest = read_json(tmp_path / "one" / "manifest.json")
    assert not manifest["dense"] and manifest["resources"]["python"].startswith("3.11")
    load_index(tmp_path / "one", config, runtime, root, False)
    with (tmp_path / "one" / "chunks.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        load_index(tmp_path / "one", config, runtime, root, False)


def test_help_and_no_setup_does_not_fabricate_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for script in ("build_index.py", "predict.py", "prepare_data.py", "prepare_models.py", "evaluate.py", "run_experiments.py"):
        result = subprocess.run([sys.executable, str(root / script), "--help"], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    from rag.models import load_counter
    from rag.config import load_settings
    config, runtime, config_root = load_settings(root / "config.json")
    with pytest.raises(ValueError, match="revision"):
        load_counter(config, runtime, config_root)
