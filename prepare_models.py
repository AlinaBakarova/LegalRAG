"""The only command that may download model artifacts; run before indexing/inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import load_settings
from rag.io import read_json, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resolve-revisions", action="store_true", help="Resolve repository HEAD exactly once, persist SHAs into config/runtime")
    args = parser.parse_args()
    config, runtime, root = load_settings(args.config)
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    roles = [("embedding", config.embedding), ("tokenizer", runtime.llm)]
    lock = {"schema_version": 1}
    for role, spec in roles:
        revision = spec.revision
        if revision is None:
            if not args.resolve_revisions:
                raise ValueError(f"{role}: set an immutable revision or explicitly use --resolve-revisions")
            revision = api.model_info(spec.repository, token=False).sha
        elif len(revision) != 40:
            raise ValueError("Revisions must be immutable 40-character commit SHAs")
        relative = f"{role}-{revision}"
        target = root / config.models_dir / relative
        patterns = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.vocab", "*.merges"]
        if role == "embedding":
            patterns.append("*.safetensors")
        # Never download generator weights: generation must use the assigned API.
        snapshot_download(spec.repository, revision=revision, local_dir=target, allow_patterns=patterns, token=False)
        files = {p.relative_to(target).as_posix(): sha256_file(p) for p in sorted(target.rglob("*")) if p.is_file() and ".cache" not in p.parts}
        if not files:
            raise ValueError(f"No assets downloaded for {role}")
        lock[role] = {"repository": spec.repository, "revision": revision, "directory": relative, "files": files}
        spec.revision = revision
    write_json(root / config.runtime_path, runtime.model_dump())
    write_json(Path(args.config), config.model_dump())
    write_json(root / config.models_lock_path, lock)
    print("Revisions and asset hashes locked. Serving revision/quantization still need verification in runtime.json.")


if __name__ == "__main__":
    main()
