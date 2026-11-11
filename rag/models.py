from __future__ import annotations

import os
import re
from pathlib import Path

from .io import read_json, sha256_file


def offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def locked_model(config, root, role, runtime=None):
    lock = read_json(root / config.models_lock_path)
    entry = lock[role]
    spec = config.embedding if role == "embedding" else runtime.llm
    if entry["repository"] != spec.repository or entry["revision"] != spec.revision:
        raise ValueError(f"{role}: model lock does not match configuration/runtime")
    if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
        raise ValueError("A pinned immutable model revision is required")
    model_dir = (root / config.models_dir).resolve()
    path = (model_dir / entry["directory"]).resolve()
    if not path.is_relative_to(model_dir):
        raise ValueError("Model directory escapes models_dir")
    actual_files = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and ".cache" not in p.relative_to(path).parts}
    if actual_files != set(entry["files"]):
        raise ValueError(f"{role}: undeclared or missing model assets")
    for name, expected in entry["files"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path) or sha256_file(file) != expected:
            raise ValueError(f"Invalid local model asset: {name}")
    return path


def load_embedder(config, root):
    offline_mode()
    from sentence_transformers import SentenceTransformer

    path = locked_model(config, root, "embedding")
    model = SentenceTransformer(str(path), device="cpu", local_files_only=True, trust_remote_code=False)
    count = sum(p.numel() for p in model.parameters())
    if count > config.embedding.max_parameters:
        raise ValueError(f"Embedding parameter budget exceeded: {count}")
    model.max_seq_length = config.embedding.max_seq_length
    model.eval()
    return model, count


class ChatCounter:
    def __init__(self, tokenizer, kwargs):
        self.tokenizer, self.kwargs = tokenizer, kwargs

    def count(self, messages):
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.kwargs
        ))


def load_counter(config, runtime, root):
    offline_mode()
    if runtime.llm.revision is None or runtime.llm.quantization is None or not runtime.llm.server_revision_verified:
        raise ValueError("runtime: pin the Qwen revision and confirm the serving revision/quantization before generation")
    from transformers import AutoTokenizer

    path = locked_model(config, root, "tokenizer", runtime)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Generator tokenizer has no chat template")
    return ChatCounter(tokenizer, runtime.llm.chat_template_kwargs)

