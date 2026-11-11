from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, model_validator

from .io import digest_json, read_json
from .schemas import StrictModel, Text


class Limits(StrictModel):
    max_llm_calls: Annotated[int, Field(ge=1, le=4)] = 4
    max_input_tokens: Annotated[int, Field(ge=1, le=16384)] = 16384
    max_output_tokens: Annotated[int, Field(ge=1, le=4096)] = 4096
    max_request_tokens: Annotated[int, Field(ge=1, le=8192)] = 8192


class Resources(StrictModel):
    ram_mb: Annotated[int, Field(ge=512)]
    cpu_threads: Annotated[int, Field(ge=1)]
    request_timeout_seconds: Annotated[float, Field(gt=0)]


class LLM(StrictModel):
    repository: Text
    revision: str | None
    quantization: str | None
    generation: dict[str, Any]
    chat_template_kwargs: dict[str, Any]
    server_revision_verified: bool = False

    @model_validator(mode="after")
    def allowed_model(self):
        if self.repository != "Qwen/Qwen3.5-4B":
            raise ValueError("Only Qwen/Qwen3.5-4B is allowed for generation")
        if self.revision is not None and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Generator revision must be an immutable 40-character commit")
        allowed = {"temperature", "top_p", "top_k", "max_tokens", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty", "response_format"}
        if not set(self.generation) <= allowed:
            raise ValueError("Unrecognized generation parameters; explicitly adapt the runtime schema")
        maximum = self.generation.get("max_tokens")
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("A fixed max_tokens in 1..4096 is required")
        if "enable_thinking" not in self.chat_template_kwargs:
            raise ValueError("Thinking mode must be explicit")
        if self.quantization is not None and not self.quantization.strip():
            raise ValueError("Empty quantization")
        return self


class Runtime(StrictModel):
    schema_version: Annotated[int, Field(ge=1, le=1)]
    provenance: Text
    official: bool
    llm: LLM
    limits: Limits
    resources: Resources

    @model_validator(mode="after")
    def request_fits(self):
        if self.llm.generation["max_tokens"] >= self.limits.max_request_tokens:
            raise ValueError("No room for input tokens")
        return self


class Embedding(StrictModel):
    repository: Text
    revision: str | None
    max_parameters: Annotated[int, Field(ge=1, le=1_000_000_000)]
    batch_size: Annotated[int, Field(ge=1, le=256)]
    max_seq_length: Annotated[int, Field(ge=32, le=512)]
    query_prefix: str
    passage_prefix: str


class Retrieval(StrictModel):
    chunk_chars: Annotated[int, Field(ge=100, le=4000)]
    overlap_chars: Annotated[int, Field(ge=0)]
    candidate_k: Annotated[int, Field(ge=1, le=200)]
    context_chunks: Annotated[int, Field(ge=1, le=24)]
    rrf_k: Annotated[int, Field(ge=1)]
    bm25_k1: Annotated[float, Field(gt=0)]
    bm25_b: Annotated[float, Field(ge=0, le=1)]
    neighbor_radius: Annotated[int, Field(ge=0, le=2)]

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError("overlap_chars must be below chunk_chars")
        return self


class Config(StrictModel):
    schema_version: Annotated[int, Field(ge=1, le=1)]
    runtime_path: Text
    models_lock_path: Text
    models_dir: Text
    embedding: Embedding
    retrieval: Retrieval
    token_margin: Annotated[int, Field(ge=32, le=1024)]
    final_max_calls: Annotated[int, Field(ge=1, le=4)]


def load_settings(config_path):
    path = Path(config_path).resolve()
    config = Config.model_validate(read_json(path))
    runtime = Runtime.model_validate(read_json(path.parent / config.runtime_path))
    return config, runtime, path.parent


def index_signature(config: Config):
    # Search-time k, RRF and context settings may change without rebuilding vectors.
    return digest_json({
        "chunk_chars": config.retrieval.chunk_chars,
        "overlap_chars": config.retrieval.overlap_chars,
        "embedding": config.embedding.model_dump(),
        "lexical_normalization": "unicode-snowball-ru-v2",
    })

