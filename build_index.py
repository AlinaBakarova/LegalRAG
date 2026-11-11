from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rag.config import index_signature, load_settings
from rag.io import read_jsonl, sha256_file, write_json, write_jsonl
from rag.schemas import Block


def build(corpus, output, config_path, lexical_only=False):
    started = time.perf_counter()
    config, runtime, root = load_settings(config_path)
    from rag.resources import configure_resources, measurements

    stop = configure_resources(runtime.resources)
    from rag.models import load_embedder, offline_mode
    from rag.retrieval import BM25, lexical_tokens, make_chunks

    offline_mode()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Index output must be empty: build into a new directory")
    blocks = read_jsonl(corpus, Block)
    embedder, parameter_count, fits = None, None, None
    if not lexical_only:
        embedder, parameter_count = load_embedder(config, root)

        def fits(text):
            return len(embedder.tokenizer(config.embedding.passage_prefix + text)["input_ids"]) <= config.embedding.max_seq_length

    chunks = make_chunks(blocks, config.retrieval.chunk_chars, config.retrieval.overlap_chars, fits)
    if not chunks:
        raise ValueError("No indexable text")
    bm25 = BM25([lexical_tokens(c.section + "\n" + c.text) for c in chunks])
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "blocks.jsonl", blocks)
    write_jsonl(output / "chunks.jsonl", [asdict(c) for c in chunks])
    write_json(output / "bm25.json", bm25.export())
    if embedder is not None:
        import faiss
        faiss.omp_set_num_threads(runtime.resources.cpu_threads)
        dense = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
        batch = config.embedding.batch_size
        for start in range(0, len(chunks), batch):
            texts = [config.embedding.passage_prefix + c.text for c in chunks[start:start + batch]]
            vectors = embedder.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            dense.add(vectors.astype("float32"))
        faiss.write_index(dense, str(output / "dense.faiss"))
    manifest = {
        "schema_version": 1, "corpus_sha256": sha256_file(corpus),
        "index_signature": index_signature(config),
        "embedding_revision": config.embedding.revision if embedder is not None else None,
        "embedding_parameters": parameter_count,
        "block_count": len(blocks), "chunk_count": len(chunks),
        "dense": embedder is not None,
        "files": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
        "resources": measurements(started),
    }
    manifest["index_bytes_without_manifest"] = sum(p.stat().st_size for p in output.iterdir())
    write_json(output / "manifest.json", manifest)
    stop.set()
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a CPU index; never calls an LLM or downloads models.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lexical-only", action="store_true", help="Offline diagnostic BM25 index; cannot run hybrid profiles")
    args = parser.parse_args()
    try:
        result = build(args.corpus, args.output, args.config, args.lexical_only)
        print(f"Indexed {result['block_count']} blocks, {result['chunk_count']} chunks", file=sys.stderr)
    except Exception as exc:
        print(f"build_index failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
