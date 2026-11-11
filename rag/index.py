from __future__ import annotations

from pathlib import Path

from .config import index_signature
from .io import read_json, read_jsonl, sha256_file
from .schemas import Block


def load_index(index_path, config, runtime, root, hybrid):
    from .models import load_embedder, offline_mode
    from .retrieval import BM25, Chunk, Retriever

    offline_mode()
    directory = Path(index_path).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["index_signature"] != index_signature(config):
        raise ValueError("Index does not match this configuration")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != expected:
            raise ValueError(f"Index integrity check failed: {name}")
    blocks = {b.block_id: b for b in read_jsonl(directory / "blocks.jsonl", Block)}
    chunks = [Chunk(**c) for c in read_jsonl(directory / "chunks.jsonl")]
    for c in chunks:
        block = blocks[c.block_id]
        if (not 0 <= c.start < c.end <= len(block.text) or c.text != block.text[c.start:c.end]
                or c.page != block.page or c.section != block.section):
            raise ValueError("Chunk provenance check failed")
    bm25 = BM25.restore(read_json(directory / "bm25.json"), config.retrieval.bm25_k1, config.retrieval.bm25_b)
    if len(bm25.lengths) != len(chunks):
        raise ValueError("BM25/chunk count mismatch")
    embedder, dense = None, None
    if hybrid:
        if not manifest["dense"]:
            raise ValueError("This index is lexical only; rebuild with pinned embedding weights")
        embedder, _ = load_embedder(config, root)
        import faiss
        faiss.omp_set_num_threads(runtime.resources.cpu_threads)
        dense = faiss.read_index(str(directory / "dense.faiss"))
        if dense.ntotal != len(chunks):
            raise ValueError("FAISS/chunk count mismatch")
    return Retriever(chunks, bm25, config, dense, embedder), manifest

