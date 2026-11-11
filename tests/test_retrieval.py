from __future__ import annotations

import numpy as np

from rag.retrieval import BM25, Retriever, lexical_tokens, make_chunks, query_text, rrf


def test_bm25_russian_and_formula_signs():
    docs = [lexical_tokens("Условия Коши—Римана ∂f + i∂f"), lexical_tokens("Вычет интеграла")]
    assert BM25(docs).search("условиями Коши Римана", 2)[0] == 0
    assert "+" in docs[0]
    assert rrf([[1, 2], [2, 3]], 60)[0] == 2
    assert rrf([[1, 1, 2]], 60) == rrf([[1, 2]], 60)


def test_question_ids_and_option_labels_do_not_enter_retrieval(question):
    changed = question.model_copy(deep=True)
    changed.id = "totally-new-id"
    for i, option in enumerate(changed.options):
        option.id = str(i + 100)
    assert query_text(changed) == query_text(question)


def test_neighbors_change_context_but_keep_source_ids(blocks, settings):
    config, _, _ = settings
    chunks = make_chunks(blocks, 170, 30)
    retriever = Retriever(chunks, BM25([lexical_tokens(c.text) for c in chunks]), config)
    # Unranked adjacent windows must be added by the neighbor component.
    plain = retriever.context_order([1, 6], False)
    expanded = retriever.context_order([1, 6], True)
    assert set(expanded) > set(plain)
    assert all(chunks[i].block_id in {b.block_id for b in blocks} for i in expanded)


def test_real_faiss_cosine_and_stable_fusion():
    import faiss
    index = faiss.IndexFlatIP(2)
    vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype="float32")
    index.add(vectors)
    _, dense = index.search(np.array([[1, 0]], dtype="float32"), 3)
    assert dense.tolist() == [[0, 1, 2]]
    assert rrf([[2, 0, 1], dense[0].tolist()], 60)[0] == 0


def test_hybrid_path_with_real_faiss_and_mock_embedding(blocks, settings, question):
    import faiss
    config, _, _ = settings
    chunks = make_chunks(blocks, 170, 30)
    index = faiss.IndexFlatIP(2)
    vectors = np.array([[0, 1]] * len(chunks), dtype="float32")
    vectors[-1] = [1, 0]
    index.add(vectors)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text)}
        def decode(self, tokens):
            return "".join(tokens)

    class MockEmbedder:
        tokenizer = Tokenizer()
        encoded = []
        def encode(self, texts, **kwargs):
            self.encoded.extend(texts)
            return np.array([[1, 0]] * len(texts), dtype="float32")

    embedder = MockEmbedder()
    retriever = Retriever(chunks, BM25([lexical_tokens(c.text) for c in chunks]), config, index, embedder)
    ranked = retriever.search(question, "hybrid")
    assert len(chunks) - 1 in ranked
    assert embedder.encoded and all(len(t) <= config.embedding.max_seq_length for t in embedder.encoded)
    assert all(t.startswith(config.embedding.query_prefix) for t in embedder.encoded)


