from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass


def lexical_tokens(text):
    import snowballstemmer

    if not hasattr(lexical_tokens, "stemmer"):
        lexical_tokens.stemmer = snowballstemmer.stemmer("russian")
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е").replace("\u0305", "\u0304")
    tokens = re.findall(r"\\[a-z]+|z\u0304|[а-яa-z]+|\d+(?:\.\d+)*|[∂∫∮∞≠≤≥+−=^]", text)
    # Stemming affects only the search representation, never the displayed source.
    return [lexical_tokens.stemmer.stemWord(t) if re.fullmatch("[а-я]+", t) else t for t in tokens]


@dataclass(frozen=True)
class Chunk:
    block_id: str
    page: int
    section: str
    start: int
    end: int
    text: str


def make_chunks(blocks, chars, overlap, fits=None):
    chunks = []
    for block in blocks:
        start = 0
        while start < len(block.text):
            end = min(start + chars, len(block.text))
            if end < len(block.text):
                boundary = block.text.rfind("\n", start + chars // 2, end)
                if boundary > start:
                    end = boundary + 1
            if fits:
                while not fits(block.text[start:end]):
                    end = start + (end - start) // 2
                    if end <= start:
                        raise ValueError("Embedding tokenizer cannot fit even one character")
            text = block.text[start:end]
            if text.strip():
                chunks.append(Chunk(block.block_id, block.page, block.section, start, end, text))
            if end == len(block.text):
                break
            start = max(start + 1, end - min(overlap, (end - start) // 3))
    return chunks


class BM25:
    def __init__(self, tokenized, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.lengths = [len(tokens) for tokens in tokenized]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.postings = defaultdict(list)
        for i, tokens in enumerate(tokenized):
            for word, frequency in Counter(tokens).items():
                self.postings[word].append((i, frequency))

    def search(self, query, k):
        scores = defaultdict(float)
        total = len(self.lengths)
        for term in set(lexical_tokens(query)):
            posting = self.postings.get(term, [])
            idf = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, frequency in posting:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_length, 1))
                scores[i] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]

    def export(self):
        return {"lengths": self.lengths, "avg_length": self.avg_length, "postings": dict(self.postings)}

    @classmethod
    def restore(cls, data, k1, b):
        obj = cls([], k1, b)
        obj.lengths, obj.avg_length = data["lengths"], data["avg_length"]
        obj.postings = data["postings"]
        return obj


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += 1 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def query_text(question):
    # No question ID, category, answer labels or gold enters the retrieval query.
    return question.question + "\n" + "\n".join(o.text for o in question.options)


class Retriever:
    def __init__(self, chunks, bm25, config, dense_index=None, embedder=None):
        self.chunks, self.bm25, self.config = chunks, bm25, config
        self.dense_index, self.embedder = dense_index, embedder

    def search(self, question, mode):
        query = query_text(question)
        settings = self.config.retrieval
        lexical = self.bm25.search(query, settings.candidate_k)
        if mode == "bm25":
            return lexical
        if self.dense_index is None or self.embedder is None:
            raise ValueError("Hybrid search requires a dense index and local embedding weights")
        tokens = self.embedder.tokenizer(query, add_special_tokens=False)["input_ids"]
        # Avoid silent query truncation: split long queries deterministically and fuse.
        step = self.config.embedding.max_seq_length - 32
        parts = [self.embedder.tokenizer.decode(tokens[i:i + step]) for i in range(0, len(tokens), step)]
        if not parts:
            parts = [query]
        pending, texts = parts[:], []
        while pending:
            part = pending.pop(0)
            text = self.config.embedding.query_prefix + part
            if len(self.embedder.tokenizer(text)["input_ids"]) <= self.config.embedding.max_seq_length:
                texts.append(text)
            elif len(part) > 1:
                middle = len(part) // 2
                pending[0:0] = [part[:middle], part[middle:]]
            else:
                raise ValueError("Query fragment cannot fit the embedding tokenizer")
        vectors = self.embedder.encode(
            texts,
            batch_size=self.config.embedding.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")
        _, positions = self.dense_index.search(vectors, min(settings.candidate_k, len(self.chunks)))
        dense = rrf([[int(i) for i in row if i >= 0] for row in positions], settings.rrf_k)
        return rrf([lexical, dense], settings.rrf_k)[:settings.candidate_k]

    def context_order(self, ranked, neighbors):
        # First preserve diversity across source pages/blocks, then add nearby windows.
        seen_blocks, seeds, remaining = set(), [], []
        for i in ranked:
            if self.chunks[i].block_id not in seen_blocks:
                seen_blocks.add(self.chunks[i].block_id)
                seeds.append(i)
            else:
                remaining.append(i)
        ordered = seeds[:4]
        if neighbors:
            radius = self.config.retrieval.neighbor_radius
            adjacent = []
            for seed in seeds[:4]:
                chunk = self.chunks[seed]
                nearby = []
                for offset in range(1, radius + 1):
                    for j in (seed - offset, seed + offset):
                        if 0 <= j < len(self.chunks):
                            other = self.chunks[j]
                            if abs(other.page - chunk.page) <= 1 and other.section == chunk.section:
                                nearby.append(j)
                adjacent.append(nearby)
            for position in range(max(map(len, adjacent), default=0)):
                ordered.extend(items[position] for items in adjacent if position < len(items))
        ordered.extend(seeds[4:])
        ordered.extend(remaining)
        return list(dict.fromkeys(ordered))

