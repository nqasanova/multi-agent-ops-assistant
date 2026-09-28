"""Section-aware chunking and TF-IDF retrieval over the internal docs.

Each chunk is one paragraph, indexed together with its document title and
section heading so a short paragraph still carries its context. TF-IDF is
enough for a corpus of this size and keeps the project runnable without
downloading an embedding model. The Retriever interface (search) is the only
thing the agents depend on, so a vector store can replace it later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@dataclass(frozen=True)
class Chunk:
    doc: str          # file name, e.g. support_sla.md
    title: str        # document title
    section: str      # section heading
    text: str         # paragraph text

    @property
    def source(self) -> str:
        return f"{self.doc} > {self.section}"


def load_chunks(docs_dir: Path = config.DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        title, section = path.stem, "Overview"
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            block = block.strip()
            if not block:
                continue
            if block.startswith("# "):
                title = block[2:].strip()
                continue
            if block.startswith("## "):
                lines = block.splitlines()
                section = lines[0][3:].strip()
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            chunks.append(Chunk(path.name, title, section, " ".join(block.split())))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        corpus = [f"{c.title} {c.section} {c.section} {c.text}" for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_chunks())
