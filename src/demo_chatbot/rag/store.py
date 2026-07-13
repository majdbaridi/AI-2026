"""Store text chunks as vectors and find the closest ones to a query.

InMemoryVectorStore keeps everything in a numpy array — no database needed.
Because vectors are normalized, a dot product == cosine similarity, so the
"closest" match is just the highest dot product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchHit]:
        if self._matrix is None:
            return []
        scores = self._matrix @ query_vector          # similarity to every chunk
        top = np.argsort(scores)[::-1][:k]             # indexes of the k best
        return [SearchHit(self._chunks[i], float(scores[i])) for i in top]