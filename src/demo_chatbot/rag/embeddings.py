"""Turn text into vectors so we can search by meaning.

HashingEmbedder is offline and dependency-free: it hashes content words into a
fixed-size vector. Crude but real — enough to rank relevance and test the whole
RAG pipeline without downloading a model. Swap for real embeddings in prod.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

# Common words carry no meaning for matching; dropping them makes similarity
# reflect CONTENT words ("bicycle", "refund") instead of "the/is/of" overlap.
_STOPWORDS = frozenset(
    "a an the is are was were be to of in on at for and or but if then with as "
    "do does did i you it this that these those can could would should may "
    "what which who how where when".split()
)


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalized vectors."""
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                if token in _STOPWORDS or len(token) <= 1:
                    continue
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[row, bucket] += 1.0
        # normalize so dot product == cosine similarity later
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms