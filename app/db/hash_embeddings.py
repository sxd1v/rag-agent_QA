"""Deterministic local embeddings for offline smoke tests and demos.

This is not a semantic embedding model. It is useful when external embedding
quota is unavailable, but real retrieval evaluation should use a production
embedding provider.
"""

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """A lightweight hashing-vector embedding implementation."""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        lower = text.lower()
        words = re.findall(r"[a-z0-9_+-]+", lower)
        chars = [ch for ch in lower if "\u4e00" <= ch <= "\u9fff"]
        bigrams = [
            lower[i : i + 2]
            for i in range(max(len(lower) - 1, 0))
            if any("\u4e00" <= ch <= "\u9fff" for ch in lower[i : i + 2])
        ]
        return words + chars + bigrams

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)
