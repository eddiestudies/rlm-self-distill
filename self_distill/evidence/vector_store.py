"""
Vector storage using FAISS for efficient similarity search.

Supports:
- Adding vectors with associated metadata
- Similarity search (k-nearest neighbors)
- Persistence (save/load)
- Incremental updates
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SearchResult:
    """Result from a similarity search."""

    id: str
    text: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Store and search vectors using FAISS."""

    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension.
            index_type: FAISS index type.
                       "flat" - exact search, good for <100k vectors
                       "ivf" - approximate search, good for larger datasets
        """
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    @property
    def index(self):
        """Lazy load FAISS index."""
        if self._index is None:
            import faiss

            if self.index_type == "flat":
                # Exact L2 search - convert to cosine later
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            elif self.index_type == "ivf":
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

        return self._index

    def __len__(self) -> int:
        """Return number of vectors in store."""
        return len(self._ids)

    def add(
        self,
        id: str,
        text: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for this item.
            text: Original text (stored for retrieval).
            vector: Embedding vector.
            metadata: Optional metadata dict.
        """
        if id in self._ids:
            # Update existing
            idx = self._ids.index(id)
            self._texts[idx] = text
            self._metadata[idx] = metadata or {}
            # Note: FAISS doesn't support in-place updates, would need rebuild
            return

        # Normalize for cosine similarity
        vector = vector / np.linalg.norm(vector)
        vector = vector.astype(np.float32).reshape(1, -1)

        self.index.add(vector)
        self._ids.append(id)
        self._texts.append(text)
        self._metadata.append(metadata or {})

    def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        vectors: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Add multiple vectors to the store.

        Args:
            ids: List of unique identifiers.
            texts: List of original texts.
            vectors: Embedding vectors (n, dimension).
            metadata: Optional list of metadata dicts.
        """
        if metadata is None:
            metadata = [{}] * len(ids)

        # Normalize all vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        vectors = vectors.astype(np.float32)

        self.index.add(vectors)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadata.extend(metadata)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.
            threshold: Optional minimum similarity threshold (0-1).

        Returns:
            List of SearchResult objects, sorted by similarity (highest first).
        """
        if len(self) == 0:
            return []

        # Normalize query
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Search
        k = min(k, len(self))
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            # Convert inner product to similarity (already normalized, so it's cosine)
            similarity = float(score)

            if threshold is not None and similarity < threshold:
                continue

            results.append(SearchResult(
                id=self._ids[idx],
                text=self._texts[idx],
                score=similarity,
                metadata=self._metadata[idx],
            ))

        return results

    def get(self, id: str) -> Optional[tuple[str, dict]]:
        """Get text and metadata by ID."""
        if id not in self._ids:
            return None
        idx = self._ids.index(id)
        return self._texts[idx], self._metadata[idx]

    def save(self, path: Path) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory to save to.
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "ids": self._ids,
            "texts": self._texts,
            "metadata": self._metadata,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """
        Load a vector store from disk.

        Args:
            path: Directory to load from.

        Returns:
            Loaded VectorStore instance.
        """
        import faiss

        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
        )
        store._index = faiss.read_index(str(path / "index.faiss"))
        store._ids = metadata["ids"]
        store._texts = metadata["texts"]
        store._metadata = metadata["metadata"]

        return store
