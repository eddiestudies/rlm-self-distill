"""
Embedding generation for task text.

Uses sentence-transformers for efficient local embeddings.
Supports caching to avoid re-computing embeddings for the same text.
"""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


class EmbeddingModel:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Sentence-transformer model name.
                        "all-MiniLM-L6-v2" is fast and good quality (384 dims).
                        "all-mpnet-base-v2" is higher quality (768 dims).
            cache_dir: Optional directory to cache embeddings.
            device: Device to use ("cpu", "cuda", "mps"). Auto-detected if None.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._device = device

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def _text_hash(self, text: str) -> str:
        """Generate a hash for cache lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_path(self, text_hash: str) -> Optional[Path]:
        """Get cache file path for a text hash."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{text_hash}.npy"

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.
            use_cache: Whether to use cached embeddings if available.

        Returns:
            numpy array of shape (dimension,)
        """
        if use_cache and self.cache_dir:
            text_hash = self._text_hash(text)
            cache_path = self._cache_path(text_hash)
            if cache_path and cache_path.exists():
                return np.load(cache_path)

        embedding = self.model.encode(text, convert_to_numpy=True)

        if use_cache and self.cache_dir:
            cache_path = self._cache_path(text_hash)
            if cache_path:
                np.save(cache_path, embedding)

        return embedding

    def embed_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cached embeddings.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        embeddings = []
        texts_to_encode = []
        text_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache and self.cache_dir:
                text_hash = self._text_hash(text)
                cache_path = self._cache_path(text_hash)
                if cache_path and cache_path.exists():
                    embeddings.append((i, np.load(cache_path)))
                    continue

            texts_to_encode.append(text)
            text_indices.append(i)

        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )

            for idx, text, emb in zip(text_indices, texts_to_encode, new_embeddings):
                embeddings.append((idx, emb))

                # Cache the new embedding
                if use_cache and self.cache_dir:
                    text_hash = self._text_hash(text)
                    cache_path = self._cache_path(text_hash)
                    if cache_path:
                        np.save(cache_path, emb)

        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in embeddings])
