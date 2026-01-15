"""Tests for embedding generation."""

import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


class MockSentenceTransformer:
    """Mock sentence transformer for testing."""

    def __init__(self, model_name: str = "test", device: str = None):
        self.model_name = model_name
        self.dimension = 384

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(
        self, texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False
    ):
        if isinstance(texts, str):
            # Single text - return 1D array
            np.random.seed(hash(texts) % 2**32)
            return np.random.randn(self.dimension).astype(np.float32)
        else:
            # Batch - return 2D array
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % 2**32)
                embeddings.append(np.random.randn(self.dimension).astype(np.float32))
            return np.stack(embeddings)


# Create mock module before importing EmbeddingModel
mock_st_module = ModuleType("sentence_transformers")
mock_st_module.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = mock_st_module

from self_distill.evidence.embeddings import EmbeddingModel  # noqa: E402


@pytest.fixture
def mock_transformer():
    """Fixture for test consistency (mock already installed)."""
    yield


class TestEmbeddingModelInit:
    """Tests for EmbeddingModel initialization."""

    def test_init_default_model(self, mock_transformer):
        model = EmbeddingModel()
        assert model.model_name == "all-MiniLM-L6-v2"
        assert model.cache_dir is None

    def test_init_custom_model(self, mock_transformer):
        model = EmbeddingModel(model_name="all-mpnet-base-v2")
        assert model.model_name == "all-mpnet-base-v2"

    def test_init_with_cache_dir(self, mock_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            model = EmbeddingModel(cache_dir=cache_dir)
            assert model.cache_dir == cache_dir
            assert cache_dir.exists()


class TestEmbeddingModelEmbed:
    """Tests for single text embedding."""

    def test_embed_returns_vector(self, mock_transformer):
        model = EmbeddingModel()
        embedding = model.embed("test text")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_same_text_same_result(self, mock_transformer):
        model = EmbeddingModel()
        emb1 = model.embed("test text")
        emb2 = model.embed("test text")
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_different_text_different_result(self, mock_transformer):
        model = EmbeddingModel()
        emb1 = model.embed("text one")
        emb2 = model.embed("text two")
        assert not np.array_equal(emb1, emb2)

    def test_embed_with_cache(self, mock_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model = EmbeddingModel(cache_dir=cache_dir)

            # First embed
            emb1 = model.embed("cached text")

            # Check cache file exists
            cache_files = list(cache_dir.glob("*.npy"))
            assert len(cache_files) == 1

            # Second embed should use cache
            emb2 = model.embed("cached text")
            np.testing.assert_array_equal(emb1, emb2)


class TestEmbeddingModelEmbedBatch:
    """Tests for batch embedding."""

    def test_embed_batch_empty(self, mock_transformer):
        model = EmbeddingModel()
        result = model.embed_batch([])
        assert len(result) == 0

    def test_embed_batch_single(self, mock_transformer):
        model = EmbeddingModel()
        result = model.embed_batch(["single text"])
        assert result.shape == (1, 384)

    def test_embed_batch_multiple(self, mock_transformer):
        model = EmbeddingModel()
        texts = ["text one", "text two", "text three"]
        result = model.embed_batch(texts)
        assert result.shape == (3, 384)

    def test_embed_batch_with_cache(self, mock_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model = EmbeddingModel(cache_dir=cache_dir)

            texts = ["text a", "text b"]
            result1 = model.embed_batch(texts)

            # Should have cache files
            cache_files = list(cache_dir.glob("*.npy"))
            assert len(cache_files) == 2

            # Re-embed should use cache
            result2 = model.embed_batch(texts)
            np.testing.assert_array_equal(result1, result2)


class TestEmbeddingModelDimension:
    """Tests for embedding dimension."""

    def test_dimension_property(self, mock_transformer):
        model = EmbeddingModel()
        assert model.dimension == 384
