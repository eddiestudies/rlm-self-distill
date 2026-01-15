"""Tests for vector storage."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from self_distill.evidence.vector_store import VectorStore, SearchResult


@pytest.fixture
def dimension():
    return 384


@pytest.fixture
def store(dimension):
    return VectorStore(dimension=dimension)


@pytest.fixture
def sample_vectors(dimension):
    """Generate sample vectors that are normalized."""
    np.random.seed(42)
    vectors = np.random.randn(10, dimension).astype(np.float32)
    # Normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


class TestVectorStoreInit:
    """Tests for VectorStore initialization."""

    def test_init_creates_empty_store(self, dimension):
        store = VectorStore(dimension=dimension)
        assert len(store) == 0
        assert store.dimension == dimension

    def test_init_flat_index(self, dimension):
        store = VectorStore(dimension=dimension, index_type="flat")
        assert store.index_type == "flat"


class TestVectorStoreAdd:
    """Tests for adding vectors."""

    def test_add_single_vector(self, store, sample_vectors):
        store.add("id1", "text one", sample_vectors[0])
        assert len(store) == 1

    def test_add_multiple_vectors(self, store, sample_vectors):
        for i in range(5):
            store.add(f"id{i}", f"text {i}", sample_vectors[i])
        assert len(store) == 5

    def test_add_with_metadata(self, store, sample_vectors):
        store.add("id1", "text", sample_vectors[0], metadata={"type": "test"})
        text, meta = store.get("id1")
        assert text == "text"
        assert meta["type"] == "test"

    def test_add_batch(self, store, sample_vectors, dimension):
        ids = [f"id{i}" for i in range(5)]
        texts = [f"text {i}" for i in range(5)]
        vectors = sample_vectors[:5]

        store.add_batch(ids, texts, vectors)
        assert len(store) == 5


class TestVectorStoreSearch:
    """Tests for similarity search."""

    def test_search_empty_store(self, store, sample_vectors):
        results = store.search(sample_vectors[0], k=5)
        assert len(results) == 0

    def test_search_returns_results(self, store, sample_vectors):
        # Add vectors
        for i in range(5):
            store.add(f"id{i}", f"text {i}", sample_vectors[i])

        # Search with first vector - should find itself
        results = store.search(sample_vectors[0], k=5)
        assert len(results) == 5
        assert results[0].id == "id0"
        assert results[0].score > 0.99  # Near-perfect match

    def test_search_with_threshold(self, store, sample_vectors):
        for i in range(5):
            store.add(f"id{i}", f"text {i}", sample_vectors[i])

        # High threshold should filter results
        results = store.search(sample_vectors[0], k=5, threshold=0.99)
        # Only the exact match should pass
        assert len(results) >= 1
        assert results[0].id == "id0"

    def test_search_result_structure(self, store, sample_vectors):
        store.add("id1", "test text", sample_vectors[0], metadata={"key": "value"})

        results = store.search(sample_vectors[0], k=1)
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.id == "id1"
        assert result.text == "test text"
        assert result.score > 0
        assert result.metadata == {"key": "value"}


class TestVectorStoreGet:
    """Tests for retrieving by ID."""

    def test_get_existing(self, store, sample_vectors):
        store.add("id1", "text one", sample_vectors[0], metadata={"a": 1})
        text, meta = store.get("id1")
        assert text == "text one"
        assert meta == {"a": 1}

    def test_get_nonexistent(self, store):
        result = store.get("nonexistent")
        assert result is None


class TestVectorStorePersistence:
    """Tests for save/load."""

    def test_save_and_load(self, sample_vectors, dimension):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store"

            # Create and populate store
            store = VectorStore(dimension=dimension)
            for i in range(5):
                store.add(f"id{i}", f"text {i}", sample_vectors[i], metadata={"i": i})

            # Save
            store.save(path)

            # Load
            loaded = VectorStore.load(path)

            assert len(loaded) == 5
            assert loaded.dimension == dimension

            # Check data preserved
            text, meta = loaded.get("id2")
            assert text == "text 2"
            assert meta["i"] == 2

            # Check search still works
            results = loaded.search(sample_vectors[0], k=1)
            assert results[0].id == "id0"
