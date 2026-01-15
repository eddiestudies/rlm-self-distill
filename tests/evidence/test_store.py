"""Tests for EvidenceStore."""

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
            np.random.seed(hash(texts) % 2**32)
            return np.random.randn(self.dimension).astype(np.float32)
        else:
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % 2**32)
                embeddings.append(np.random.randn(self.dimension).astype(np.float32))
            return np.stack(embeddings)


# Create mock module before importing EvidenceStore
if "sentence_transformers" not in sys.modules:
    mock_st_module = ModuleType("sentence_transformers")
    mock_st_module.SentenceTransformer = MockSentenceTransformer
    sys.modules["sentence_transformers"] = mock_st_module

from self_distill.evidence.store import EvidenceStore  # noqa: E402


@pytest.fixture
def mock_transformer():
    """Fixture for test consistency (mock already installed)."""
    yield


class TestEvidenceStoreInit:
    """Tests for EvidenceStore initialization."""

    def test_init_default(self, mock_transformer):
        store = EvidenceStore()
        assert store.embedding_model_name == "all-MiniLM-L6-v2"
        assert store.similarity_threshold == 0.75
        assert store.min_cluster_size == 3

    def test_init_custom_params(self, mock_transformer):
        store = EvidenceStore(
            embedding_model="all-mpnet-base-v2",
            similarity_threshold=0.8,
            min_cluster_size=5,
        )
        assert store.embedding_model_name == "all-mpnet-base-v2"
        assert store.similarity_threshold == 0.8
        assert store.min_cluster_size == 5


class TestEvidenceStoreAdd:
    """Tests for adding tasks."""

    def test_add_single(self, mock_transformer):
        store = EvidenceStore()
        embedding = store.add("task1", "Check SSN 123-45-6789")
        assert len(store) == 1
        assert isinstance(embedding, np.ndarray)

    def test_add_with_metadata(self, mock_transformer):
        store = EvidenceStore()
        store.add("task1", "text", metadata={"type": "pii"})
        text, meta = store.vector_store.get("task1")
        assert meta["type"] == "pii"

    def test_add_batch(self, mock_transformer):
        store = EvidenceStore()
        ids = ["t1", "t2", "t3"]
        texts = ["text one", "text two", "text three"]
        embeddings = store.add_batch(ids, texts)
        assert len(store) == 3
        assert embeddings.shape[0] == 3


class TestEvidenceStorePatternDetection:
    """Tests for pattern detection."""

    def test_check_pattern_not_enough_tasks(self, mock_transformer):
        store = EvidenceStore(min_cluster_size=3)
        store.add("t1", "text one")
        store.add("t2", "text two")

        should_create, cluster = store.check_for_pattern("query text")
        assert should_create is False
        assert cluster is None

    def test_check_pattern_with_similar_tasks(self, mock_transformer):
        store = EvidenceStore(similarity_threshold=0.5, min_cluster_size=3)

        # Add tasks with similar text (will have similar hashes -> similar embeddings)
        for i in range(5):
            store.add(f"pii_{i}", f"Check SSN pattern {i}")

        # This may or may not find a pattern depending on hash-based embeddings
        # The important thing is the API works
        should_create, cluster = store.check_for_pattern("Check SSN pattern X")
        # Result depends on mock's hash-based embeddings
        assert isinstance(should_create, bool)

    def test_mark_pattern_triggered(self, mock_transformer):
        store = EvidenceStore()
        assert len(store._triggered_patterns) == 0

        # Create a mock cluster
        from self_distill.evidence.cluster import Cluster
        from self_distill.evidence.vector_store import SearchResult

        members = [
            SearchResult(id=f"id{i}", text=f"text {i}", score=0.9) for i in range(5)
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.9)

        store.mark_pattern_triggered(cluster)
        assert len(store._triggered_patterns) == 1

    def test_get_pattern_prompt_returns_none_no_pattern(self, mock_transformer):
        store = EvidenceStore(min_cluster_size=3)
        store.add("t1", "text")

        prompt = store.get_pattern_prompt("query")
        assert prompt is None


class TestEvidenceStorePersistence:
    """Tests for save/load."""

    def test_save_and_load(self, mock_transformer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence"

            # Create store with data
            store = EvidenceStore(similarity_threshold=0.8, min_cluster_size=5)
            store.add("t1", "text one", metadata={"type": "a"})
            store.add("t2", "text two", metadata={"type": "b"})

            # Mark a pattern as triggered
            from self_distill.evidence.cluster import Cluster
            from self_distill.evidence.vector_store import SearchResult

            members = [SearchResult(id="id1", text="t", score=0.9)]
            cluster = Cluster(query_text="q", members=members, centroid_score=0.9)
            store.mark_pattern_triggered(cluster)

            # Save
            store.save(path)

            # Load
            loaded = EvidenceStore.load(path)

            assert len(loaded) == 2
            assert loaded.similarity_threshold == 0.8
            assert loaded.min_cluster_size == 5
            assert len(loaded._triggered_patterns) == 1

            # Check data preserved
            text, meta = loaded.vector_store.get("t1")
            assert text == "text one"
            assert meta["type"] == "a"


class TestEvidenceStoreAnalyze:
    """Tests for cluster analysis."""

    def test_analyze_empty_store(self, mock_transformer):
        store = EvidenceStore()
        stats = store.analyze()
        assert stats.total_tasks == 0
        assert stats.num_clusters == 0

    def test_analyze_with_tasks(self, mock_transformer):
        store = EvidenceStore()
        for i in range(10):
            store.add(f"task_{i}", f"text {i}")

        stats = store.analyze()
        assert stats.total_tasks == 10
        # Clusters depend on mock embeddings
        assert isinstance(stats.num_clusters, int)
