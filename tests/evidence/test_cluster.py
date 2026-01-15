"""Tests for cluster detection."""

import numpy as np
import pytest

from self_distill.evidence.vector_store import VectorStore, SearchResult
from self_distill.evidence.cluster import Cluster, ClusterDetector, ClusterStats


@pytest.fixture
def dimension():
    return 384


@pytest.fixture
def store(dimension):
    return VectorStore(dimension=dimension)


def make_similar_vectors(base_vector: np.ndarray, n: int, noise: float = 0.1) -> list[np.ndarray]:
    """Create vectors similar to base with small noise."""
    vectors = []
    for i in range(n):
        noise_vec = np.random.randn(*base_vector.shape) * noise
        vec = base_vector + noise_vec
        vec = vec / np.linalg.norm(vec)  # Normalize
        vectors.append(vec.astype(np.float32))
    return vectors


class TestCluster:
    """Tests for Cluster dataclass."""

    def test_cluster_size(self):
        members = [
            SearchResult(id="1", text="text 1", score=0.9),
            SearchResult(id="2", text="text 2", score=0.85),
            SearchResult(id="3", text="text 3", score=0.8),
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.85)
        assert cluster.size == 3

    def test_cluster_ids(self):
        members = [
            SearchResult(id="a", text="text a", score=0.9),
            SearchResult(id="b", text="text b", score=0.85),
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.875)
        assert cluster.ids == ["a", "b"]

    def test_cluster_texts(self):
        members = [
            SearchResult(id="1", text="first text", score=0.9),
            SearchResult(id="2", text="second text", score=0.85),
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.875)
        assert cluster.texts == ["first text", "second text"]

    def test_cluster_sample_all(self):
        members = [
            SearchResult(id="1", text="text 1", score=0.9),
            SearchResult(id="2", text="text 2", score=0.85),
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.875)
        samples = cluster.sample(n=5)  # More than available
        assert len(samples) == 2

    def test_cluster_sample_subset(self):
        members = [
            SearchResult(id=str(i), text=f"text {i}", score=0.9 - i * 0.01)
            for i in range(10)
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.85)
        samples = cluster.sample(n=3)
        assert len(samples) == 3
        # All samples should be from member texts
        for sample in samples:
            assert sample in cluster.texts

    def test_cluster_sample_with_query(self):
        members = [SearchResult(id="1", text="member", score=0.9)]
        cluster = Cluster(query_text="query text", members=members, centroid_score=0.9)
        samples = cluster.sample(n=2, include_query=True)
        assert "query text" in samples

    def test_cluster_format_for_prompt(self):
        members = [
            SearchResult(id="1", text="SSN: 123-45-6789", score=0.9),
            SearchResult(id="2", text="SSN: 987-65-4321", score=0.85),
            SearchResult(id="3", text="SSN: 555-12-3456", score=0.8),
        ]
        cluster = Cluster(query_text="query", members=members, centroid_score=0.85)
        prompt = cluster.format_for_prompt(n_samples=2)

        assert "3 similar tasks" in prompt
        assert "0.85" in prompt
        assert "Consider creating a reusable tool" in prompt


class TestClusterDetector:
    """Tests for ClusterDetector."""

    def test_find_cluster_not_enough_similar(self, store, dimension):
        np.random.seed(42)

        # Add dissimilar vectors
        for i in range(5):
            vec = np.random.randn(dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            store.add(f"id{i}", f"text {i}", vec)

        detector = ClusterDetector(store, similarity_threshold=0.9, min_cluster_size=3)

        query = np.random.randn(dimension).astype(np.float32)
        cluster = detector.find_cluster(query, "query text")

        # Should not find cluster - vectors are random/dissimilar
        assert cluster is None

    def test_find_cluster_enough_similar(self, store, dimension):
        np.random.seed(42)

        # Create a base vector and similar vectors with very low noise
        base = np.random.randn(dimension).astype(np.float32)
        base = base / np.linalg.norm(base)

        similar_vectors = make_similar_vectors(base, 5, noise=0.01)  # Lower noise for higher similarity

        for i, vec in enumerate(similar_vectors):
            store.add(f"id{i}", f"similar text {i}", vec)

        detector = ClusterDetector(store, similarity_threshold=0.7, min_cluster_size=3)  # Lower threshold

        # Query with the base vector
        cluster = detector.find_cluster(base, "query similar text")

        assert cluster is not None
        assert cluster.size >= 3

    def test_should_create_tool_true(self, store, dimension):
        np.random.seed(42)
        base = np.random.randn(dimension).astype(np.float32)
        base = base / np.linalg.norm(base)

        similar_vectors = make_similar_vectors(base, 5, noise=0.01)  # Lower noise
        for i, vec in enumerate(similar_vectors):
            store.add(f"id{i}", f"text {i}", vec)

        detector = ClusterDetector(store, similarity_threshold=0.7, min_cluster_size=3)  # Lower threshold
        should_create, cluster = detector.should_create_tool(base, "query")

        assert should_create is True
        assert cluster is not None

    def test_should_create_tool_false_not_enough(self, store, dimension):
        np.random.seed(42)

        # Only add 2 vectors
        for i in range(2):
            vec = np.random.randn(dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            store.add(f"id{i}", f"text {i}", vec)

        detector = ClusterDetector(store, similarity_threshold=0.8, min_cluster_size=3)

        query = np.random.randn(dimension).astype(np.float32)
        should_create, cluster = detector.should_create_tool(query, "query")

        assert should_create is False
        assert cluster is None


class TestClusterStats:
    """Tests for ClusterStats dataclass."""

    def test_cluster_stats_structure(self):
        stats = ClusterStats(
            total_tasks=100,
            num_clusters=5,
            largest_cluster_size=25,
            avg_cluster_size=15.0,
            unclustered_tasks=25,
        )
        assert stats.total_tasks == 100
        assert stats.num_clusters == 5
        assert stats.largest_cluster_size == 25
        assert stats.avg_cluster_size == 15.0
        assert stats.unclustered_tasks == 25
