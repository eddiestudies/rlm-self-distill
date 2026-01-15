"""
Cluster detection for finding patterns in task data.

Detects when multiple similar tasks have been seen, indicating
a pattern that could benefit from tool creation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from self_distill.evidence.vector_store import VectorStore, SearchResult


@dataclass
class Cluster:
    """A group of similar tasks."""

    query_text: str
    members: list[SearchResult]
    centroid_score: float  # Average similarity to query

    @property
    def size(self) -> int:
        """Number of tasks in the cluster."""
        return len(self.members)

    @property
    def ids(self) -> list[str]:
        """IDs of all cluster members."""
        return [m.id for m in self.members]

    @property
    def texts(self) -> list[str]:
        """Texts of all cluster members."""
        return [m.text for m in self.members]

    def sample(self, n: int = 3, include_query: bool = False) -> list[str]:
        """
        Sample representative texts from the cluster.

        Args:
            n: Number of samples to return.
            include_query: Whether to include the query text.

        Returns:
            List of sample texts.
        """
        samples = []

        if include_query:
            samples.append(self.query_text)
            n -= 1

        if n <= 0:
            return samples

        # Sample from members, preferring higher similarity
        if len(self.members) <= n:
            samples.extend([m.text for m in self.members])
        else:
            # Weight by similarity score
            weights = np.array([m.score for m in self.members])
            weights = weights / weights.sum()

            indices = np.random.choice(
                len(self.members),
                size=min(n, len(self.members)),
                replace=False,
                p=weights,
            )
            samples.extend([self.members[i].text for i in indices])

        return samples

    def format_for_prompt(self, n_samples: int = 3) -> str:
        """
        Format cluster info for inclusion in a prompt.

        Args:
            n_samples: Number of example texts to include.

        Returns:
            Formatted string describing the cluster.
        """
        samples = self.sample(n=n_samples)
        examples = "\n".join(f"  - {s[:100]}{'...' if len(s) > 100 else ''}" for s in samples)

        return f"""Pattern detected: {self.size} similar tasks found (avg similarity: {self.centroid_score:.2f})

Examples:
{examples}

Consider creating a reusable tool for this pattern."""


class ClusterDetector:
    """Detect clusters of similar tasks."""

    def __init__(
        self,
        vector_store: VectorStore,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
    ):
        """
        Initialize the cluster detector.

        Args:
            vector_store: VectorStore to search in.
            similarity_threshold: Minimum similarity to be considered part of cluster.
            min_cluster_size: Minimum number of similar tasks to form a cluster.
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def find_cluster(
        self,
        query_vector: np.ndarray,
        query_text: str,
        max_results: int = 50,
    ) -> Optional[Cluster]:
        """
        Find a cluster of similar tasks for a query.

        Args:
            query_vector: Embedding of the query text.
            query_text: The query text itself.
            max_results: Maximum number of similar items to consider.

        Returns:
            Cluster if enough similar tasks found, None otherwise.
        """
        results = self.vector_store.search(
            query_vector,
            k=max_results,
            threshold=self.similarity_threshold,
        )

        if len(results) < self.min_cluster_size:
            return None

        # Calculate centroid score (average similarity)
        centroid_score = np.mean([r.score for r in results])

        return Cluster(
            query_text=query_text,
            members=results,
            centroid_score=centroid_score,
        )

    def should_create_tool(
        self,
        query_vector: np.ndarray,
        query_text: str,
    ) -> tuple[bool, Optional[Cluster]]:
        """
        Determine if a tool should be created based on cluster evidence.

        Args:
            query_vector: Embedding of the query text.
            query_text: The query text itself.

        Returns:
            Tuple of (should_create, cluster).
        """
        cluster = self.find_cluster(query_vector, query_text)

        if cluster is None:
            return False, None

        # Additional heuristics could be added here:
        # - Check if tasks in cluster share metadata (e.g., same task_type)
        # - Check if cluster is growing over time
        # - Check if existing tools don't already cover this pattern

        return True, cluster


@dataclass
class ClusterStats:
    """Statistics about clusters in the data."""

    total_tasks: int
    num_clusters: int
    largest_cluster_size: int
    avg_cluster_size: float
    unclustered_tasks: int
    clusters: list[Cluster] = field(default_factory=list)


def analyze_clusters(
    vector_store: VectorStore,
    embeddings: np.ndarray,
    texts: list[str],
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> ClusterStats:
    """
    Analyze all tasks to find natural clusters.

    This is useful for understanding the data before processing.

    Args:
        vector_store: VectorStore with all task embeddings.
        embeddings: All task embeddings (n, dimension).
        texts: All task texts.
        similarity_threshold: Minimum similarity for clustering.
        min_cluster_size: Minimum cluster size.

    Returns:
        ClusterStats with analysis results.
    """
    detector = ClusterDetector(
        vector_store,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )

    clusters = []
    clustered_ids = set()

    # Find clusters starting from each unclustered point
    for i, (emb, text) in enumerate(zip(embeddings, texts)):
        task_id = f"task_{i}"
        if task_id in clustered_ids:
            continue

        cluster = detector.find_cluster(emb, text)
        if cluster:
            clusters.append(cluster)
            clustered_ids.update(cluster.ids)

    total_clustered = len(clustered_ids)
    total_tasks = len(texts)

    return ClusterStats(
        total_tasks=total_tasks,
        num_clusters=len(clusters),
        largest_cluster_size=max((c.size for c in clusters), default=0),
        avg_cluster_size=np.mean([c.size for c in clusters]) if clusters else 0,
        unclustered_tasks=total_tasks - total_clustered,
        clusters=clusters,
    )
