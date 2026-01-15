"""
Evidence Store - Main interface for evidence-based tool creation.

Combines embeddings, vector storage, and cluster detection into
a simple API for tracking tasks and detecting patterns.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from self_distill.evidence.embeddings import EmbeddingModel
from self_distill.evidence.vector_store import VectorStore
from self_distill.evidence.cluster import (
    Cluster,
    ClusterDetector,
    ClusterStats,
    analyze_clusters,
)


class EvidenceStore:
    """
    Main interface for evidence-based tool creation.

    Tracks task embeddings and detects patterns to guide tool creation.

    Usage:
        store = EvidenceStore()

        # As tasks are processed, add them
        for task in tasks:
            # Check if we should create a tool
            should_create, cluster = store.check_for_pattern(task.text)
            if should_create:
                prompt_addition = cluster.format_for_prompt()
                # Add to model prompt: "Pattern detected..."

            # Always add the task for future pattern detection
            store.add(task.id, task.text, metadata={"type": task.type})

        # Save for future runs
        store.save("evidence_store/")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the evidence store.

        Args:
            embedding_model: Sentence-transformer model name.
            similarity_threshold: Minimum similarity for pattern detection.
            min_cluster_size: Minimum tasks to form a pattern.
            cache_dir: Optional directory for embedding cache.
            device: Device for embeddings ("cpu", "cuda", "mps").
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

        self._embedder = EmbeddingModel(
            model_name=embedding_model,
            cache_dir=cache_dir,
            device=device,
        )
        self._vector_store: Optional[VectorStore] = None
        self._cluster_detector: Optional[ClusterDetector] = None

        # Track which patterns have already triggered tool creation
        self._triggered_patterns: set[str] = set()

    @property
    def embedder(self) -> EmbeddingModel:
        """Get the embedding model."""
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(dimension=self._embedder.dimension)
        return self._vector_store

    @property
    def cluster_detector(self) -> ClusterDetector:
        """Get or create the cluster detector."""
        if self._cluster_detector is None:
            self._cluster_detector = ClusterDetector(
                vector_store=self.vector_store,
                similarity_threshold=self.similarity_threshold,
                min_cluster_size=self.min_cluster_size,
            )
        return self._cluster_detector

    def __len__(self) -> int:
        """Return number of tracked tasks."""
        return len(self.vector_store)

    def add(
        self,
        id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Add a task to the evidence store.

        Args:
            id: Unique identifier for this task.
            text: Task text to embed.
            metadata: Optional metadata (e.g., task_type, labels).

        Returns:
            The embedding vector for the task.
        """
        embedding = self._embedder.embed(text)
        self.vector_store.add(id, text, embedding, metadata)
        return embedding

    def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        metadata: Optional[list[dict]] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Add multiple tasks to the evidence store.

        Args:
            ids: List of unique identifiers.
            texts: List of task texts.
            metadata: Optional list of metadata dicts.
            show_progress: Show progress bar during embedding.

        Returns:
            Embedding vectors for all tasks.
        """
        embeddings = self._embedder.embed_batch(texts, show_progress=show_progress)
        self.vector_store.add_batch(ids, texts, embeddings, metadata)
        return embeddings

    def check_for_pattern(
        self,
        text: str,
        exclude_triggered: bool = True,
    ) -> tuple[bool, Optional[Cluster]]:
        """
        Check if a task matches an existing pattern.

        Args:
            text: Task text to check.
            exclude_triggered: Skip patterns that already triggered tool creation.

        Returns:
            Tuple of (should_create_tool, cluster).
        """
        if len(self.vector_store) < self.min_cluster_size:
            return False, None

        embedding = self._embedder.embed(text)
        should_create, cluster = self.cluster_detector.should_create_tool(
            embedding, text
        )

        if not should_create or cluster is None:
            return False, None

        # Check if this pattern already triggered
        if exclude_triggered:
            # Use first few member IDs as pattern fingerprint
            pattern_key = tuple(sorted(cluster.ids[:5]))
            if pattern_key in self._triggered_patterns:
                return False, None

        return True, cluster

    def mark_pattern_triggered(self, cluster: Cluster) -> None:
        """
        Mark a pattern as having triggered tool creation.

        Prevents the same pattern from triggering multiple times.

        Args:
            cluster: The cluster that triggered tool creation.
        """
        pattern_key = tuple(sorted(cluster.ids[:5]))
        self._triggered_patterns.add(pattern_key)

    def get_pattern_prompt(
        self,
        text: str,
        n_samples: int = 3,
    ) -> Optional[str]:
        """
        Get a prompt addition if a pattern is detected.

        Convenience method combining check_for_pattern and format_for_prompt.

        Args:
            text: Task text to check.
            n_samples: Number of examples to include in prompt.

        Returns:
            Formatted prompt string if pattern found, None otherwise.
        """
        should_create, cluster = self.check_for_pattern(text)
        if not should_create or cluster is None:
            return None

        self.mark_pattern_triggered(cluster)
        return cluster.format_for_prompt(n_samples=n_samples)

    def analyze(self) -> ClusterStats:
        """
        Analyze all stored tasks to find natural clusters.

        Useful for understanding patterns in the data.

        Returns:
            ClusterStats with analysis results.
        """
        if len(self.vector_store) == 0:
            return ClusterStats(
                total_tasks=0,
                num_clusters=0,
                largest_cluster_size=0,
                avg_cluster_size=0,
                unclustered_tasks=0,
            )

        # Re-embed all texts to get embeddings matrix
        texts = self.vector_store._texts
        embeddings = self._embedder.embed_batch(texts)

        return analyze_clusters(
            self.vector_store,
            embeddings,
            texts,
            similarity_threshold=self.similarity_threshold,
            min_cluster_size=self.min_cluster_size,
        )

    def save(self, path: Path) -> None:
        """
        Save the evidence store to disk.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vector store
        self.vector_store.save(path / "vectors")

        # Save config and state
        config = {
            "embedding_model": self.embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "min_cluster_size": self.min_cluster_size,
            "triggered_patterns": [list(p) for p in self._triggered_patterns],
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "EvidenceStore":
        """
        Load an evidence store from disk.

        Args:
            path: Directory to load from.
            device: Device for embeddings.

        Returns:
            Loaded EvidenceStore instance.
        """
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        store = cls(
            embedding_model=config["embedding_model"],
            similarity_threshold=config["similarity_threshold"],
            min_cluster_size=config["min_cluster_size"],
            device=device,
        )

        store._vector_store = VectorStore.load(path / "vectors")
        store._triggered_patterns = {
            tuple(p) for p in config.get("triggered_patterns", [])
        }

        return store
