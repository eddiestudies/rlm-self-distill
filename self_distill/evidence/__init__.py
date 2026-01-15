"""
Evidence-Based Tool Creation

This module provides infrastructure for detecting patterns in task data
to guide tool creation decisions. Instead of relying on the LLM to
recognize patterns on-the-fly, we use embeddings and clustering to
find similar tasks and provide evidence for tool creation.

Components:
- embeddings: Generate vector representations of tasks
- vector_store: Store and retrieve vectors with FAISS
- cluster: Detect clusters and sample representative tasks

Usage:
    from self_distill.evidence import EvidenceStore

    store = EvidenceStore(embedding_model="all-MiniLM-L6-v2")

    # Add tasks as they're processed
    store.add("task_001", "Check if SSN 123-45-6789 is valid")
    store.add("task_002", "Validate email user@example.com")
    store.add("task_003", "Check SSN 987-65-4321 format")

    # Check if new task matches a cluster
    cluster = store.find_cluster("Verify SSN 555-12-3456", threshold=0.8)
    if cluster and cluster.size >= 3:
        samples = cluster.sample(n=3)
        # Prompt model: "Found {cluster.size} similar tasks. Examples: {samples}"
"""

from self_distill.evidence.embeddings import EmbeddingModel
from self_distill.evidence.vector_store import VectorStore
from self_distill.evidence.cluster import ClusterDetector, Cluster
from self_distill.evidence.store import EvidenceStore

__all__ = [
    "EmbeddingModel",
    "VectorStore",
    "ClusterDetector",
    "Cluster",
    "EvidenceStore",
]
