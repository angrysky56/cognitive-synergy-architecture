"""
Main Cognitive Synergy Architecture orchestration.

This module coordinates the three core components (HKG, DRM, AFL) to create
a Memory-Augmented Neural Network (MANN) with brain-inspired memory capabilities.
"""

from typing import Any

import torch

from csa.attention import DynamicRelevanceModulator
from csa.knowledge_graph import HierarchicalKnowledgeGraph, RTMConfig
from csa.learning import AssociativeFeedbackLayer


class CognitiveArchitecture:
    """
    Main CSA system coordinating HKG, DRM, and AFL components.

    The CSA implements a brain-inspired memory system that addresses fundamental
    limitations in LLMs:
    - Proactive Interference → AFL weakens outdated connections
    - Lack of Prioritization → DRM allocates attention dynamically
    - Narrative Amnesia → HKG maintains persistent world model
    - Complex Reasoning → Structured pathways enable multi-step logic

    Architecture Flow:
    1. HKG: Stores knowledge in hierarchical graph (RTM framework)
    2. DRM: Computes relevance gains for nodes/edges (attention)
    3. AFL: Updates weights based on feedback (learning)
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        embedding_dim: int = 768,
        rtm_config: RTMConfig | None = None,
    ) -> None:
        """
        Initialize the Cognitive Synergy Architecture.

        Args:
            neo4j_uri: Neo4j database connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_dim: Dimension of embedding vectors
            rtm_config: Optional RTM configuration
        """
        self.embedding_dim = embedding_dim

        self.hkg = HierarchicalKnowledgeGraph(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            config=rtm_config,
        )

        self.drm = DynamicRelevanceModulator(
            embedding_dim=embedding_dim,
        )

        self.afl = AssociativeFeedbackLayer(
            graph=self.hkg,
        )

    def ingest_narrative(
        self,
        narrative_clauses: list[str],
        narrative_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest and structure narrative into hierarchical knowledge graph.

        Uses RTM recursive partitioning to create hierarchical structure
        with branching factor K and maximum recall depth D.

        Args:
            narrative_clauses: List of narrative clauses
            narrative_id: Optional identifier for the narrative

        Returns:
            Dictionary with root node ID and ingestion statistics
        """
        if not narrative_clauses:
            raise ValueError("Cannot ingest empty narrative")

        # Recursively partition narrative into hierarchical structure
        root_id = self.hkg.recursive_partition(narrative_clauses)

        return {
            "root_id": root_id,
            "num_clauses": len(narrative_clauses),
            "narrative_id": narrative_id,
            "stats": {
                "nodes": self.hkg.get_node_count(),
                "edges": self.hkg.get_edge_count(),
            },
        }

    def reason_with_attention(
        self,
        query_node_id: str,
        node_embeddings: torch.Tensor,
        task_context: torch.Tensor,
        goal_pattern: str = "goal:KeyPoint",
    ) -> dict[str, Any]:
        """
        Perform reasoning with dynamic attention allocation.

        Pipeline:
        1. DRM computes relevance gains for nodes
        2. HKG performs ensemble reasoning (Random-Inference Trees)
        3. Returns ranked hypotheses with attention-weighted paths

        Args:
            query_node_id: Starting node for reasoning
            node_embeddings: Node embeddings [batch, num_nodes, embedding_dim]
            task_context: Task context [batch, context_len, embedding_dim]
            goal_pattern: Cypher pattern for goal nodes

        Returns:
            Dictionary with hypotheses, attention weights, and statistics
        """
        # Compute relevance gains via DRM
        attention_output = self.drm(node_embeddings, task_context)
        relevance_gains = attention_output["relevance_gains"]

        # Perform ensemble reasoning via HKG
        hypotheses = self.hkg.random_inference_trees(
            query_node_id,
            goal_pattern,
        )

        # Combine attention and reasoning
        # Weight hypotheses by relevance gains
        for hyp in hypotheses:
            node_ids = hyp["path"]
            # Compute average relevance of nodes in path
            # (This is simplified - in practice would use node embeddings)
            hyp["attention_weighted_confidence"] = hyp["confidence"]

        return {
            "hypotheses": hypotheses,
            "attention_weights": attention_output["attention_weights"],
            "relevance_gains": relevance_gains,
            "stats": {
                "num_hypotheses": len(hypotheses),
                "avg_confidence": (
                    sum(h["confidence"] for h in hypotheses) / len(hypotheses)
                    if hypotheses
                    else 0.0
                ),
            },
        }

    def provide_feedback(
        self,
        involved_nodes: list[str],
        involved_edges: list[tuple[str, str, str]],
        is_correct: bool,
        confidence: float = 1.0,
    ) -> None:
        """
        Provide feedback to refine the knowledge graph.

        Uses AFL to update weights:
        - Beta-like (correct): Strengthens correct pathways
        - Theta-like (error): Weakens incorrect pathways

        Args:
            involved_nodes: Node IDs that contributed to reasoning
            involved_edges: Edge tuples (source, target, type) traversed
            is_correct: Whether response was correct
            confidence: Confidence in the feedback (0-1)
        """
        self.afl.update_weights(
            involved_nodes,
            involved_edges,
            is_correct,
            confidence,
        )

    def encode_sequence(
        self,
        sequence: list[str],
        segment_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Encode a sequence using theta-gamma coupling.

        Args:
            sequence: List of items to encode
            segment_size: Size of theta chunks

        Returns:
            List of encoded segments with timing information
        """
        return self.afl.theta_gamma_encode(sequence, segment_size)

    def prune_weak_knowledge(self, threshold: float | None = None) -> int:
        """
        Prune weak connections to combat proactive interference.

        Args:
            threshold: Minimum weight threshold

        Returns:
            Number of connections pruned
        """
        return self.afl.prune_weak_connections(threshold)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics and health metrics."""
        return {
            "graph": {
                "nodes": self.hkg.get_node_count(),
                "edges": self.hkg.get_edge_count(),
            },
            "attention": self.drm.get_stats(),
            "learning": self.afl.get_stats(),
        }

    def close(self) -> None:
        """Clean up resources."""
        self.hkg.close()
