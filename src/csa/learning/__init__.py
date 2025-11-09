"""
Associative Feedback Layer (AFL) - Learning & Refinement.

Inspired by hippocampal-prefrontal brainwave feedback loops.
Implements beta-like (correct) and theta-like (error) weight updates.
Simulates frequency-specific feedback for associative learning.

References:
- Beta-band (8-30Hz): Success signaling, strengthens connections
- Theta-band (4-8Hz): Error signaling, weakens connections
- Directional Influence: PFC→HPC (theta), HPC→PFC (beta)
- Theta-Gamma Coupling: Sequential encoding
"""

import math
from typing import Any


class AssociativeFeedbackLayer:
    """
    Self-correcting learning mechanism for knowledge graph refinement.

    Implements brain-inspired feedback loops:
    - Beta-like updates: Strengthen correct associations (8-30Hz analogue)
    - Theta-like updates: Weaken incorrect/outdated connections (4-8Hz analogue)
    - Directional influence: PFC→HPC for errors, HPC→PFC for success
    - Theta-gamma coupling: Sequential organization of information

    The AFL addresses proactive interference by pruning outdated associations
    and reinforcing validated knowledge pathways.
    """

    def __init__(
        self,
        graph: Any,
        learning_rate: float = 0.1,
        beta_factor: float = 0.3,  # Reinforcement strength (success)
        theta_factor: float = 0.2,  # Weakening strength (error)
        gamma_frequency: int = 40,  # Hz for sequential encoding
        theta_frequency: int = 6,  # Hz for segmentation
        prune_threshold: float = 0.1,  # Minimum weight to keep
    ) -> None:
        """
        Initialize the feedback layer.

        Args:
            graph: HierarchicalKnowledgeGraph instance
            learning_rate: Base rate of weight updates
            beta_factor: Strength of positive reinforcement (beta-band)
            theta_factor: Strength of negative reinforcement (theta-band)
            gamma_frequency: High-frequency for item encoding (Hz)
            theta_frequency: Low-frequency for chunking (Hz)
            prune_threshold: Weight threshold for connection pruning
        """
        self.graph = graph
        self.learning_rate = learning_rate
        self.beta_factor = beta_factor
        self.theta_factor = theta_factor
        self.gamma_frequency = gamma_frequency
        self.theta_frequency = theta_frequency
        self.prune_threshold = prune_threshold

        # Directional pathway strengths
        # PFC → HPC: error feedback pathway
        # HPC → PFC: success feedback pathway
        self.pfc_to_hpc_strength = 1.0
        self.hpc_to_pfc_strength = 1.0

        self.stats = {
            "beta_updates": 0,  # Correct reinforcement
            "theta_updates": 0,  # Error correction
            "total_updates": 0,
            "connections_pruned": 0,
            "avg_weight_delta": 0.0,
            "pfc_hpc_activations": 0,  # Error pathway usage
            "hpc_pfc_activations": 0,  # Success pathway usage
        }

    def update_weights(
        self,
        involved_nodes: list[str],
        involved_edges: list[tuple[str, str, str]],  # (source, target, rel_type)
        is_correct: bool,
        confidence: float = 1.0,
    ) -> None:
        """
        Update graph weights based on feedback.

        Implements frequency-specific feedback:
        - Beta-like (correct): Strengthen pathways, increase HPC→PFC
        - Theta-like (error): Weaken pathways, increase PFC→HPC

        Args:
            involved_nodes: Node IDs that contributed to reasoning
            involved_edges: Edge tuples (source, target, type) that were traversed
            is_correct: Whether response was correct
            confidence: Confidence in the feedback (0-1)
        """
        if is_correct:
            self._beta_update(involved_edges, confidence)
            self.stats["beta_updates"] += 1
            # Increase HPC→PFC pathway (success feedback)
            self.hpc_to_pfc_strength = min(1.5, self.hpc_to_pfc_strength * 1.05)
            self.stats["hpc_pfc_activations"] += 1
        else:
            self._theta_update(involved_edges, confidence)
            self.stats["theta_updates"] += 1
            # Increase PFC→HPC pathway (error feedback)
            self.pfc_to_hpc_strength = min(1.5, self.pfc_to_hpc_strength * 1.05)
            self.stats["pfc_hpc_activations"] += 1

        self.stats["total_updates"] += 1

        # Decay directional pathways over time (return to baseline)
        self.pfc_to_hpc_strength *= 0.99
        self.hpc_to_pfc_strength *= 0.99

    def _beta_update(
        self,
        edges: list[tuple[str, str, str]],
        confidence: float,
    ) -> None:
        """
        Beta-like reinforcement: Strengthen correct associations.

        Simulates the high-frequency beta oscillation (8-30Hz) that
        signals successful learning and strengthens synaptic connections.

        Args:
            edges: List of (source, target, rel_type) tuples
            confidence: Confidence multiplier for update
        """
        for source_id, target_id, rel_type in edges:
            # Fetch current weight
            current_weight = self._get_edge_weight(source_id, target_id, rel_type)

            # Beta-like reinforcement with HPC→PFC modulation
            delta = (
                self.learning_rate
                * self.beta_factor
                * confidence
                * self.hpc_to_pfc_strength
                * (1.0 - current_weight)  # Diminishing returns
            )

            new_weight = min(1.0, current_weight + delta)

            # Update in graph
            self.graph.add_relationship(
                source_id,
                target_id,
                rel_type,
                weight=new_weight,
            )

            self.stats["avg_weight_delta"] = (
                self.stats["avg_weight_delta"] * 0.9 + abs(delta) * 0.1
            )

    def _theta_update(
        self,
        edges: list[tuple[str, str, str]],
        confidence: float,
    ) -> None:
        """
        Theta-like weakening: Reduce weight of incorrect associations.

        Simulates the low-frequency theta oscillation (4-8Hz) that
        signals learning errors and weakens erroneous connections.

        Args:
            edges: List of (source, target, rel_type) tuples
            confidence: Confidence multiplier for update
        """
        for source_id, target_id, rel_type in edges:
            # Fetch current weight
            current_weight = self._get_edge_weight(source_id, target_id, rel_type)

            # Theta-like weakening with PFC→HPC modulation
            delta = (
                self.learning_rate
                * self.theta_factor
                * confidence
                * self.pfc_to_hpc_strength
            )

            new_weight = max(0.0, current_weight - delta)

            # Update in graph
            self.graph.add_relationship(
                source_id,
                target_id,
                rel_type,
                weight=new_weight,
            )

            self.stats["avg_weight_delta"] = (
                self.stats["avg_weight_delta"] * 0.9 + abs(delta) * 0.1
            )

            # Auto-prune if below threshold
            if new_weight < self.prune_threshold:
                self._prune_edge(source_id, target_id, rel_type)

    def _get_edge_weight(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> float:
        """
        Retrieve current weight of an edge.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type

        Returns:
            Current weight (defaults to 1.0 if not found)
        """
        # Query graph for edge weight
        with self.graph.driver.session() as session:
            query = f"""
            MATCH (s {{id: $source}})-[r:{rel_type}]->(t {{id: $target}})
            RETURN r.weight as weight
            """

            result = session.run(
                query,
                source=source_id,
                target=target_id,
            )

            record = result.single()
            return record["weight"] if record else 1.0

    def _prune_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> None:
        """
        Remove a weak edge from the graph.

        Implements proactive interference prevention by deleting
        outdated or erroneous associations.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type
        """
        with self.graph.driver.session() as session:
            query = f"""
            MATCH (s {{id: $source}})-[r:{rel_type}]->(t {{id: $target}})
            DELETE r
            """

            session.run(
                query,
                source=source_id,
                target=target_id,
            )

        self.stats["connections_pruned"] += 1

    def theta_gamma_encode(
        self,
        sequence: list[str],
        segment_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Encode a sequence using theta-gamma coupling.

        Implements hierarchical temporal organization:
        - Theta (4-8Hz): Segments sequence into chunks
        - Gamma (40Hz): Encodes items within chunks
        - Phase-timing: Sequential order via gamma phase within theta cycle

        Args:
            sequence: List of items to encode
            segment_size: Size of theta chunks (auto-computed if None)

        Returns:
            List of encoded segments with theta/gamma timing
        """
        if segment_size is None:
            # Compute segment size based on theta/gamma ratio
            # One theta cycle contains multiple gamma cycles
            segment_size = max(1, self.gamma_frequency // self.theta_frequency)

        encoded_segments = []

        for i in range(0, len(sequence), segment_size):
            segment = sequence[i : i + segment_size]

            # Theta phase for this segment (0 to 2π)
            theta_phase = 2 * math.pi * (i / len(sequence))

            # Encode each item with gamma phase within theta cycle
            items_with_timing = []
            for j, item in enumerate(segment):
                # Gamma phase within theta cycle
                gamma_phase = 2 * math.pi * (j / len(segment))

                items_with_timing.append({
                    "item": item,
                    "theta_phase": theta_phase,
                    "gamma_phase": gamma_phase,
                    "segment_idx": len(encoded_segments),
                    "item_idx_in_segment": j,
                })

            encoded_segments.append({
                "segment_idx": len(encoded_segments),
                "theta_phase": theta_phase,
                "items": items_with_timing,
            })

        return encoded_segments

    def prune_weak_connections(self, threshold: float | None = None) -> int:
        """
        Remove connections below weight threshold.

        Combats proactive interference by removing outdated associations.
        Implements theta-like weakening at scale.

        Args:
            threshold: Minimum weight to keep (uses default if None)

        Returns:
            Number of connections pruned
        """
        if threshold is None:
            threshold = self.prune_threshold

        with self.graph.driver.session() as session:
            query = """
            MATCH ()-[r]->()
            WHERE r.weight < $threshold
            DELETE r
            RETURN count(r) as pruned_count
            """

            result = session.run(query, threshold=threshold)
            count = result.single()["pruned_count"]

        self.stats["connections_pruned"] += count
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        stats = self.stats.copy()

        # Add computed metrics
        if stats["total_updates"] > 0:
            stats["beta_ratio"] = stats["beta_updates"] / stats["total_updates"]
            stats["theta_ratio"] = stats["theta_updates"] / stats["total_updates"]

        stats["pfc_hpc_strength"] = self.pfc_to_hpc_strength
        stats["hpc_pfc_strength"] = self.hpc_to_pfc_strength

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "beta_updates": 0,
            "theta_updates": 0,
            "total_updates": 0,
            "connections_pruned": 0,
            "avg_weight_delta": 0.0,
            "pfc_hpc_activations": 0,
            "hpc_pfc_activations": 0,
        }
