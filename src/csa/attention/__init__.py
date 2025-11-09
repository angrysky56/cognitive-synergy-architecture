"""
Dynamic Relevance Modulator (DRM) - Attention Mechanism.

Inspired by frontal-visual cortex dynamics for resource allocation.
Computes relevance gains for nodes/edges based on task context.
Implements brain-inspired attention via neural gain modulation.

References:
- Top-down Control: Frontal cortex signals visual cortex priorities
- Dynamic Resource Allocation: Adjustable neural gain
- Multi-time-scale Memory: CMS with frequency-specific updates
"""

from typing import Any

import torch
import torch.nn as nn


class DynamicRelevanceModulator(nn.Module):
    """
    Neural attention mechanism for dynamic resource allocation.

    Implements brain-inspired attention by:
    - Computing relevance scores for graph nodes/edges
    - Dynamically adjusting computational "gain"
    - Focusing on task-relevant information
    - Down-weighting irrelevant/distracting data
    - Multi-time-scale memory updates (CMS concept)

    The DRM addresses the "working memory bottleneck" by prioritizing
    information rather than treating all context uniformly.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_scales: int = 3,  # Multi-time-scale memory
    ) -> None:
        """
        Initialize the attention module.

        Args:
            embedding_dim: Dimension of node/edge embeddings
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for regularization
            num_scales: Number of time scales for CMS
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales

        # Multi-head attention for context-aware relevance
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Multi-time-scale memory modules (CMS)
        # Each scale updates at different frequency
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_scales)
        ])

        # Relevance gain computation network
        # Takes concatenated [node, attended_context] and outputs gain
        self.gain_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Gain in [0, 1]
        )

        # Gating mechanism for selective storage/retrieval
        self.input_gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.forget_gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.stats: dict[str, Any] = {
            "forward_passes": 0,
            "avg_gain": 0.0,
            "avg_input_gate": 0.0,
            "avg_forget_gate": 0.0,
            "scale_activations": [0.0] * num_scales,
        }

    def forward(
        self,
        node_embeddings: torch.Tensor,
        task_context: torch.Tensor,
        scale_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute relevance gains and gating for nodes given task context.

        Implements dynamic resource allocation:
        1. Multi-head attention: nodes attend to task context
        2. Relevance gain: compute priority scores
        3. Gating: selective storage/retrieval
        4. Multi-scale: optional time-scale specific processing

        Args:
            node_embeddings: [batch_size, num_nodes, embedding_dim]
            task_context: [batch_size, context_length, embedding_dim]
            scale_idx: Optional time-scale index for CMS

        Returns:
            Dictionary with:
                - relevance_gains: [batch_size, num_nodes, 1]
                - input_gates: [batch_size, num_nodes, 1]
                - forget_gates: [batch_size, num_nodes, 1]
                - attended_nodes: [batch_size, num_nodes, embedding_dim]
        """
        batch_size, num_nodes, _ = node_embeddings.shape

        # Multi-head attention: nodes attend to task context
        # This implements the "frontal cortex signaling visual cortex" mechanism
        attended, attn_weights = self.attention(
            query=node_embeddings,
            key=task_context,
            value=task_context,
        )

        # Concatenate original and attended representations
        # This gives the gain network both baseline and context-modulated info
        combined = torch.cat([node_embeddings, attended], dim=-1)

        # Compute relevance gain for each node
        # This is the core "neural gain" mechanism for prioritization
        gains = self.gain_network(combined)  # [batch_size, num_nodes, 1]

        # Compute gating values for selective processing
        # Input gate: should we store this information?
        # Forget gate: should we discard old information?
        input_gates = self.input_gate(attended)
        forget_gates = self.forget_gate(attended)

        # Optional multi-time-scale processing (CMS)
        if scale_idx is not None and scale_idx < self.num_scales:
            scale_gain = self.scale_gates[scale_idx](attended)
            gains = gains * scale_gain
            self.stats["scale_activations"][scale_idx] = float(scale_gain.mean().item())

        # Apply gain to attended nodes (modulated representation)
        modulated_nodes = attended * gains

        # Update statistics
        self.stats["forward_passes"] += 1
        self.stats["avg_gain"] = float(gains.mean().item())
        self.stats["avg_input_gate"] = float(input_gates.mean().item())
        self.stats["avg_forget_gate"] = float(forget_gates.mean().item())

        return {
            "relevance_gains": gains,
            "input_gates": input_gates,
            "forget_gates": forget_gates,
            "attended_nodes": modulated_nodes,
            "attention_weights": attn_weights,
        }

    def compute_edge_gains(
        self,
        node_gains: torch.Tensor,
        edge_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate node gains to edges.

        Edge relevance is computed as the average of its connected nodes' gains.
        This allows attention to flow through the graph structure.

        Args:
            node_gains: [batch_size, num_nodes, 1]
            edge_indices: [2, num_edges] (source, target indices)

        Returns:
            Edge gains: [batch_size, num_edges, 1]
        """
        # Average the gains of connected nodes
        source_gains = node_gains[:, edge_indices[0]]
        target_gains = node_gains[:, edge_indices[1]]

        edge_gains = (source_gains + target_gains) / 2.0
        return edge_gains

    def apply_proactive_interference_prevention(
        self,
        current_memory: torch.Tensor,
        new_info: torch.Tensor,
        relevance_gains: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prevent proactive interference from old, irrelevant information.

        Uses relevance gains and forget gates to down-weight or remove
        outdated information that would interfere with new learning.

        Args:
            current_memory: [batch_size, memory_size, embedding_dim]
            new_info: [batch_size, num_new, embedding_dim]
            relevance_gains: [batch_size, memory_size, 1]

        Returns:
            Updated memory: [batch_size, memory_size, embedding_dim]
        """
        # Compute forget decision based on low relevance
        forget_mask = relevance_gains < 0.3  # Threshold for forgetting

        # Apply forgetting to current memory
        updated_memory = current_memory * (~forget_mask)

        return updated_memory

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "forward_passes": 0,
            "avg_gain": 0.0,
            "avg_input_gate": 0.0,
            "avg_forget_gate": 0.0,
            "scale_activations": [0.0] * self.num_scales,
        }
