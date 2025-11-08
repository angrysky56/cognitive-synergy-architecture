"""
Dynamic Relevance Modulator (DRM) - Attention Mechanism.

Inspired by frontal-visual cortex dynamics for resource allocation.
Computes relevance gains for nodes/edges based on task context.
"""

import torch
import torch.nn as nn
from typing import Any


class DynamicRelevanceModulator(nn.Module):
    """
    Neural attention mechanism for dynamic resource allocation.
    
    Implements brain-inspired attention by:
    - Computing relevance scores for graph nodes/edges
    - Dynamically adjusting computational "gain"
    - Focusing on task-relevant information
    - Down-weighting irrelevant/distracting data
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        """
        Initialize the attention module.
        
        Args:
            embedding_dim: Dimension of node/edge embeddings
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for context-aware relevance
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # MLP for relevance gain computation
        self.gain_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Gain in [0, 1]
        )
        
        self.stats: dict[str, Any] = {
            "forward_passes": 0,
            "avg_gain": 0.0,
        }
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        task_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relevance gains for nodes given task context.
        
        Args:
            node_embeddings: [batch_size, num_nodes, embedding_dim]
            task_context: [batch_size, context_length, embedding_dim]
            
        Returns:
            Relevance gains: [batch_size, num_nodes, 1]
        """
        batch_size, num_nodes, _ = node_embeddings.shape
        
        # Multi-head attention: nodes attend to task context
        attended, _ = self.attention(
            query=node_embeddings,
            key=task_context,
            value=task_context,
        )
        
        # Concatenate original and attended representations
        combined = torch.cat([node_embeddings, attended], dim=-1)
        
        # Compute relevance gain for each node
        gains = self.gain_network(combined)  # [batch_size, num_nodes, 1]
        
        # Update statistics
        self.stats["forward_passes"] += 1
        self.stats["avg_gain"] = float(gains.mean().item())
        
        return gains
    
    def compute_edge_gains(
        self,
        node_gains: torch.Tensor,
        edge_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate node gains to edges.
        
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
    
    def get_stats(self) -> dict[str, Any]:
        """Get module statistics."""
        return self.stats.copy()
