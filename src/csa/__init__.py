"""
Cognitive Synergy Architecture (CSA)

A neuro-symbolic framework for advanced AI memory and reasoning,
implementing brain-inspired memory systems for LLMs.

Core Components:
- HKG: Hierarchical Knowledge Graph (persistent world model)
- DRM: Dynamic Relevance Modulator (attention mechanism)
- AFL: Associative Feedback Layer (learning & refinement)
"""

from csa.architecture import CognitiveArchitecture
from csa.knowledge_graph import HierarchicalKnowledgeGraph
from csa.attention import DynamicRelevanceModulator
from csa.learning import AssociativeFeedbackLayer

__version__ = "0.1.0"
__author__ = "Ty (angrysky56)"

__all__ = [
    "CognitiveArchitecture",
    "HierarchicalKnowledgeGraph",
    "DynamicRelevanceModulator",
    "AssociativeFeedbackLayer",
]
