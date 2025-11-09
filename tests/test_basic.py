"""
Basic tests for CSA components.
"""

import pytest

from csa.attention import DynamicRelevanceModulator
from csa.knowledge_graph import HierarchicalKnowledgeGraph
from csa.learning import AssociativeFeedbackLayer


def test_hkg_initialization() -> None:
    """Test HKG can be initialized."""
    # Note: Requires Neo4j running for full test
    # This is a placeholder for structure
    pass


def test_drm_initialization() -> None:
    """Test DRM can be initialized."""
    drm = DynamicRelevanceModulator(embedding_dim=768)
    assert drm.embedding_dim == 768
    assert drm.num_heads == 8


def test_afl_initialization() -> None:
    """Test AFL can be initialized."""
    # Mock graph for testing
    afl = AssociativeFeedbackLayer(graph=None, learning_rate=0.1)
    assert afl.learning_rate == 0.1
    assert afl.stats["total_updates"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
