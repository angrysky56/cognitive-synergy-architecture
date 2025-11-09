# Cognitive Synergy Architecture (CSA)

Don't try to use this yet- unfinished and completely theoretical.

**A Neuro-Symbolic Framework for Advanced AI Memory and Reasoning**

## Overview

The Cognitive Synergy Architecture addresses fundamental limitations in Large Language Models by implementing a brain-inspired memory system. Drawing from neuroscience and cognitive science, CSA provides structured, prioritized, and self-correcting memory capabilities.

## Core Components

### 1. Hierarchical Knowledge Graph (HKG)
- Structured representation of narrative elements and relationships
- Based on Random Tree Memory (RTM) framework
- Persistent world model preventing narrative amnesia
- Neo4j-backed graph database

### 2. Dynamic Relevance Modulator (DRM)
- Attention mechanism inspired by frontal-visual cortex dynamics
- Dynamically assigns relevance gains to graph nodes/edges
- Task-aware computational resource allocation
- Flexible, context-sensitive focus

### 3. Associative Feedback Layer (AFL)
- Learning mechanism based on hippocampal-prefrontal feedback loops
- Strengthens correct associations (beta-like updates)
- Weakens outdated connections (theta-like updates)
- Implemented with Probabilistic Soft Logic (PSL)

## Theoretical Foundations

- **Neuroscience**: Dynamic resource allocation in working memory
- **Cognitive Science**: Random Tree Model (RTM) for hierarchical encoding
- **Brain Rhythms**: Beta/theta frequency bands for learning signals
- **Machine Learning**: Memory-Augmented Neural Networks (MANNs)

## Key Features

- **Solves Proactive Interference**: Actively prunes irrelevant connections
- **Dynamic Prioritization**: Context-aware attention mechanism
- **Narrative Coherence**: Maintains consistent long-form generation
- **Complex Reasoning**: Structured pathways enable multi-step logic
- **Continual Learning**: Self-correcting feedback loops

## Installation

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

```python
from csa import CognitiveArchitecture
from csa.knowledge_graph import HierarchicalKnowledgeGraph
from csa.attention import DynamicRelevanceModulator
from csa.learning import AssociativeFeedbackLayer

# Initialize CSA
csa = CognitiveArchitecture(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Add narrative knowledge
csa.ingest_text("Once upon a time...")

# Query with dynamic attention
response = csa.query("What happened to the main character?")
```

## Research Paper

This implementation is based on the research proposal:
**"The Cognitive Synergy Architecture: A Neuro-Symbolic Framework for Advanced AI Memory and Reasoning"**

See [PROPOSAL.md](PROPOSAL.md) for the complete research document.

## Project Structure

```
cognitive-synergy-architecture/
├── src/csa/                    # Main package
│   ├── __init__.py
│   ├── architecture.py         # Main CSA orchestration
│   ├── knowledge_graph/        # HKG implementation
│   ├── attention/              # DRM implementation
│   ├── learning/               # AFL implementation
│   └── utils/
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
└── pyproject.toml
```

## Requirements

- Python 3.12+
- Neo4j 5.x
- PyTorch 2.x
- PSL (via pslpython)

## Development

```bash
# Run tests
pytest

# Format code
ruff format .

# Type checking
mypy src/

# Linting
ruff check .
```

## References

1. Random Tree Memory (RTM) Framework
2. Neuroscience of Working Memory Prioritization
3. Hippocampal-Prefrontal Brainwave Rhythms
4. Probabilistic Soft Logic (PSL)
5. Memory-Augmented Neural Networks

## License

Apache 2.0

## Authors

Research and implementation by Ty (angrysky56)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
