# Cognitive Synergy Architecture - Development Guide

## Project Overview

The CSA is a research implementation of a brain-inspired memory system for LLMs.
It addresses fundamental limitations through three integrated components:

1. **HKG** (Hierarchical Knowledge Graph) - Persistent, structured memory
2. **DRM** (Dynamic Relevance Modulator) - Context-aware attention
3. **AFL** (Associative Feedback Layer) - Self-correcting learning

## Setup

### Prerequisites

- Python 3.12+
- Neo4j 5.x (running locally or remotely)
- uv (recommended) or pip

### Installation

```bash
cd /cognitive-synergy-architecture

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"

# Optional: Install PSL support
uv pip install -e ".[psl]"

# Optional: Install all features
uv pip install -e ".[all]"
```

### Neo4j Setup

```bash
# Using Docker
docker run \
    --name neo4j-csa \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Verify connection
# Open http://localhost:7474 in browser
```

## Architecture

### Data Flow

```
Text Input
    ↓
Entity Extraction
    ↓
HKG (Knowledge Graph)
    ↓
DRM (Relevance Computation)
    ↓
LLM Controller
    ↓
Response Generation
    ↓
AFL (Feedback & Learning)
```

### Component Interaction

```python
# Initialize
csa = CognitiveArchitecture(...)

# Ingest knowledge
csa.ingest_text("Once upon a time...")

# Query with attention
response = csa.query("What happened?")

# Provide feedback
csa.provide_feedback(query, response, is_correct=True)
```

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure
- [x] Core component interfaces
- [ ] Neo4j integration
- [ ] Basic entity extraction

### Phase 2: Core Implementation
- [ ] HKG: Full graph operations
- [ ] DRM: Attention mechanism training
- [ ] AFL: PSL integration
- [ ] Test suite expansion

### Phase 3: Integration
- [ ] End-to-end pipeline
- [ ] LLM controller integration
- [ ] Benchmark suite
- [ ] Documentation

### Phase 4: Advanced Features
- [ ] Multi-modal support
- [ ] Distributed inference
- [ ] Real-time learning
- [ ] Production optimization

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=csa --cov-report=html

# Run specific test
pytest tests/test_basic.py -v
```

## Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking
mypy src/

# All checks
ruff format . && ruff check . && mypy src/ && pytest
```

## Research References

See PROPOSAL.md for complete research foundation.

Key papers:
1. Random Tree Memory (RTM)
2. Hippocampal-Prefrontal Feedback Loops
3. Probabilistic Soft Logic (PSL)
4. Memory-Augmented Neural Networks

## Contributing

1. Create feature branch
2. Implement with tests
3. Run quality checks
4. Submit for review

## License

Apache 2.0 - See LICENSE file
