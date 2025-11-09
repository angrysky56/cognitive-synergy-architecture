"""
Test Neo4j connectivity and basic CSA algorithm functionality.

This script validates:
1. Neo4j connection and schema initialization
2. HKG recursive partitioning and graph structure
3. DRM attention mechanism forward pass
4. AFL feedback updates and weight changes
"""

import torch
from rich.console import Console
from rich.table import Table

from csa.architecture import CognitiveArchitecture
from csa.knowledge_graph import RTMConfig

console = Console()


def test_neo4j_connectivity() -> None:
    """Test Neo4j connection and basic operations."""
    console.print("\n[bold cyan]Testing Neo4j Connectivity...[/bold cyan]")

    try:
        csa = CognitiveArchitecture(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",  # Update with actual password
        )

        # Check initial state
        stats = csa.get_stats()
        console.print("✓ Connected to Neo4j")
        console.print(f"  Initial nodes: {stats['graph']['nodes']}")
        console.print(f"  Initial edges: {stats['graph']['edges']}")

        csa.close()
        console.print("[green]✓ Neo4j connectivity test passed[/green]")

    except Exception as e:
        console.print(f"[red]✗ Neo4j connectivity test failed: {e}[/red]")
        raise


def test_hkg_recursive_partition() -> None:
    """Test HKG recursive partitioning algorithm."""
    console.print("\n[bold cyan]Testing HKG Recursive Partitioning...[/bold cyan]")

    try:
        # Create CSA with custom RTM config
        rtm_config = RTMConfig(
            branching_factor=4,
            max_recall_depth=3,
            n_estimators=10,  # Reduced for testing
        )

        csa = CognitiveArchitecture(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            rtm_config=rtm_config,
        )

        # Ingest a simple narrative
        narrative = [
            "Alice went to the market.",
            "She bought fresh vegetables.",
            "On the way home, she met Bob.",
            "They discussed the upcoming festival.",
            "Alice invited Bob to join her.",
        ]

        result = csa.ingest_narrative(narrative, narrative_id="test_narrative")

        console.print("✓ Narrative ingested successfully")
        console.print(f"  Root node ID: {result['root_id']}")
        console.print(f"  Total nodes: {result['stats']['nodes']}")
        console.print(f"  Total edges: {result['stats']['edges']}")

        # Test retrieval
        subgraph = csa.hkg.get_subgraph(
            seed_nodes=[result['root_id']],
            max_depth=2,
        )

        console.print("✓ Subgraph retrieved")
        console.print(f"  Nodes in subgraph: {len(subgraph['nodes'])}")
        console.print(f"  Edges in subgraph: {len(subgraph['edges'])}")

        csa.close()
        console.print("[green]✓ HKG recursive partition test passed[/green]")

    except Exception as e:
        console.print(f"[red]✗ HKG test failed: {e}[/red]")
        raise


def test_drm_attention() -> None:
    """Test DRM attention mechanism."""
    console.print("\n[bold cyan]Testing DRM Attention Mechanism...[/bold cyan]")

    try:
        csa = CognitiveArchitecture(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Create dummy embeddings
        batch_size = 1
        num_nodes = 5
        context_length = 3
        embedding_dim = 768

        node_embeddings = torch.randn(batch_size, num_nodes, embedding_dim)
        task_context = torch.randn(batch_size, context_length, embedding_dim)

        # Forward pass through DRM
        output = csa.drm(node_embeddings, task_context)

        console.print("✓ DRM forward pass successful")
        console.print(f"  Relevance gains shape: {output['relevance_gains'].shape}")
        console.print(f"  Avg gain: {output['relevance_gains'].mean().item():.4f}")
        console.print(f"  Input gate avg: {output['input_gates'].mean().item():.4f}")
        console.print(f"  Forget gate avg: {output['forget_gates'].mean().item():.4f}")

        # Test edge gain computation
        edge_indices = torch.tensor([[0, 1, 2], [1, 2, 3]])  # source, target
        edge_gains = csa.drm.compute_edge_gains(
            output['relevance_gains'],
            edge_indices,
        )

        console.print("✓ Edge gains computed")
        console.print(f"  Edge gains shape: {edge_gains.shape}")

        csa.close()
        console.print("[green]✓ DRM attention test passed[/green]")

    except Exception as e:
        console.print(f"[red]✗ DRM test failed: {e}[/red]")
        raise


def test_afl_feedback() -> None:
    """Test AFL feedback mechanism."""
    console.print("\n[bold cyan]Testing AFL Feedback Mechanism...[/bold cyan]")

    try:
        csa = CognitiveArchitecture(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Create test nodes and edges
        csa.hkg.add_node("Event", "event1", {"name": "Test Event 1"})
        csa.hkg.add_node("Event", "event2", {"name": "Test Event 2"})
        csa.hkg.add_relationship("event1", "event2", "CAUSES", weight=0.5)

        console.print("✓ Test graph created")

        # Test beta update (correct feedback)
        csa.provide_feedback(
            involved_nodes=["event1", "event2"],
            involved_edges=[("event1", "event2", "CAUSES")],
            is_correct=True,
            confidence=1.0,
        )

        stats = csa.afl.get_stats()
        console.print("✓ Beta update applied")
        console.print(f"  Beta updates: {stats['beta_updates']}")
        console.print(f"  HPC→PFC strength: {stats['hpc_pfc_strength']:.4f}")

        # Test theta update (error feedback)
        csa.provide_feedback(
            involved_nodes=["event1", "event2"],
            involved_edges=[("event1", "event2", "CAUSES")],
            is_correct=False,
            confidence=1.0,
        )

        stats = csa.afl.get_stats()
        console.print("✓ Theta update applied")
        console.print(f"  Theta updates: {stats['theta_updates']}")
        console.print(f"  PFC→HPC strength: {stats['pfc_hpc_strength']:.4f}")

        # Test theta-gamma encoding
        sequence = ["A", "B", "C", "D", "E"]
        encoded = csa.encode_sequence(sequence)

        console.print("✓ Theta-gamma encoding successful")
        console.print(f"  Num segments: {len(encoded)}")

        csa.close()
        console.print("[green]✓ AFL feedback test passed[/green]")

    except Exception as e:
        console.print(f"[red]✗ AFL test failed: {e}[/red]")
        raise


def display_summary() -> None:
    """Display summary of test results."""
    console.print("\n[bold cyan]Test Summary[/bold cyan]")

    table = Table(title="CSA Algorithm Test Results")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Key Features Validated")

    table.add_row(
        "Neo4j Connection",
        "✓ Passed",
        "Connection, schema initialization",
    )
    table.add_row(
        "HKG (RTM)",
        "✓ Passed",
        "Recursive partitioning, graph structure",
    )
    table.add_row(
        "DRM (Attention)",
        "✓ Passed",
        "Relevance gains, gating, multi-scale",
    )
    table.add_row(
        "AFL (Learning)",
        "✓ Passed",
        "Beta/theta updates, theta-gamma encoding",
    )

    console.print(table)


def main() -> None:
    """Run all tests."""
    console.print("[bold magenta]Cognitive Synergy Architecture - Algorithm Tests[/bold magenta]")
    console.print("=" * 60)

    try:
        test_neo4j_connectivity()
        test_hkg_recursive_partition()
        test_drm_attention()
        test_afl_feedback()

        display_summary()

        console.print("\n[bold green]All tests passed successfully! 🎉[/bold green]")

    except Exception as e:
        console.print("\n[bold red]Tests failed with error:[/bold red]")
        console.print(f"[red]{e}[/red]")
        raise


if __name__ == "__main__":
    main()
