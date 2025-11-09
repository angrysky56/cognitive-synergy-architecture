"""
Basic CSA usage example.

Demonstrates initializing the architecture and basic operations.
"""

from csa import CognitiveArchitecture


def main() -> None:
    """Run basic CSA example."""
    print("Initializing Cognitive Synergy Architecture...")

    # Initialize CSA (requires Neo4j running locally)
    csa = CognitiveArchitecture(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )

    print("CSA initialized successfully!")

    # Get initial stats
    stats = csa.get_stats()
    print("\nInitial Statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")

    # TODO: Add example text ingestion
    # TODO: Add example query
    # TODO: Add example feedback

    print("\nCleaning up...")
    csa.close()
    print("Done!")


if __name__ == "__main__":
    main()
