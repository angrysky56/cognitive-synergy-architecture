"""
Hierarchical Knowledge Graph (HKG) implementation.

Based on Random Tree Memory (RTM) principles for structured memory representation.
Uses Neo4j for persistent graph storage with typed nodes and weighted edges.
Implements ensemble-based reasoning with Random-Inference Trees.

References:
- RTM Framework: Recursive partitioning with branching factor K
- Ensemble Reasoning: Random sub-graphs (bagging) + feature randomness
- Top-down Recall: Maximum depth D constraint
"""

import random
from dataclasses import dataclass
from typing import Any

from neo4j import Driver, GraphDatabase, ManagedTransaction


@dataclass
class RTMConfig:
    """Configuration for Random Tree Memory framework."""

    branching_factor: int = 4  # Maximum K for working memory chunking
    max_recall_depth: int = 3  # Maximum D for top-down traversal
    n_estimators: int = 500  # Number of Random-Inference Trees
    max_features: float = 0.5  # Proportion of edge types per tree
    min_weight: float = 0.5  # Minimum edge weight threshold


class HierarchicalKnowledgeGraph:
    """
    Persistent knowledge graph with hierarchical structure.

    Implements the RTM framework with:
    - Typed nodes: Character, Object, Location, Event, KeyPoint
    - Typed edges: CAUSES, PRECEDES, ALLY_OF, PARENT_OF, etc.
    - Weighted relationships for confidence/importance
    - Hierarchical organization (root → gist → details)
    - Ensemble reasoning via Random-Inference Trees
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        config: RTMConfig | None = None,
    ) -> None:
        """
        Initialize connection to Neo4j database.

        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
            config: RTM configuration parameters
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.config = config or RTMConfig()
        self._initialize_schema()
        self._edge_types: list[str] = []
        self._refresh_edge_types()

    def _initialize_schema(self) -> None:
        """Create constraints and indexes for optimal graph structure."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Object) REQUIRE o.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:KeyPoint) REQUIRE k.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n)-[r]-() ON (r.weight)")

    def _refresh_edge_types(self) -> None:
        """Refresh the cached list of relationship types in the graph."""
        with self.driver.session() as session:
            result = session.run("CALL db.relationshipTypes()")
            self._edge_types = [record[0] for record in result]

    def add_node(
        self,
        node_type: str,
        node_id: str,
        properties: dict[str, Any],
    ) -> None:
        """
        Add a node to the knowledge graph.

        Args:
            node_type: Type of node (Character, Object, Location, Event, KeyPoint)
            node_id: Unique identifier
            properties: Node properties (name, description, size, etc.)
        """
        def _create_node_tx(tx: ManagedTransaction) -> None:
            from typing import LiteralString, cast
            query = cast(LiteralString, f"""
            MERGE (n:{node_type} {{id: $node_id}})
            SET n += $properties
            RETURN n
            """)
            tx.run(query, node_id=node_id, properties=properties)

        with self.driver.session() as session:
            session.execute_write(_create_node_tx)

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a weighted relationship between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type (CAUSES, PRECEDES, PARENT_OF, etc.)
            weight: Relationship weight/confidence
            properties: Additional relationship properties
        """
        props = properties or {}
        props["weight"] = weight
        def _create_rel_tx(tx: ManagedTransaction) -> None:
            from typing import LiteralString, cast
            query = cast(LiteralString, f"""
            MATCH (s {{id: $source_id}})
            MATCH (t {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $properties
            RETURN r
            """)
            tx.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=props,
            )

        with self.driver.session() as session:
            session.execute_write(_create_rel_tx)

        # Refresh edge types if this is a new relationship type
        if rel_type not in self._edge_types:
            self._refresh_edge_types()

    def recursive_partition(
        self,
        narrative_clauses: list[str],
        parent_id: str | None = None,
        depth: int = 0,
    ) -> str:
        """
        Recursively partition narrative into hierarchical structure.

        Implements RTM's recursive partitioning algorithm with branching factor K.
        Uses random splitting to divide parent segments into child segments.

        Args:
            narrative_clauses: List of narrative clauses to partition
            parent_id: ID of parent KeyPoint node
            depth: Current depth in the hierarchy

        Returns:
            ID of the root KeyPoint node
        """
        n = len(narrative_clauses)
        if n == 0:
            raise ValueError("Cannot partition empty narrative")

        # Create KeyPoint node for this segment
        node_id = f"keypoint_{depth}_{random.randint(0, 999999)}"
        self.add_node(
            "KeyPoint",
            node_id,
            {
                "depth": depth,
                "size": n,
                "text": " ".join(narrative_clauses[:50]),  # Summary
            },
        )

        # Link to parent if exists
        if parent_id:
            self.add_relationship(
                parent_id,
                node_id,
                "PARENT_OF",
                weight=1.0,
            )

        # Base case: single clause or max depth reached
        if n == 1 or depth >= self.config.max_recall_depth:
            return node_id

        # Recursive partitioning with random splitting
        k = min(random.randint(2, self.config.branching_factor), n)

        # Randomly place k-1 boundaries to create k child segments
        boundaries = sorted(random.sample(range(1, n), k - 1))
        boundaries = [0] + boundaries + [n]

        # Recursively partition each child segment
        for i in range(k):
            start, end = boundaries[i], boundaries[i + 1]
            child_clauses = narrative_clauses[start:end]
            if child_clauses:
                self.recursive_partition(child_clauses, node_id, depth + 1)

        return node_id

    def _generate_random_subgraph(self, tx: ManagedTransaction) -> tuple[list[str], list[str]]:
        """
        Generate a random sub-graph for bagging (bootstrap aggregating).

        Returns:
            Tuple of (sampled_node_ids, sampled_edge_types)
        """
        # Sample nodes with replacement
        node_result = tx.run("MATCH (n) RETURN n.id as id")
        all_node_ids = [record["id"] for record in node_result]

        if not all_node_ids:
            return [], []

        # Bootstrap sampling
        sampled_nodes = random.choices(all_node_ids, k=len(all_node_ids))

        # Sample edge types (feature randomness)
        if self._edge_types:
            n_features = max(1, int(len(self._edge_types) * self.config.max_features))
            sampled_edges = random.sample(self._edge_types, n_features)
        else:
            sampled_edges = []

        return sampled_nodes, sampled_edges

    def _weighted_graph_search(
        self,
        tx: ManagedTransaction,
        start_id: str,
        goal_pattern: str,
        allowed_edges: list[str],
        max_depth: int,
    ) -> list[dict[str, Any]]:
        """
        Perform weighted A* search constrained by allowed edge types.

        Args:
            tx: Transaction object
            start_id: Starting node ID
            goal_pattern: Cypher pattern for goal nodes
            allowed_edges: List of allowed relationship types
            max_depth: Maximum traversal depth

        Returns:
            List of paths with weights
        """
        # Build Cypher query with edge type constraints
        from typing import LiteralString, cast
        edge_filter = "|".join(allowed_edges) if allowed_edges else "*"

        query = cast(LiteralString, f"""
        MATCH path = (start {{id: $start_id}})-[:{edge_filter}*1..{max_depth}]-(goal)
        WHERE {goal_pattern}
        WITH path, relationships(path) as rels
        WITH path, reduce(weight = 0, r IN rels | weight + r.weight) as total_weight
        WHERE all(r IN rels WHERE r.weight >= $min_weight)
        RETURN path, total_weight
        ORDER BY total_weight DESC
        LIMIT 10
        """)

        result = tx.run(
            query,
            start_id=start_id,
            min_weight=self.config.min_weight,
        )

        paths = []
        for record in result:
            path = record["path"]
            paths.append({
                "nodes": [node["id"] for node in path.nodes],
                "weight": record["total_weight"],
            })

        return paths

    def random_inference_trees(
        self,
        query_node_id: str,
        goal_pattern: str = "goal:KeyPoint",
    ) -> list[dict[str, Any]]:
        """
        Ensemble reasoning using Random-Inference Trees.

        Implements bagging + feature randomness for robust reasoning:
        1. Generate n_estimators random sub-graphs
        2. Constrain each traversal to random edge subset
        3. Aggregate results via majority vote

        Args:
            query_node_id: Starting node for reasoning
            goal_pattern: Cypher pattern for goal nodes

        Returns:
            List of ranked hypotheses with confidence scores
        """
        all_paths: dict[str, int] = {}  # path_signature -> count

        with self.driver.session() as session:
            for _ in range(self.config.n_estimators):
                # Generate random sub-graph and allowed edges
                def _search_tree(tx: ManagedTransaction) -> list[dict[str, Any]]:
                    _, sampled_edges = self._generate_random_subgraph(tx)

                    if not sampled_edges:
                        return []

                    return self._weighted_graph_search(
                        tx,
                        query_node_id,
                        goal_pattern,
                        sampled_edges,
                        self.config.max_recall_depth,
                    )

                paths = session.execute_read(_search_tree)

                # Count path occurrences
                for path in paths:
                    signature = ",".join(path["nodes"])
                    all_paths[signature] = all_paths.get(signature, 0) + 1

        # Aggregate and rank by vote count
        ranked_hypotheses = [
            {
                "path": path.split(","),
                "confidence": count / self.config.n_estimators,
            }
            for path, count in sorted(
                all_paths.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return ranked_hypotheses[:10]  # Top 10 hypotheses

    def get_subgraph(
        self,
        seed_nodes: list[str],
        max_depth: int = 3,
        min_weight: float = 0.5,
    ) -> dict[str, Any]:
        """
        Retrieve a subgraph around seed nodes with top-down recall constraint.

        Args:
            seed_nodes: Starting node IDs
            max_depth: Maximum traversal depth (D parameter)
            min_weight: Minimum relationship weight to include
        """
        def _get_subgraph_tx(tx: ManagedTransaction) -> dict[str, Any]:
            from typing import LiteralString, cast
            query = cast(LiteralString, f"""
            MATCH path = (start)-[*1..{max_depth}]-(connected)
            WHERE start.id IN $seed_nodes
              AND all(r IN relationships(path) WHERE r.weight >= $min_weight)
            WITH path
            LIMIT 1000
            RETURN
                [n IN nodes(path) | {{id: n.id, labels: labels(n), properties: properties(n)}}] as nodes,
                [r IN relationships(path) | {{type: type(r), weight: r.weight}}] as edges
            """)

            result = tx.run(
                query,
                seed_nodes=seed_nodes,
                min_weight=min_weight,
            )

            nodes_set: dict[str, Any] = {}
            edges_list: list[dict[str, Any]] = []

            for record in result:
                for node in record["nodes"]:
                    nodes_set[node["id"]] = node
                edges_list.extend(record["edges"])

            return {
                "nodes": list(nodes_set.values()),
                "edges": edges_list,
            }

        with self.driver.session() as session:
            return session.execute_read(_get_subgraph_tx)

    def get_node_count(self) -> int:
        """Get total number of nodes in graph."""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            return record["count"] if record else 0

    def get_edge_count(self) -> int:
        """Get total number of relationships in graph."""
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            return record["count"] if record else 0

    def close(self) -> None:
        """Close database connection."""
        self.driver.close()
