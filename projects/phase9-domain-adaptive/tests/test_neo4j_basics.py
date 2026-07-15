from __future__ import annotations

from neo4j_basics import (
    EXAMPLE_QUERIES,
    SAMPLE_GRAPH,
    GraphNode,
    GraphRelationship,
    ServiceGraph,
    build_create_cypher,
    query_via_langchain,
)


def test_graph_node_model() -> None:
    node = GraphNode(name="svc", labels=["Service"])
    assert node.name == "svc"
    assert node.labels == ["Service"]


def test_graph_relationship_model() -> None:
    rel = GraphRelationship(from_node="a", to_node="b", rel_type="DEPENDS_ON")
    assert rel.from_node == "a"
    assert rel.to_node == "b"
    assert rel.rel_type == "DEPENDS_ON"


def test_service_graph_model() -> None:
    graph = ServiceGraph(nodes=[], relationships=[])
    assert graph.nodes == []
    assert graph.relationships == []


def test_sample_graph_has_four_nodes() -> None:
    assert len(SAMPLE_GRAPH.nodes) == 4


def test_sample_graph_has_relationships() -> None:
    assert len(SAMPLE_GRAPH.relationships) > 0


def test_sample_graph_auth_service_present() -> None:
    assert any(n.name == "auth-service" for n in SAMPLE_GRAPH.nodes)


def test_example_queries_count() -> None:
    assert len(EXAMPLE_QUERIES) >= 5


def test_example_queries_are_cypher() -> None:
    assert all("MATCH" in q or "CREATE" in q for q in EXAMPLE_QUERIES)


def test_build_create_cypher_non_empty() -> None:
    stmts = build_create_cypher(SAMPLE_GRAPH)
    assert isinstance(stmts, list)
    assert len(stmts) > 0


def test_build_create_cypher_contains_create() -> None:
    stmts = build_create_cypher(SAMPLE_GRAPH)
    assert all("CREATE" in s for s in stmts[:4])


def test_build_create_cypher_count() -> None:
    stmts = build_create_cypher(SAMPLE_GRAPH)
    assert len(stmts) == len(SAMPLE_GRAPH.nodes) + len(SAMPLE_GRAPH.relationships)


def test_query_via_langchain_returns_string() -> None:
    result = query_via_langchain("test")
    assert isinstance(result, str)


def test_query_via_langchain_handles_missing_deps() -> None:
    result = query_via_langchain("test")
    assert "[" in result
