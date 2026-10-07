"""Neo4j graph modeling and Cypher queries. Requires: podman-compose up -d neo4j"""

from __future__ import annotations

import json

from pydantic import BaseModel


class GraphNode(BaseModel):
    name: str
    labels: list[str]
    properties: dict = {}


class GraphRelationship(BaseModel):
    from_node: str
    to_node: str
    rel_type: str
    properties: dict = {}


class ServiceGraph(BaseModel):
    nodes: list[GraphNode]
    relationships: list[GraphRelationship]


SAMPLE_GRAPH: ServiceGraph = ServiceGraph(
    nodes=[
        GraphNode(
            name="auth-service",
            labels=["Service"],
            properties={"language": "Python", "team": "platform"},
        ),
        GraphNode(
            name="user-service", labels=["Service"], properties={"language": "Go", "team": "core"}
        ),
        GraphNode(
            name="payment-service",
            labels=["Service"],
            properties={"language": "Java", "team": "billing"},
        ),
        GraphNode(
            name="notification-service",
            labels=["Service"],
            properties={"language": "Python", "team": "platform"},
        ),
    ],
    relationships=[
        GraphRelationship(from_node="user-service", to_node="auth-service", rel_type="DEPENDS_ON"),
        GraphRelationship(
            from_node="payment-service", to_node="auth-service", rel_type="DEPENDS_ON"
        ),
        GraphRelationship(
            from_node="payment-service", to_node="user-service", rel_type="DEPENDS_ON"
        ),
        GraphRelationship(
            from_node="notification-service",
            to_node="user-service",
            rel_type="DEPENDS_ON",
        ),
    ],
)


def build_create_cypher(graph: ServiceGraph) -> list[str]:
    stmts: list[str] = []
    for node in graph.nodes:
        label_str = ":".join(node.labels)
        props = json.dumps(node.properties)
        stmts.append(f"CREATE (:{label_str} {{name: '{node.name}', {props[1:-1]}}})")
    for rel in graph.relationships:
        stmts.append(
            f"MATCH (a {{name: '{rel.from_node}'}}), (b {{name: '{rel.to_node}'}})"
            f" CREATE (a)-[:{rel.rel_type}]->(b)"
        )
    return stmts


EXAMPLE_QUERIES: list[str] = [
    "MATCH (s:Service) RETURN s.name, s.team ORDER BY s.name",
    "MATCH (a:Service)-[:DEPENDS_ON]->(b:Service) RETURN a.name AS caller, b.name AS dependency",
    "MATCH (s:Service)-[:DEPENDS_ON*2]->(dep:Service) RETURN s.name, dep.name AS transitive_dep",
    (
        "MATCH (s:Service)-[:DEPENDS_ON]->(dep:Service)"
        " RETURN dep.name, count(*) AS dependents ORDER BY dependents DESC"
    ),
    (
        "MATCH path=(s:Service)-[:DEPENDS_ON*1..3]->(auth:Service {name: 'auth-service'})"
        " RETURN s.name, length(path) AS hops ORDER BY hops"
    ),
]


def query_via_langchain(question: str, neo4j_url: str = "bolt://localhost:7687") -> str:
    try:
        from langchain_community.chains import GraphCypherQAChain  # noqa: F401
        from langchain_community.graphs import Neo4jGraph  # noqa: F401

        return "[LangChain Neo4j: requires podman-compose up neo4j and langchain-community]"
    except ImportError:
        return "[langchain-community not installed for this phase]"


if __name__ == "__main__":
    print(SAMPLE_GRAPH.model_dump_json(indent=2))
    print("\nExample Cypher queries:")
    for q in EXAMPLE_QUERIES:
        print(f"  {q}")
