from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)
print("Connected to Neo4j!")

# -------------------------------
# Helper function to run CYPHER query
# -------------------------------
def run_cypher_query(query: str):
    output = graph.query(query)
    if (len(output) > 20):
        output = output[:20]
    # print("KG QUERY OUTPUT:", output)

    return output

# --- Example outputs ---
# Counting queries: [{'COUNT(p)': 1946}]

# -------------------------------
# Helper function to validate CYPHER query
# -------------------------------
def validate_cypher(query: str) -> bool:
    try:
        test_query = f"{query.strip()} LIMIT 1"
        graph.query(test_query)
        return True
    except Exception as e:
        return False
