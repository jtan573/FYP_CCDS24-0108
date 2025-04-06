# ---------------------
# Basic Testing
# ---------------------
from models.llama_handler import get_llama_llm
from modules.chain_functions import evaluate_kg_ans, extract_concepts, generate_cypher, resolve_synonym, synthesize_answer
import json
from modules.neo4j_client import run_cypher_query

# Load LLM
chatmodel = get_llama_llm()

# query1 = "Do patients with alcohol use disorder usually have liver condition?"
# queries_simple = [
#     "Find all disorders with the term 'diabetes' in their description.",
#     "Get all findings related to smoking.",
#     "How many patients have asthma?",
#     "Find all disorders with an associated morphology of inflammation.",
#     "Get procedures that use a catheter."
# ]

# queries_intermediate = [
#     "List all findings that have an interpretation of \"abnormal\".",
#     "Retrieve drugs that contain \"paracetamol\" as an active ingredient.",
#     "Which patients have more than 150 minutes of moderate exercise weekly?",
#     "Find disorders due to bacterial organisms.",
#     "Find procedures with both a procedure site and a device used."
# ]

query_test = ["Are people with sedentary lifestyles more likely to have high cholesterol?"]

MODEL_NAME = "llama"
for query in query_test:
    print("\n -------------------- NEXT QUERY -------------------- \n")
    concepts = extract_concepts(query, 'llama', chatmodel)
    resolved_dict = resolve_synonym(concepts)
    cypher = generate_cypher(query, resolved_dict, MODEL_NAME, chatmodel)

concepts = extract_concepts(query, 'llama', chatmodel)

# temp_content = """{"asthma": "Disorder, Finding", "sedentary lifestyle": "Finding, Situation"}"""
# concepts = json.loads(temp_content)

# resolved_dict = resolve_synonym(concepts)

# cypher = generate_cypher(query, resolved_dict, MODEL_NAME, chatmodel)

# cypher_result = run_cypher_query(cypher)

# supporting_info = evaluate_kg_ans(query, cypher_result, chatmodel)

# result_dict = {
#     "Result from Knowledge Graph": "",
#     "Supporting Information": "supporting_info"
# }
# result = synthesize_answer(query_test, result_dict, "llama", chatmodel)