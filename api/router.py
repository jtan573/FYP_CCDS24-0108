import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.llama_handler import get_llama_llm
from modules.chain_functions import *
from pydantic import BaseModel
from modules.neo4j_client import run_cypher_query, validate_cypher

router = APIRouter()

class QueryRequest(BaseModel):
    user_query: str
    llm_provider: str = "openai"  # or llama, etc.

class QueryResponse(BaseModel):
    result: str

@router.post("/query")
def stream_query(payload: QueryRequest):

    llm = get_llama_llm()

    def stream_steps():
        try:
            yield "Extracting Concepts...\n"
            concepts = extract_concepts(payload.user_query, payload.llm_provider, llm)
            if not concepts:
                yield "\n❌ Could not resolve any valid medical concepts.\n"
                return
            yield f"\n{json.dumps(concepts, indent=2)}\n"

            yield "\nResolving Synonyms...\n"
            resolved_fsns = resolve_synonym(concepts)
            yield f"\n{json.dumps(resolved_fsns, indent=2)}\n"

            yield "\nGenerating Cypher Query...\n"
            cypher = generate_cypher(payload.user_query, resolved_fsns, payload.llm_provider, llm)
            is_valid_cypher = validate_cypher(cypher)
            if not (is_valid_cypher):
                cypher = refine_cypher(cypher, payload.llm_provider, llm)
                is_valid_cypher = validate_cypher(cypher)
                
            yield f"\n{cypher}\n"
            flag=True
            if (is_valid_cypher):
                yield "\nExecuting Cypher Query...\n"
                cypher_result = run_cypher_query(cypher)
                yield f"\n{json.dumps(cypher_result, indent=2)}\n"
            else:
                flag = False
                cypher_result = ""
                yield "\nCypher Query is invalid, attempt to find information from backup database...\n"

            if (flag): 
                yield "\nEvaluating Answer from KG...\n"
                docs = evaluate_kg_ans(payload.user_query, cypher_result, llm)
                if docs:
                    yield f"\n{json.dumps(docs, indent=2)}\n"
                else:
                    message = "No additional information required."
                    yield f"\n{json.dumps(message, indent=2)}\n"
            else:
                yield "\nSearching chroma database for relevant information...\n"
                docs = get_supporting_info(payload.user_query)
                yield f"\n{json.dumps(docs, indent=2)}\n"

            yield "\nSynthesizing Final Answer...\n"
            final_answer = synthesize_answer(payload.user_query, cypher_result, docs, payload.llm_provider, llm)
            yield f"\n{final_answer}\n"

        except Exception as e:
            yield f"🚨 **Error:** `{str(e)}`\n"

    return StreamingResponse(stream_steps(), media_type="text/plain")

@router.post("/query-test")
def stream_query(payload: QueryRequest):

    llm = get_llama_llm()

    def stream_steps():
        try:
            yield "🔍 Extracting Concepts...\n"
            concepts = {"asthma": "Disorder"}
            if not concepts:
                yield "\n❌ Could not resolve any valid medical concepts.\n"
                return
            yield f"\n{json.dumps(concepts, indent=4)}\n"

            yield "\n🔁 Resolving Synonyms...\n"
            resolved_fsns = {"asthma": "asthma (disorder)"}
            yield f"\n{json.dumps(resolved_fsns, indent=4)}\n"

            yield "\n🧠 Generating Cypher Query...\n"
            cypher = "MATCH (patient:Patient)-[:HAS_DISORDER]->(disorder:Disorder) WHERE toLower(disorder.FSN) CONTAINS 'asthma'"
            yield f"\n{cypher}\n"

            yield "\n🔄 Executing Cypher Query...\n"
            cypher_result = "[{\"total_patients\": 1946}]"
            yield f"\n{json.dumps(cypher_result, indent=4)}\n"

            yield "\n📎 Evaluating Answer from KG...\n"
            docs = None
            if docs:
                yield f"\n{json.dumps(docs, indent=4)}\n"
            else:
                message = "No additional information required."
                yield f"\n{json.dumps(message, indent=4)}\n"
            
            yield "\n🧩 Synthesizing Final Answer...\n"
            final_answer = "There are 1946 patients with asthma."
            yield f"\n{final_answer}\n"

        except Exception as e:
            yield f"🚨 **Error:** `{str(e)}`\n"

    return StreamingResponse(stream_steps(), media_type="text/plain")