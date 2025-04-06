## Not used; For experimentation purposes
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
import os
from dotenv import load_dotenv

def get_mistral_llm():
    # Load environment variables for authentication
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN_2")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")
    login(token=HF_TOKEN)
    print("Successfully authenticated with Hugging Face Hub!")

    model_name = "HuggingFaceH4/zephyr-7b-beta"

    llm = HuggingFaceEndpoint(
        repo_id=model_name, 
        max_new_tokens=128, 
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation"
    )
    print("LLM Endpoint connected!")

    chat_model = ChatHuggingFace(llm=llm)
    print("Chat model initialised.")

    return chat_model

# ---------------------
# Mistral Prompt Templates
# ---------------------
CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """You are a Cypher query generator for a Neo4j medical knowledge graph. Only return the Cypher query. Do not include any explanations."""
            )
        ),
        (
            "user",
            (
                """Generate a Cypher query using the schema, user question, and resolved FSNs below.

                Instructions:
                - Match string properties (e.g. FSN, description) using toLower() and CONTAINS for partial matches.
                - Focus only on key medical terms from the resolved FSNs (e.g. 'lung', 'alcohol') to keep the query general.

                Examples:
                User: Which procedures target the lungs?  
                FSNs: {{'lungs': 'lungs (body structure)'}}  
                Cypher: MATCH (proc:Procedure)-[:HAS_ROLE_GROUP]->(rg:RoleGroup)  
                        MATCH (rg)-[:PROCEDURE_SITE]->(b:BodyStructure)  
                        WHERE toLower(b.FSN) CONTAINS 'lung'  
                        RETURN DISTINCT proc AS procedure

                User: What disorders are associated with cough?  
                FSNs: {{'cough': 'cough (disorder)'}}  
                Cypher: MATCH (d:Disorder)-[:ISA]->(f:Finding) WHERE toLower(f.FSN) = 'cough (finding)' RETURN d

                Schema: {schema}  
                User: {query}  
                FSNs: {fsn}
                
                
                Cypher:

                """
            ),
        ),
    ])