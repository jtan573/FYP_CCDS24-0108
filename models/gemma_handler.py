## Not used; For experimentation purposes
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
import os
from dotenv import load_dotenv

def get_gemma_llm():
    # Load environment variables for authentication
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")
    login(token=HF_TOKEN)
    print("Successfully authenticated with Hugging Face Hub!")

    model_name = "google/gemma-1.1-7b-it"

    llm = HuggingFaceEndpoint(
        repo_id=model_name, 
        max_new_tokens=256, 
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN,
        # task="text-generation"
    )
    print("LLM Endpoint connected!")

    chat_model = ChatHuggingFace(llm=llm)
    print("Chat model initialised.")

    return chat_model

# ---------------------
# LLAMA Prompt Templates
# ---------------------
CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """You generate Cypher queries for a Neo4j medical knowledge graph."""
            )
        ),
        (
            "user",
            (
                """Generate a Cypher query using the schema, user question, and resolved FSNs below.

                Instructions:
                - Match string properties (e.g. FSN, description) using toLower() and CONTAINS for partial matches.
                - Focus only on key medical terms from the resolved FSNs (e.g. 'lung', 'alcohol') to keep the query general.
                - Only return the Cypher query. Do not include any explanations.

                Examples:

                User: Which procedures target the lungs?  
                FSNs: {{'lungs': 'lungs (body structure)'}}  
                Cypher: MATCH (proc:Procedure)-[:HAS_ROLE_GROUP]->(rg:RoleGroup)  
                        MATCH (rg)-[:PROCEDURE_SITE]->(b:BodyStructure)  
                        WHERE toLower(b.FSN) CONTAINS 'lung'  
                        RETURN DISTINCT proc AS procedure

                User: What disorders are associated with cough?  
                FSNs: {{'cough': 'cough (disorder)'}} 
                Cypher: MATCH (d:Disorder)-[:ISA]->(f:Finding)  
                        WHERE toLower(f.FSN) = 'cough (finding)'  
                        RETURN d

                Schema: {schema}  
                User: {query}  
                FSNs: {fsn}

                Cypher:

                """
            ),
        ),
    ])

SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical language model.",
        ),
        (
            "user",
            (
                "Based on the following extracted information, provide a concise and accurate answer."
                "Context: {context}"
                "Answer:"
            ),
        ),
    ])

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical language model.",
        ),
        (
            "user",
            (
                """You are provided with two types of information: (1) a result from a medical knowledge graph,
                and (2) clinical supporting information. Use both to generate a concise answer to the patient's question.
                Knowledge Graph Result: {context}
                Question: {query}
                Answer:
                """
            ),
        ),
    ])