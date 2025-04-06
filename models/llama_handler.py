from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
import os
from dotenv import load_dotenv

def get_llama_llm():
    # Load environment variables for authentication
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN_2")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")
    login(token=HF_TOKEN)
    print("Successfully authenticated with Hugging Face Hub!")

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    llm = HuggingFaceEndpoint(
        repo_id=model_name, 
        max_new_tokens=256, 
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation"
    )
    print("LLM Endpoint connected!")

    chat_model = ChatHuggingFace(llm=llm)
    print("Chat model initialised.")

    return chat_model

# ---------------------
# LLAMA Prompt Templates
# ---------------------
CONCEPT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            """You are a medical language model. Your task is to extract only the specific and clinically meaningful medical concepts that are explicitly mentioned in the user query.
            """
        )
    ),
    (
        "user",
        (
            """Each extracted concept must be mapped to three most representative clinical category from the following list:
            Catgories:
                Description - Textual representation of a concept (term or synonym).
                QualifierValue - Modifiers for other concepts (e.g. good, granular, high frequency.
                Disorder - Diseases, illnesses, and pathological conditions.
                Finding - Clinical observations not necessarily diseases (e.g. fever, lifestyle, alcohol use).
                Procedure - Medical actions or interventions (e.g. surgery, imaging).
                RegimeorTherapy - Ongoing care plans or therapies (e.g. chemotherapy regimen).
                Situation - Clinical contexts or scenarios (e.g. risk of stroke, past history).
                ObservableEntity - Measurable properties (e.g. blood pressure, core body temperature).
                MorphologicAbnormality - Structural tissue changes (e.g. tumor, fibrosis).
                BodyStructure - Anatomical parts (e.g. heart, femur).
                CellStructure - Microscopic parts of cells (e.g. mitochondria).
                Cell - Individual biological cells (e.g. red blood cell).
                Event - Healthcare-related occurrences (e.g. heart attack, trauma).
                PhysicalObject - Tangible items used in healthcare (e.g. stethoscope, implant).
                PhysicalForce - Forces affecting the body (e.g. radiation, gravity).
                Substance - Chemical constituents of medicinal and non-medicinal products (e.g. paracetamol, endorphin).
                Product - Products identified to contain a particular substance (e.g. product containing lipotropic agent).
                ClinicalDrug - Specific drug formulation with dose and route (e.g. 500 mg oral ibuprofen).
                DoseForm - Drug physical form (e.g. tablet, injection).
                MedicinalProduct - Packaged drug products (e.g. insulin pen).
                AssessmentScale - Scales for rating health states (e.g. pain scale).
                TumorStaging - Specific cancer staging systems (e.g. TNM stage II).
                Environment - Environmental factors (e.g. air pollution, workplace).
                Organism - Living entities (e.g. bacteria, human).
                Person - Individual people (e.g. patient, caregiver).
                Specimen - Samples taken for analysis (e.g. blood sample, biopsy).

            Requirements:
            Extract only specific medical concepts (e.g., "chest pain", "type 2 diabetes", "chemotherapy").
            Do not extract vague or general words (e.g., "disorder", "symptom", "finding", "problem").
            Extract only concepts verbatim from the query.
            Concepts should be clinically relevant terms—not question words or general phrasing.

            Output format: Return only a JSON dictionary where:
            Each key is a unique concept string.
            Each value is the appropriate category from the list above.

            Examples: 
            User Query: "What are the symptoms of a heart attack?"
            Output: {{"heart attack": "Disorder, Situation, Finding"}}
            User Query: "Can alcohol use lead to liver problems?"
            Output: {{"alcohol use": "Finding, ObservableEntity, Description", "liver problems": "Disorder, MorphologicAbnormality, Finding"}}

            Only return the dictionary. Do not explain anything.
            Now use this format for the following user query:
            "{user_query}"
            """
        )
    )
])

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
                """Generate a Cypher query using the schema, user question, and resolved concepts below.

                Instructions:
                - Always use FSN for property matching.
                - Match string properties (e.g. FSN) using toLower() and CONTAINS for partial matches.
                - If relevant, the resolved concepts are used to replace their original terms.
                - Focus only on key medical terms from the resolved concepts (e.g. 'lung', 'alcohol') to keep the query general.
                - Only return the Cypher query. Do not include any explanations.

                Examples:

                User: Which procedures target the lungs?  
                FSNs: {{'lungs': 'lungs (body structure)'}}  
                Cypher: MATCH (proc:Procedure)-[:HAS_ROLE_GROUP]->(rg:RoleGroup)-[:PROCEDURE_SITE]->(b:BodyStructure)
                        WHERE toLower(b.FSN) CONTAINS 'lung'  
                        RETURN DISTINCT proc AS procedure

                User: What disorders are associated with cough?  
                FSNs: {{'cough': 'cough (disorder)'}} 
                Cypher: MATCH (d:Disorder)-[:ISA]->(f:Finding)  
                        WHERE toLower(f.FSN) = 'cough (finding)'  
                        RETURN d

                Schema: {schema}  
                User: {query}  
                Resolved concepts: {fsn}

                Cypher:

                """
            ),
        ),
    ])

REFINE_CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a neo4j cypher language expert.",
        ),
        (
            "user",
            (
                "Correct the syntax errors in the following cypher query."
                "Only return the Cypher query. Do not include any explanations."
                "Query: {query}"
                "Answer:"
            ),
        ),
    ]
)

SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical language model.",
        ),
        (
            "user",
            (
                "Based on the context given, generate a concise answer to the patient's question."
                "Do not need to explain too much. Just give a straightforward answer."
                "Context: {context}"
                "Question: {query}"
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