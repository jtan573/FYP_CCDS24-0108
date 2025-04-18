## Not used; For experimentation purposes
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

def get_text2cypher_llm(): 
    # Load environment variables for authentication
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")
    login(token=HF_TOKEN)
    print("Successfully authenticated with Hugging Face Hub!")

    model_name = "tomasonjo/text2cypher-demo-16bit"  # or your chosen model
    llm = HuggingFaceEndpoint(
        repo_id=model_name, 
        max_length=1024, 
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation"
    )
    print("LLM Endpoint connected!")

    chat_model = ChatHuggingFace(llm=llm)
    print("Chat model initialised.")

    return chat_model

# ---------------------
# Setting up 'chat' endpoint
# ---------------------
cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        (
            "human",
            (
                "Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question: "
                "\n{schema} \nQuestion: {question} \nCypher query:"
            ),
        ),
    ]
)
