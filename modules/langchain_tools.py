from langchain_core.tools import tool
from .medquad_chroma_client import get_supporting_info

@tool
def search_docs(query: str):
    """Use this tool only when the KG answer cannot adequately address the user's question at a basic or surface level — in-depth explanations are not required."""
    most_similar_doc = get_supporting_info(query)
    return most_similar_doc

tools = [search_docs]
