from modules.medquad_chroma_client import get_supporting_info
from .snomed_chroma_client import resolve_fsn
from .langchain_tools import search_docs, tools
import importlib, json

def extract_concepts(user_query, llm_choice, llm):
    # Import the correct prompt template
    module = importlib.import_module(f"models.{llm_choice}_handler")
    CONCEPT_PROMPT = module.CONCEPT_PROMPT

    # Create LLM Chain
    chain = CONCEPT_PROMPT | llm
    print("Chain created!")
    output = chain.invoke({"user_query": user_query})

    # Print outputs for debugging
    print("Extract concepts text output:", output.content)
    
    concept_dict = json.loads(output.content)

    return concept_dict

def resolve_synonym(concept_dict: dict):
    resolved_dict = {}
    for concept, categories in concept_dict.items():
        resolved_fsn = resolve_fsn(concept, categories)
        resolved_dict[concept] = resolved_fsn
    print(resolved_dict)

    return resolved_dict

def generate_cypher(user_query, fsns, llm_choice, llm):
    # Read graph schema from file
    # with open("../data/graph_schema.txt", "r") as file:
    with open("./data/summarised_graph_schema.txt", "r") as file:
        graph_schema = file.read()
    
    # Import the correct prompt template
    module = importlib.import_module(f"models.{llm_choice}_handler")
    CYPHER_PROMPT = module.CYPHER_PROMPT

    # Create LLM Chain
    chain = CYPHER_PROMPT | llm
    output = chain.invoke({'schema':graph_schema, 'query':user_query, 'fsn':fsns})
    
    # Print outputs for debugging
    print("\nInitial output:\n", output)
    print("\nGenerate cypher text output:\n", output.content)
    return output.content

def refine_cypher(cypher, llm_choice, llm):
    # Import the correct prompt template
    module = importlib.import_module(f"models.{llm_choice}_handler")
    REFINE_CYPHER_PROMPT = module.REFINE_CYPHER_PROMPT

    # Create LLM Chain
    chain = REFINE_CYPHER_PROMPT | llm
    output = chain.invoke({'query':cypher})
    return output.content

def search_medquad(query):
    most_similar_doc = get_supporting_info(query)
    return most_similar_doc


def evaluate_kg_ans(query, kg_ans, llm):
    llm_with_tools = llm.bind_tools(tools)

    prompt = f"""
        Question: {query}
        KG Answer: {kg_ans}
        """
    ai_msg = llm_with_tools.invoke(prompt)
    print("\nAI MESSAGE CONTENT:\n", ai_msg.content)

    if (ai_msg.tool_calls):
        print("\nOutput of Tool Calling:\n", ai_msg.tool_calls)
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"search_docs": search_docs}[tool_call["name"].lower()]
            print(selected_tool)
            tool_msg = selected_tool.invoke(tool_call)
        
        print("\nTOOL MESSAGE OUTPUT:\n", tool_msg.content)
        return tool_msg.content
    else: 
        print("INFORMATION: No tool calling required.")
        return None
    
def synthesize_answer(user_query, cypher_result, supporting_docs, llm_choice, llm):

    # Import the correct prompt template
    module = importlib.import_module(f"models.{llm_choice}_handler")
    SYNTH_PROMPT = module.SYNTH_PROMPT

    # Create dict
    results_dict = {
        "Result from Knowledge Graph": cypher_result,
        "Supporting Information": supporting_docs
    }

    # Create LLM Chain
    chain = SYNTH_PROMPT | llm
    output = chain.invoke({'query': user_query, 'context': results_dict})

    # Print outputs for debugging
    print("Synthesise answer text output:", output.content)
    return output.content