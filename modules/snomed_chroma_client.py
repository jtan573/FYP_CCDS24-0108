__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import math, os
import pandas as pd
from rapidfuzz import fuzz
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

def initialise_snomed_client():
    
    embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

    COLLECTION_NAME = "snomed_terms"
    PERSIST_DIRECTORY = "snomed_vector_store"

    vector_store = Chroma(
        persist_directory=os.path.join("data", PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vector_store

# vector_store = initialise_snomed_client()
# retriever = vector_store.as_retriever()

# -------------------------------
# Helper function to store SNOMED CT synonyms
# -------------------------------
def add_documents_snomed():
    vector_store = initialise_snomed_client()
    retriever = vector_store.as_retriever()

    # Intialise the data loader
    loader = CSVLoader(file_path='./data/snomed_lookup_table.csv',
        metadata_columns = ["sctid", "nodeType", "canonicalName"],         
        content_columns = "synonymTerm",
        encoding="utf8"
    )
    print("Loader initialised!")

    data = loader.load()
    for doc in data:
        doc.metadata.pop("source", None)
        doc.metadata.pop("row", None)
        doc.metadata["nodeType"] = doc.metadata["nodeType"].lower()
        doc.metadata["canonicalName"] = doc.metadata["canonicalName"].lower()
    
    print("Example of one document:")
    print("Page Content:", data[0].page_content[:100])
    print("Metadata:", data[0].metadata)
    print(f"Loaded {len(data)} documents.")

    # --- Chunk and add documents ---
    BATCH_SIZE = 10000  # adjust based on memory or API limits

    for i in range(0, len(data), BATCH_SIZE):
        chunk = data[i:i + BATCH_SIZE]
        try:
            retriever.add_documents(chunk, ids=None)
            print(f"Added batch {i // BATCH_SIZE + 1} of {math.ceil(len(data)/BATCH_SIZE)}")
        except Exception as e:
            print(f"Error in batch {i // BATCH_SIZE + 1}: {e}")
            break

    print(f"Successfully added documents to the vector store.")

# add_documents_snomed()

# -------------------------------
# Implement FSN Retriever
# -------------------------------
def resolve_fsn(term: str, nodeTypes: list):
    
    # Filter the dataframe
    df = pd.read_csv("./data/snomed_lookup_table.csv")
    category_list = [cat.strip().lower() for cat in nodeTypes.split(",")]
    df_filtered = df[df['nodeType'].str.lower().isin(category_list)]

    normalized_term = term.strip().lower()

    # Try fuzzy matching
    best_score = 0
    best_match = None

    for _, row in df_filtered.iterrows():
        candidate = str(row['synonymTerm']).strip().lower()
        score = fuzz.QRatio(normalized_term, candidate)
        if score > best_score:
            best_score = score
            best_match = row

    if best_score >= 90:
        canonical_name = best_match['canonicalName']
        print(f"Fuzzy match found in CSV (score={best_score}): {canonical_name}")
        return canonical_name

    # Fallback to semantic search
    vector_store = initialise_snomed_client()

    category_list = [cat.strip() for cat in nodeTypes.split(",")]
    cat_list_lower = [item.lower() for item in category_list]
    filter_dict = {"nodeType": {"$in": cat_list_lower}}

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, 'filter': filter_dict}
    )

    retrieved_docs = retriever.invoke(term)
    print("\nInitial results:\n", retrieved_docs)
    
    if retrieved_docs:
        doc = retrieved_docs[0]
        print(doc.metadata["canonicalName"])
        return doc.metadata["canonicalName"]
    
    return ''

# doc = resolve_fsn("blood glucose", "observableentity")
# print(doc)
