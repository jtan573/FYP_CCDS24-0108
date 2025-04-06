__import__('pysqlite3')
import sys
from typing import List
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import math, os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

def initialise_medquad_client():

    embeddings = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

    COLLECTION_NAME = "medquad"
    PERSIST_DIRECTORY = "medquad_vector_store"

    vector_store = Chroma(
        persist_directory=os.path.join("data", PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vector_store

# -------------------------------
# Helper function to store SNOMED CT synonyms
# -------------------------------
def add_documents():
    vector_store = initialise_medquad_client()
    retriever = vector_store.as_retriever() 

    # Intialise the data loader
    loader = CSVLoader(file_path='./data/medQuad.csv',
        metadata_columns = ["qtype", "Answer"],         
        content_columns = "Question"
    )
    print("Loader initialised!")

    data = loader.load()
    for doc in data:
        doc.metadata.pop("source", None)
        doc.metadata.pop("row", None)
    
    print("Example of one document:")
    print("Page Content:", data[0].page_content[:100])
    print("Metadata:", data[0].metadata)
    print(f"Loaded {len(data)} documents.")

    # --- Chunk and add documents ---
    BATCH_SIZE = 5000  # adjust based on memory or API limits

    for i in range(0, len(data), BATCH_SIZE):
        chunk = data[i:i + BATCH_SIZE]
        try:
            retriever.add_documents(chunk, ids=None)
            print(f"Added batch {i // BATCH_SIZE + 1} of {math.ceil(len(data)/BATCH_SIZE)}")
        except Exception as e:
            print(f"Error in batch {i // BATCH_SIZE + 1}: {e}")
            break

    print(f"Successfully added documents to the vector store.")

# add_documents()

# -------------------------------
# Document Retriever
# -------------------------------
def get_supporting_info(query: str):
    vector_store = initialise_medquad_client()

    # @chain
    # def retriever(query: str) -> List[Document]:
    #     docs, scores = zip(*vector_store.similarity_search_with_score(query))
    #     for doc, score in zip(docs, scores):
    #         doc.metadata["score"] = score
    #     return docs
    
    retriever = vector_store.as_retriever(
        search_type="mmr"
    ) 

    retrieved_docs = retriever.invoke(query)
    print("Initial results:", retrieved_docs)
    if retrieved_docs:
        doc = retrieved_docs[0]
        return doc.metadata['Answer']
    return None

# get_supporting_info("Get all findings related to \"smoking\".")