# Final Year Project 

This repository contains the codebase for my Final Year Project, which explores building a natural language interface for querying biomedical knowledge graphs using large language models. The system integrates Llama with Neo4j, allowing users to ask free-form questions and receive graph-based answers grounded in structured biomedical data.

The core of the project involves:
* Translating user queries into structured Cypher queries
* Interfacing with a Neo4j-based biomedical knowledge graph
* Using semantic search and retrieval techniques for schema and entity matching
* Presenting results through an interactive Streamlit front end


# Setup Instructions
## Environment Variables
In the main directory, create a `.env` file with the following values:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
LANGCHAIN_API_KEY=your_langsmith_api_key
```

## Data Preparation
### SNOMED CT
Download the SNOMED CT dataset from the official website: https://www.snomed.org/snomed-ct

Load the dataset into Neo4j by referring to the GitHub repo: https://github.com/IHTSDO/snomed-database-loader/tree/master

This will construct the initial medical knowledge graph structure.

### NHANES Data
Refer to the `data/kg_construction.txt` file for detailed instructions on:
* Adding NHANES (National Health and Nutrition Examination Survey) data to the graph
* Cleaning and merging the datasets within the graph database

See Appendix A and Appendix B of the final report for more details regarding schema design, preprocessing, and integration steps.

## Running the Application
### Step 1: Start the Neo4j Server
Make sure your local Neo4j server is running and accessible using the credentials in your .env file.

### Step 2: Start the Express Backend
Run the server from the main directory:
```
uvicorn api.main:app --reload
```

### Step 3: Run the Streamlit App
From the main directory: 
```
streamlit run interface/streamlit_ui.py
```
