import streamlit as st
import requests

# 🔗 Remote backend API
REMOTE_API_URL = "http://localhost:8080"  # Replace with your actual remote server IP

st.set_page_config(page_title="Medical KG Assistant", layout="centered")
st.title("🔍 Medical KG Assistant")

model_choice = "llama"
query = st.text_input("Enter your natural language question:")

if st.button("Submit"):
    with st.spinner("Processing..."):
        try:

            # Define per-section placeholders
            concepts_box = st.container()
            synonyms_box = st.container()
            cypher_box = st.container()
            cypher_error_box = st.container()
            cypher_result_box = st.container()
            eval_box = st.container()
            search_box = st.container()
            final_box = st.container()
            error_box = st.container()

            response = requests.post(
                f"{REMOTE_API_URL}/query",
                json={"user_query": query, "llm_provider": model_choice},
                stream=True
            )

            if response.status_code == 200:
                buffer = ""
                current_section = None

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    line = line.strip()

                    # Before switching section, flush buffer
                    def flush_buffer():
                        if current_section == "concepts":
                            concepts_box.code(buffer.strip(), language="json")
                        elif current_section == "synonyms":
                            synonyms_box.code(buffer.strip(), language="json")
                        elif current_section == "cypher":
                            cypher_box.code(buffer.strip(), language="cypher")
                        elif current_section == "cypher_error":
                            cypher_error_box.code(buffer.strip(), language="cypher")
                        elif current_section == "cypher_result":
                            cypher_result_box.code(buffer.strip(), language="cypher")
                        elif current_section == "eval":
                            eval_box.code(buffer.strip())
                        elif current_section == "search":
                            search_box.code(buffer.strip())
                        elif current_section == "final":
                            final_box.success(buffer.strip())

                    # Section headers from backend
                    if line.startswith("Extracting Concepts"):
                        flush_buffer()
                        current_section = "concepts"
                        concepts_box.markdown("**Extracting Concepts...**")
                        buffer = ""
                        continue
                    elif line.startswith("Resolving Synonyms"):
                        flush_buffer()
                        current_section = "synonyms"
                        synonyms_box.markdown("**Resolving Synonyms...**")
                        buffer = ""
                        continue
                    elif line.startswith("Generating Cypher"):
                        flush_buffer()
                        current_section = "cypher"
                        cypher_box.markdown("**Generating Cypher...**")
                        buffer = ""
                        continue
                    elif line.startswith("Executing Cypher"):
                        flush_buffer()
                        current_section = "cypher_result"
                        cypher_result_box.markdown("**Executing Cypher...**")
                        buffer = ""
                        continue
                    elif line.startswith("Cypher Query is invalid, attempt to find information from backup database"):
                        flush_buffer()
                        current_section = "cypher_error"
                        cypher_error_box.markdown(
                            "<span style='font-size: 0.9em'>Cypher Query is invalid/returns null, attempting to find information from backup database...</span>")
                        buffer = ""
                        continue
                    elif line.startswith("Evaluating Answer from KG"):
                        flush_buffer()
                        current_section = "eval"
                        eval_box.markdown("**Evaluating...**")
                        buffer = ""
                        continue
                    elif line.startswith("Searching chroma database for relevant information"):
                        flush_buffer()
                        current_section = "search"
                        search_box.markdown("**Searching database...**")
                        buffer = ""
                        continue
                    elif line.startswith("Synthesizing Final Answer"):
                        flush_buffer()
                        current_section = "final"
                        final_box.markdown(
                            "<h2 style='font-size: 28px;'>🧩 <b>Final Output...</b></h2>",
                            unsafe_allow_html=True
                        )
                        buffer = ""
                        continue
                    elif line.startswith("🚨"):
                        flush_buffer()
                        error_box.error(line)
                        continue

                    # Otherwise, accumulate the content
                    buffer += line + "\n"

                # Flush the final buffer once done
                flush_buffer()
            else:
                st.error(f"❌ Error from backend: {response.text}")

        except Exception as e:
            st.error(f"🚨 Connection error: {e}")


# Start the frontend:
# streamlit run interface/streamlit_ui.py
