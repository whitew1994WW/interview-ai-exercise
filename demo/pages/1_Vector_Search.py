"""Search page for querying the vector database directly."""

import json

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    page_title="Search - RAG Example",
    page_icon="🔍",
)

with st.sidebar:
    display_message_if_ping_fails()
    
    st.divider()
    st.subheader("Vector Database")
    
    if st.button("🔄 Rebuild Vector DB", type="primary", use_container_width=True):
        with st.spinner("Loading documents into vector database..."):
            try:
                response = requests.get("http://localhost/load", timeout=300)
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "ok":
                    st.success("✅ Vector database rebuilt successfully!")
                else:
                    st.warning(f"⚠️ Response: {result}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to rebuild vector database: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

st.title("🔍 Vector Database Search")

st.markdown(
    "Query the vector database directly to see matching chunks with their metadata and similarity scores."
)

# Search form
with st.form("search_form"):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            help="Search for relevant chunks in the vector database",
        )
    
    with col2:
        k = st.number_input(
            "Results",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of results to return",
        )
    
    search_button = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

# Display results
if search_button and query:
    with st.spinner("Searching vector database..."):
        try:
            response = requests.post(
                "http://localhost/search",
                json={"query": query, "k": k},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            
            # Display summary
            st.success(f"✅ Found {result['total_results']} matching chunk(s)")
            
            # Display each chunk
            for idx, chunk in enumerate(result["chunks"], 1):
                with st.expander(
                    f"Chunk {idx} (ID: {chunk['id']})"
                    + (f" - Distance: {chunk['distance']:.4f}" if chunk.get("distance") is not None else ""),
                    expanded=idx == 1,  # Expand first result by default
                ):
                    # Display document content
                    st.subheader("📄 Content")
                    try:
                        # Try to pretty-print if it's JSON
                        doc_content = json.loads(chunk["document"])
                        st.json(doc_content)
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, display as text
                        st.text(chunk["document"])
                    
                    # Display metadata if available
                    if chunk.get("metadata"):
                        st.subheader("🏷️ Metadata")
                        st.json(chunk["metadata"])
                    
                    # Display distance if available
                    if chunk.get("distance") is not None:
                        st.subheader("📊 Similarity")
                        st.metric("Distance", f"{chunk['distance']:.4f}")
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

elif search_button and not query:
    st.warning("⚠️ Please enter a search query")
