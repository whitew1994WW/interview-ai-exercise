"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/Chat.py
```
"""

import json

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    page_title="Chat - RAG Example",
    page_icon="🤖",
)

if "session" not in st.session_state:
    st.session_state.session = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]


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

st.title("💬 Chat with RAG")
st.markdown("Ask questions and get answers powered by the vector database.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Which path gives me the candidate list?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = ""

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost/chat", json={"query": prompt}, timeout=60
            )
            response.raise_for_status()
            result = response.json()
            msg = result["message"]
            context_chunks = result.get("context", [])
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    st.empty()

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
    # Display context in tabs
    if context_chunks:
        with st.expander(f"📚 View Context ({len(context_chunks)} chunks used)", expanded=False):
            st.markdown("**Context chunks used to generate this response:**")
            
            # Create tabs for each chunk
            tab_labels = [
                f"Chunk {idx}" + (f" ({chunk.get('distance', 0):.3f})" if chunk.get("distance") is not None else "")
                for idx, chunk in enumerate(context_chunks, 1)
            ]
            tabs = st.tabs(tab_labels)
            
            for tab, chunk in zip(tabs, context_chunks):
                with tab:
                    # Display chunk ID
                    st.caption(f"ID: {chunk.get('id', 'unknown')}")
                    
                    # Display document content
                    st.markdown("**📄 Content**")
                    try:
                        # Try to pretty-print if it's JSON
                        doc_content = json.loads(chunk["document"])
                        st.json(doc_content)
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, display as multiline text
                        st.text_area(
                            "Document",
                            chunk["document"],
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    
                    # Display metadata and distance in columns
                    if chunk.get("metadata") or chunk.get("distance") is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if chunk.get("metadata"):
                                st.markdown("**🏷️ Metadata**")
                                st.json(chunk["metadata"])
                        
                        with col2:
                            if chunk.get("distance") is not None:
                                st.markdown("**📊 Similarity**")
                                st.metric("Distance", f"{chunk['distance']:.4f}")
