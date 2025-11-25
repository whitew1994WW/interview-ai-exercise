"""Streamlit app for RAG demo.

Start from project root with :
```bash
PYTHONPATH=. streamlit run demo/main.py
```
"""

import requests
import streamlit as st

from demo.ping import display_message_if_ping_fails

st.set_page_config(
    "RAG Example",
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

st.title("RAG Example 🤖")

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
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    st.empty()

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
