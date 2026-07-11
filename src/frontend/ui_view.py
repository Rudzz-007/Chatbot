import streamlit as st
import sys
import os
import uuid
from langchain_core.messages import HumanMessage

# Append parent folders to sys.path for robust import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from backend.graph_engine import chatbot
    from backend.doc_parser import process_pdf_document
    from backend.vector_store import create_and_store_embeddings
except ModuleNotFoundError:
    from src.backend.graph_engine import chatbot
    from src.backend.doc_parser import process_pdf_document
    from src.backend.vector_store import create_and_store_embeddings

st.set_page_config(page_title="GenAI Research Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Advanced Chatbot ")


# Initialize a dynamic, unique thread ID for the LangGraph checkpointer if not present
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- SIDEBAR: DOCUMENT INGESTION & UTILITIES ---
with st.sidebar:
    st.header("📄 Knowledge Ingestion")
    st.write("Upload a PDF document to update the assistant's local database.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        temp_dir = os.path.abspath(os.path.join(root_dir, "data/temp"))
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Uploaded: {uploaded_file.name}")
        
        if st.button("⚡ Build Knowledge Index", use_container_width=True):
            with st.spinner("Parsing text and compiling vector matrix..."):
                try:
                    chunks = process_pdf_document(temp_file_path)
                    create_and_store_embeddings(chunks)
                    st.success("🎉 Vector Store updated! The chatbot now has access to this data.")
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    st.error(f"Ingestion Pipeline Failed: {e}")
                    
    st.markdown("---")
    st.header("⚙️ Session Utilities")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Chat history has been reset. How can I help you now?"}
        ]
        st.session_state.thread_id = str(uuid.uuid4())
        st.success("Conversation context cleared completely!")
        st.rerun()

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your RAG-powered Research Assistant. Upload a document in the sidebar, or ask me anything!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_graph_state = {"messages": [HumanMessage(content=user_input)]}
    
    with st.spinner("Analyzing context..."):
        try:
            updated_state = chatbot.invoke(initial_graph_state, config)
            last_ai_message = updated_state["messages"][-1].content
            
            with st.chat_message("assistant"):
                st.markdown(last_ai_message)
            st.session_state.messages.append({"role": "assistant", "content": last_ai_message})
            
        except Exception as e:
            st.error(f"Execution Error running the graph RAG workflow: {e}")