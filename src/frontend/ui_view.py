import streamlit as st
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

# Append the src directory to system paths so Python can resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.graph_engine import chatbot

st.set_page_config(page_title="GenAI Research Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Advanced AI Research Assistant")
st.caption("Day 3: Full End-to-End LangGraph Integration")

# 1. Initialize Persistent Session State for UI Display
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your LangGraph-powered Research Assistant. How can I help you today?"}
    ]

# 2. Render UI Chat Stream Matrix
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle Live User Submissions
if user_input := st.chat_input("Ask me anything..."):
    
    # Render user input immediately on screen
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 4. Prepare the LangGraph State Inputs and Configuration
    # We assign a static thread_id context so the memory checkpoint holds active sessions
    config = {"configurable": {"thread_id": "streamlit_default_session"}}
    initial_graph_state = {"messages": [HumanMessage(content=user_input)]}
    
    # 5. Invoke the Live Graph Execution Pipeline
    with st.spinner("Thinking..."):
        try:
            # Query the backend engine
            updated_state = chatbot.invoke(initial_graph_state, config)
            
            # Extract the absolute last message string from the returning state history list
            last_ai_message = updated_state["messages"][-1].content
            
            # Render the dynamic response inside the assistant view panel
            with st.chat_message("assistant"):
                st.markdown(last_ai_message)
            st.session_state.messages.append({"role": "assistant", "content": last_ai_message})
            
        except Exception as e:
            st.error(f"Execution Error running the graph compilation workflow: {e}")