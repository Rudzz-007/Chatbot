import streamlit as st

st.set_page_config(page_title="GenAI Research Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Advanced AI Research Assistant")
st.caption("Day 2: Mastering UI Layer & Session State Mechanics")

# 1. Initialize Persistent Session State
# If 'messages' doesn't exist in our state dictionary yet, initialize it as an empty list.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your upgraded Research Assistant. How can I help you today?"}
    ]

# 2. Render Chat History Matrix
# Loop through our persistent state array and render every message using Streamlit's native UI blocks.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Capture and Handle User Input Actions
if user_input := st.chat_input("Ask me anything..."):
    
    # Render the user's message immediately on screen
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Append the message to our session state dictionary so it survives the next script re-run
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Day 2 Placeholder: A simple echo response to verify interface structure and state integrity
    echo_response = f"Echoing your input to verify state integrity: '{user_input}'"
    
    with st.chat_message("assistant"):
        st.markdown(echo_response)
        
    st.session_state.messages.append({"role": "assistant", "content": echo_response})