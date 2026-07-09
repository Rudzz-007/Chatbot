import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Load environment variables
load_dotenv()

# Ensure API Key is available
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("CRITICAL: GEMINI_API_KEY is missing from your environment or .env file.")

# 2. Define the State Structure
# This tracks the structural flow of messages across the nodes.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. Define the Graph Workflow Architecture
builder = StateGraph(State)

# 4. Define the Chat processing node
def chat_node(state: State):
    """
    Processes the conversation state by sending the message history 
    to the Gemini LLM and returning the updated state.
    """
    # Using the highly optimized gemini-2.5-flash model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # Generate the response based on the conversation history
    response = llm.invoke(state["messages"])
    
    return {"messages": [response]}

# 5. Connect the Graph Nodes and Edges
builder.add_node("chatbot", chat_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 6. Initialize conversational memory checkpointer
memory = MemorySaver()

# 7. Compile the Graph
# This compiled object is what our Streamlit frontend will import and invoke on Day 3.
chatbot = builder.compile(checkpointer=memory)

# Optional: Script verification run block
if __name__ == "__main__":
    print("🤖 Testing Backend Integration Locally...")
    
    # Pass an initial test thread ID and sample message
    config = {"configurable": {"thread_id": "test_session"}}
    initial_state = {"messages": [HumanMessage(content="Hi, I am Rudra!")]}
    
    output = chatbot.invoke(initial_state, config)
    print("\n--- Model Response ---")
    print(output["messages"][-1].content)