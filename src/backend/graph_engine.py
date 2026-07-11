import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import the vector retriever we created on Day 5
from backend.vector_store import get_local_vector_retriever

# 1. Load environment variables
load_dotenv()

# 2. Define the State Structure
# We add a new field 'context' to store retrieved document text snippets
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

# 3. Define the Graph Workflow Architecture
builder = StateGraph(State)

# 4. Define the Retrieval Node
def retrieval_node(state: State):
    """
    Extracts the user's latest query, searches the local FAISS vector store,
    and stores the relevant text segments into the state context.
    """
    try:
        retriever = get_local_vector_retriever()
        # Grab the last text string typed by the user
        latest_user_query = state["messages"][-1].content
        
        # Pull the top relevant document blocks matching the meaning
        matched_docs = retriever.invoke(latest_user_query)
        
        # Combine the snippets into a single context string
        combined_context = "\n\n".join([doc.page_content for doc in matched_docs])
        return {"context": combined_context}
        
    except FileNotFoundError:
        # If no documents are uploaded yet, we pass empty context gracefully
        return {"context": ""}

# 5. Define the Upgraded Chat Node
def chat_node(state: State):
    """
    Injects the retrieved document context into a System Message template,
    and feeds it into Gemini alongside the conversation history.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # If we found relevant document data, prepend it as a System instruction
    current_context = state.get("context", "")
    system_instruction = (
        "You are an advanced AI Research Assistant. Answer the user's question based "
        "strictly on the provided document context below. If the answer cannot be found "
        "in the context, use your general knowledge but clearly state that it wasn't in the document.\n\n"
        f"--- DOCUMENT CONTEXT ---\n{current_context}"
    )
    
    # Bundle the system rule with the existing user message flow
    full_messages = [SystemMessage(content=system_instruction)] + state["messages"]
    
    response = llm.invoke(full_messages)
    return {"messages": [response]}

# 6. Map the Multi-Node Graph Architecture
builder.add_node("retrieve", retrieval_node)
builder.add_node("chatbot", chat_node)

# Flow: START ──► retrieve ──► chatbot ──► END
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "chatbot")
builder.add_edge("chatbot", END)

# 7. Initialize Memory Checkpointer and Compile
memory = MemorySaver()
chatbot = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🤖 Day 6 Multi-Node Graph compiled successfully!")