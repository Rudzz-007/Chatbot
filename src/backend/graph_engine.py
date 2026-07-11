import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import the vector retriever
from backend.vector_store import get_local_vector_retriever

# 1. Load environment variables
load_dotenv()

# 2. Define the State Structure
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
        latest_user_query = state["messages"][-1].content
        matched_docs = retriever.invoke(latest_user_query)
        combined_context = "\n\n".join([doc.page_content for doc in matched_docs])
        return {"context": combined_context}
    except FileNotFoundError:
        return {"context": ""}

# 5. Define the Prompt-Engineered Chat Node with Guardrails
def chat_node(state: State):
    """
    Applies strict guardrails and behavioral constraints to Gemini's prompt framework.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)  # Dropped temperature to 0.1 for maximum factual accuracy
    
    current_context = state.get("context", "")
    
    # Advanced System Prompt with strict constraints
    system_instruction = (
        "ROLE:\n"
        "You are an elite, highly knowledgeable AI Research Assistant. Your goal is to deliver clear, "
        "accurate, and structurally scannable answers to the user.\n\n"
        
        "STRICT GUARDRAILS:\n"
        "1. Direct Factual Grounding: You must prioritize the text provided in the DOCUMENT CONTEXT below. "
        "If the answer can be found there, base your answer entirely on it.\n"
        "2. Strict Hallucination Prevention: If the context is empty, or if the user's question asks about "
        "specific facts/data points NOT present in the context, you must clearly state: "
        "'I am sorry, but that specific information is not available in the provided document context.' "
        "Do not invent facts or guess.\n"
        "3. Clear Formatting: Use markdown elements wisely to ensure maximum scannability. Use bolding (**...**) "
        "for key phrases, headers for hierarchy, and bullet points or numbered lists to break down complex explanations.\n\n"
        
        f"--- DOCUMENT CONTEXT ---\n{current_context}"
    )
    
    # Bundle the advanced instructions with the active chat log
    full_messages = [SystemMessage(content=system_instruction)] + state["messages"]
    
    response = llm.invoke(full_messages)
    return {"messages": [response]}

# 6. Map the Multi-Node Graph Architecture
builder.add_node("retrieve", retrieval_node)
builder.add_node("chatbot", chat_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "chatbot")
builder.add_edge("chatbot", END)

# 7. Initialize Memory Checkpointer and Compile
memory = MemorySaver()
chatbot = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🤖 Day 8 Multi-Node Graph with advanced guardrails compiled successfully!")