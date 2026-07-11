import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables for the API handshake layer
load_dotenv()

# Define the absolute directory path where the vector database matrix caches locally
VECTOR_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vectors"))

def create_and_store_embeddings(document_chunks):
    """
    Converts raw document text chunks into vector embeddings using Gemini's embedding model
    and indexes them inside a local, highly scalable FAISS vector database instance.
    """
    print("[EMBED] Initializing Gemini Vector Embedding Matrix...")
    # Updated to the current mainline active embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print(f"[EMBED] Generating vectors for {len(document_chunks)} chunks and building FAISS index...")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    
    # Save the computed index vector mapping binaries locally to disk
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_DB_DIR)
    print(f"[SAVED] Vector index saved securely to disk at: {VECTOR_DB_DIR}")
    
    return vector_store

def get_local_vector_retriever():
    """
    Loads the locally stored FAISS database index file and exposes an operational 
    retriever node pipeline capable of running semantic text matching queries.
    """
    index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No indexed database files found at {VECTOR_DB_DIR}. Please ingest a document first.")
        
    # Updated to the current mainline active embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Return as an active data retriever query wrapper targeting top 3 matches
    return vector_store.as_retriever(search_kwargs={"k": 3})