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
    and indexes them inside a local highly scalable FAISS vector database instance.
    """
    print("🧠 Initializing Gemini Vector Embedding Matrix Platform...")
    # Using the standardized industry-grade text-embedding model from Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    print("⚡ Generating vectors and building FAISS local database index architecture...")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    
    # Save the computed index vector mapping binaries locally to disk
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_DB_DIR)
    print(f"💾 Vector index saved securely to disk cache at: {VECTOR_DB_DIR}")
    
    return vector_store

def get_local_vector_retriever():
    """
    Loads the locally stored FAISS database index file and exposes an operational 
    retriever node pipeline capable of running semantic text matching queries.
    """
    if not os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
        raise FileNotFoundError(f"No indexed database files found at {VECTOR_DB_DIR}. Ingest data first.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Return as an active data retriever query wrapper targeting top 3 matches
    return vector_store.as_retriever(search_kwargs={"k": 3})