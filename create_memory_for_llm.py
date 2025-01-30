# create_memory_for_llm.py

import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_environment():
    """
    Load environment variables from .env file.
    """
    load_dotenv(find_dotenv())
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables.")
    return hf_token

def load_pdf_files(data_path: str):
    """
    Load PDF files from the specified directory.
    """
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def create_chunks(documents):
    """
    Split documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    """
    Initialize the HuggingFace Embeddings model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def store_embeddings(text_chunks, embedding_model, db_path: str):
    """
    Create and save the FAISS vector store.
    """
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    print(f"Embeddings stored successfully at '{db_path}'.")

def main():
    DATA_PATH = "data/"
    DB_FAISS_PATH = "vectorstore/db_faiss"

    # Load environment variables
    load_environment()

    # Create vectorstore directory if it doesn't exist
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)

    # Load PDFs
    print("Loading PDF files...")
    documents = load_pdf_files(DATA_PATH)
    print(f"Number of PDF documents loaded: {len(documents)}")

    # Create text chunks
    print("Creating text chunks...")
    text_chunks = create_chunks(documents)
    print(f"Number of text chunks created: {len(text_chunks)}")

    # Initialize embedding model
    print("Initializing embedding model...")
    embedding_model = get_embedding_model()

    # Store embeddings
    print("Storing embeddings...")
    store_embeddings(text_chunks, embedding_model, DB_FAISS_PATH)

if __name__ == "__main__":
    main()
