import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Check for DirectML support
import onnxruntime as ort
if "DmlExecutionProvider" in ort.get_available_providers():
    PROVIDERS = ["DmlExecutionProvider"]
    print("AMD GPU (DirectML) detected for embeddings.")
else:
    PROVIDERS = None
    print("DirectML not found, using CPU for embeddings.")

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
)

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = "faiss_index"

def load_documents(data_dir: str) -> List[Document]:
    documents = []
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return documents

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                if ext == ".pdf":
                    print(f"Loading PDF: {file}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif ext == ".docx":
                    print(f"Loading Word: {file}")
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif ext == ".csv":
                    print(f"Loading CSV: {file}")
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())
                elif ext == ".json":
                    print(f"Loading JSON: {file}")
                    # Using jq schema to extract all content, adjust as needed
                    loader = JSONLoader(file_path, jq_schema=".", text_content=False) 
                    documents.extend(loader.load())
                elif ext == ".txt":
                    print(f"Loading Text: {file}")
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                else:
                    print(f"Skipping unsupported file: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
    return documents

def count_files(data_dir: str):
    counts = {}
    if not os.path.exists(data_dir):
        return counts
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
    return counts

def ingest():
    default_dir = os.getenv("DATA_DIR", "data")
    user_input = input(f"Enter data directory (default: '{default_dir}'): ").strip()
    data_dir = user_input if user_input else default_dir
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    print(f"\nScanning '{data_dir}'...")
    counts = count_files(data_dir)
    if not counts:
        print("No files found.")
        return
        
    print("Found files:")
    for ext, count in counts.items():
        print(f"  {ext}: {count}")
    
    confirm = input("\nProceed with ingestion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Ingestion cancelled.")
        return

    print(f"\nLoading documents from '{data_dir}'...")
    raw_documents = load_documents(data_dir)
    
    if not raw_documents:
        print("No supported documents found to ingest.")
        return

    print(f"Loaded {len(raw_documents)} documents.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks.")

    print("Loading FastEmbed embeddings...")
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        providers=PROVIDERS
    )
    
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    if os.path.exists(INDEX_PATH):
        import shutil
        print(f"Removing existing index at '{INDEX_PATH}'...")
        shutil.rmtree(INDEX_PATH)

    print("Saving vector store...")
    vectorstore.save_local(INDEX_PATH)
    print(f"Ingestion complete. Index saved to '{INDEX_PATH}'.")

if __name__ == "__main__":
    ingest()
