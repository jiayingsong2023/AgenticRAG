import os
import hashlib
import json
import shutil
from typing import List, Dict, Set
from dotenv import load_dotenv

import onnxruntime as ort
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
)
import networkx as nx
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

class Librarian:
    def __init__(self, data_dir: str = None, index_path: str = "faiss_index", state_path: str = "librarian_state.json"):
        self.data_dir = data_dir or os.getenv("DATA_DIR", "data")
        self.index_path = index_path
        self.state_path = state_path
        self.state = self._load_state()
        
        # Check for DirectML support
        if "DmlExecutionProvider" in ort.get_available_providers():
            self.providers = ["DmlExecutionProvider"]
            print("Librarian: AMD GPU (DirectML) detected for embeddings.")
        else:
            self.providers = None
            print("Librarian: DirectML not found, using CPU for embeddings.")
            
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            providers=self.providers
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        # Graph setup
        self.graph_path = "graph_index.json"
        self.graph = nx.Graph()
        
        # Local LLM for Graph Extraction (Qwen 0.5B is very small and fast)
        print("Librarian: Loading local LLM for graph extraction (Qwen/Qwen2.5-0.5B-Instruct)...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)

    def _load_state(self) -> Dict[str, str]:
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _load_single_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".csv":
                loader = CSVLoader(file_path)
            elif ext == ".json":
                loader = JSONLoader(file_path, jq_schema=".", text_content=False)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            else:
                return []
            return loader.load()
        except Exception as e:
            print(f"Librarian: Error loading {file_path}: {e}")
            return []

    def sync(self, force: bool = False):
        """Synchronize the vector store with the data directory."""
        if not os.path.exists(self.data_dir):
            print(f"Librarian: Data directory '{self.data_dir}' not found.")
            return

        current_files = {}
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                current_files[file_path] = self._get_file_hash(file_path)

        # Identify changes
        new_or_changed = []
        deleted = []
        
        if force:
            new_or_changed = list(current_files.keys())
            self.state = {}
        else:
            for path, h in current_files.items():
                if path not in self.state or self.state[path] != h:
                    new_or_changed.append(path)
            
            for path in self.state:
                if path not in current_files:
                    deleted.append(path)

        if not new_or_changed and not deleted:
            print("Librarian: Knowledge base is up to date. No changes detected.")
            return

        print(f"Librarian: Found {len(new_or_changed)} new/changed files and {len(deleted)} deleted files.")

        # For FAISS, incremental deletion is hard. If anything changed, we rebuild for now
        # but only load the files that are actually present.
        # Future optimization: Use a vector DB that supports incremental updates better.
        
        all_documents = []
        for path in current_files:
            print(f"Librarian: Processing {os.path.basename(path)}...")
            docs = self._load_single_document(path)
            all_documents.extend(docs)

        if not all_documents:
            print("Librarian: No documents found to index.")
            return

        print(f"Librarian: Splitting {len(all_documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(all_documents)
        
        print(f"Librarian: Creating vector store with {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        print("\nLibrarian: Embedding complete. Indexed documents:")
        for path in current_files:
            print(f"  - {os.path.basename(path)}")
        print("")

        print(f"Librarian: Saving index to '{self.index_path}'...")
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        vectorstore.save_local(self.index_path)
        
        # Update state
        self.state = current_files
        self._save_state()
        
        # --- Graph Extraction ---
        print("Librarian: Extracting knowledge graph (this may take a while)...")
        
        # Process chunks in batches to avoid OOM and provide progress
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Librarian: Processing graph batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
            try:
                graph_documents = self.graph_transformer.convert_to_graph_documents(batch)
                
                for g_doc in graph_documents:
                    for node in g_doc.nodes:
                        # Normalize node ID and type
                        node_id = str(node.id).strip().title()
                        self.graph.add_node(node_id, type=node.type)
                    for edge in g_doc.edges:
                        # Normalize source and target IDs
                        source_id = str(edge.source.id).strip().title()
                        target_id = str(edge.target.id).strip().title()
                        self.graph.add_edge(source_id, target_id, relation=edge.type)
            except Exception as e:
                print(f"Librarian: Error in graph extraction batch: {e}")
        
        print(f"Librarian: Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        self._save_graph()
        
        print("Librarian: Sync complete.")

    def _save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, 'w') as f:
            json.dump(data, f)
        print(f"Librarian: Graph saved to '{self.graph_path}'.")

    def _load_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
            print(f"Librarian: Graph loaded from '{self.graph_path}'.")

if __name__ == "__main__":
    librarian = Librarian()
    librarian.sync(force=True)
