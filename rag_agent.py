import os
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI

# Check for DirectML support
import onnxruntime as ort
if "DmlExecutionProvider" in ort.get_available_providers():
    PROVIDERS = ["DmlExecutionProvider"]
    print("AMD GPU (DirectML) detected for embeddings.")
else:
    PROVIDERS = None
    print("DirectML not found, using CPU for embeddings.")
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import END, StateGraph, START
import networkx as nx
import json
import difflib

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
INDEX_PATH = "faiss_index"
GRAPH_PATH = "graph_index.json"

# --- State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]
    graph_context: str
    retry_count: int

# --- Components ---

# 1. Retriever
print("Loading vector store (FastEmbed)...")
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    providers=PROVIDERS
)
try:
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"Error loading vector store: {e}")
    print("Please run ingest.py first.")
    exit(1)

# 1.1 Graph Store
print("Loading knowledge graph...")
graph = nx.Graph()
if os.path.exists(GRAPH_PATH):
    try:
        with open(GRAPH_PATH, 'r') as f:
            data = json.load(f)
            graph = nx.node_link_graph(data)
        print(f"Graph loaded with {graph.number_of_nodes()} nodes.")
    except Exception as e:
        print(f"Error loading graph: {e}")
else:
    print("Graph index not found. Hybrid retrieval will be limited.")

# 2. LLM
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=LLM_BASE_URL,
    temperature=0
)

# 3. Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

system_prompt_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_grader + "\n\n {format_instructions}"),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

retrieval_grader = grade_prompt | llm | parser

# 4. Generator
system_prompt_generator = """You are an assistant for question-answering tasks. \n
    Use the following pieces of retrieved context to answer the question. \n
    If you don't know the answer, just say that you don't know. \n
    Use three sentences maximum and keep the answer concise. \n
    Context: {context}"""

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_generator),
        ("human", "Question: {question}"),
    ]
)

rag_chain = generate_prompt | llm | StrOutputParser()

# 5. Query Rewriter
system_prompt_rewriter = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_rewriter),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

# 6. Entity Extractor
class ExtractedEntities(BaseModel):
    """Extracted entities from a query."""
    entities: List[str] = Field(description="List of named entities (people, places, concepts, etc.) found in the question")

system_prompt_extractor = """You are an expert at extracting key entities from user questions for knowledge graph lookup. \n
    Extract all significant entities (names, organizations, locations, specialized concepts). \n
    Return only the entities as a list."""

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_extractor + "\n\n {format_instructions}"),
        ("human", "Question: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) # Reuse JsonOutputParser

entity_extractor = extract_prompt | llm | parser

# --- Nodes ---

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Vector Retrieval
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} vector documents.")
    
    # Graph Retrieval (Enhanced Entity Linking)
    print("---GRAPH RETRIEVAL---")
    graph_context = ""
    
    try:
        # 1. Extract entities using LLM
        print("Extracting entities from question...")
        extraction_result = entity_extractor.invoke({"question": question})
        extracted_entities = extraction_result.get("entities", [])
        print(f"Extracted entities: {extracted_entities}")

        # 2. Fuzzy match entities to graph nodes
        linked_nodes = []
        graph_nodes = list(graph.nodes)
        
        for entity in extracted_entities:
            # Try exact match first (case-insensitive)
            matches = [node for node in graph_nodes if str(node).lower() == entity.lower()]
            if not matches:
                # Try fuzzy match
                matches = difflib.get_close_matches(entity, graph_nodes, n=1, cutoff=0.7)
            
            if matches:
                linked_nodes.extend(matches)
        
        linked_nodes = list(set(linked_nodes)) # De-duplicate

        if linked_nodes:
            print(f"Linked to graph nodes: {linked_nodes}")
            related_info = []
            
            # 3. Retrieve neighbors (1-hop)
            for node in linked_nodes:
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(node, neighbor)
                    relation = edge_data.get('relation', 'related')
                    related_info.append(f"{node} --({relation})--> {neighbor}")
            
            graph_context = "\n".join(related_info)
            print(f"Retrieved {len(related_info)} graph relations.")
        else:
            print("No matching entities linked in graph.")
            
    except Exception as e:
        print(f"Error during graph retrieval: {e}")
        # Fallback to simple keyword search if LLM extraction fails
        print("Falling back to simple keyword search...")
        words = question.replace("?", "").split()
        found_entities = [node for node in graph.nodes if any(word.lower() in str(node).lower() for word in words if len(word) > 3)]
        if found_entities:
            related_info = []
            for entity in found_entities:
                neighbors = list(graph.neighbors(entity))
                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(entity, neighbor)
                    related_info.append(f"{entity} --({edge_data.get('relation', 'related')})--> {neighbor}")
            graph_context = "\n".join(related_info)

    return {"documents": documents, "question": question, "graph_context": graph_context}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    try:
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            print(f"Grading result: {score}")
            grade = score['binary_score']
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}
    except Exception as e:
        print(f"ERROR IN GRADE DOCUMENTS: {e}")
        raise e

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    context = "\n\n".join([d.page_content for d in documents])
    if state.get("graph_context"):
        context += "\n\nKnowledge Graph Context:\n" + state["graph_context"]
        
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "retry_count": retry_count + 1}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        if retry_count >= 3:
            print("---DECISION: MAX RETRIES REACHED, GENERATING WITH NO CONTEXT---")
            return "generate"
            
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

# --- Graph ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

def run_rag_loop():
    print("（提示：您可以随时向 'data' 文件夹添加新文档，系统会自动同步）")
    print("输入 'exit' 或 'quit' 可退出程序。")
    
    # Track the last loaded index time to detect updates
    last_index_time = 0
    last_graph_time = 0
    if os.path.exists(INDEX_PATH):
        last_index_time = os.path.getmtime(INDEX_PATH)
    if os.path.exists(GRAPH_PATH):
        last_graph_time = os.path.getmtime(GRAPH_PATH)

    while True:
        # Check if index was updated by Librarian
        if os.path.exists(INDEX_PATH):
            current_index_time = os.path.getmtime(INDEX_PATH)
            if current_index_time > last_index_time:
                print("\n[Assistant] New vector knowledge detected! Reloading vector store...")
                global vectorstore, retriever
                vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever()
                last_index_time = current_index_time
        
        if os.path.exists(GRAPH_PATH):
            current_graph_time = os.path.getmtime(GRAPH_PATH)
            if current_graph_time > last_graph_time:
                print("\n[Assistant] New graph knowledge detected! Reloading knowledge graph...")
                global graph
                with open(GRAPH_PATH, 'r') as f:
                    data = json.load(f)
                    graph = nx.node_link_graph(data)
                last_graph_time = current_graph_time

        question = input("\n[提问]: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        inputs = {"question": question}
        print("\n--- Processing ---")
        try:
            value = None
            for output in app.stream(inputs, {"recursion_limit": 50}):
                for key, val in output.items():
                    print(f"Node '{key}' completed.")
                    value = val
            
            if value and 'generation' in value:
                print(f"\nAnswer: {value['generation']}")
            else:
                print("\nAnswer: I'm sorry, I couldn't generate an answer.")
        except Exception as e:
            print(f"An error occurred: {e}")
