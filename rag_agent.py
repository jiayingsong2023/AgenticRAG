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

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
INDEX_PATH = "faiss_index"

# --- State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]
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

    # Retrieval
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    for i, doc in enumerate(documents):
        print(f"Doc {i}: {doc.page_content[:100]}...")
    return {"documents": documents, "question": question}

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
    generation = rag_chain.invoke({"context": documents, "question": question})
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

if __name__ == "__main__":
    print("Agentic RAG System Ready.")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        question = input("\nEnter your question: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        inputs = {"question": question}
        print("\n--- Processing ---")
        try:
            for output in app.stream(inputs, {"recursion_limit": 50}):
                for key, value in output.items():
                    print(f"Node '{key}' completed.")
            
            # The final state is in 'value' from the last iteration
            print(f"\nAnswer: {value['generation']}")
        except Exception as e:
            print(f"An error occurred: {e}")
