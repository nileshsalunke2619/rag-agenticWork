# =============================================================================
# app/graph/nodes/rag_nodes.py
# =============================================================================
# WHY THIS FILE EXISTS:
# LangGraph works by moving a "State" (a shared piece of data) through a
# series of "Nodes" (plain Python functions). This file defines:
#   1. What that State looks like (GraphState)
#   2. The two Nodes that will process it (retrieve_node, generate_node)
#
# Think of State as a clipboard that gets passed from person to person:
#   - Person 1 (retrieve_node) writes "here are the relevant chunks" on it
#   - Person 2 (generate_node) reads the chunks off the clipboard, writes
#     "here's the final answer" on it
# =============================================================================

from typing import List, TypedDict

from app.vectordb import opensearch_client

# "as llm" keeps every reference below (llm.ask_claude(...)) identical
# to how this file worked before the folder reorganization.
from app.llm import bedrock_client as llm


class GraphState(TypedDict):
    """
    WHAT THIS IS:
    A TypedDict describing every piece of data that flows through our
    graph.

    FIELDS:
    - question:         the user's original question (set before the
                         graph even starts running)
    - retrieved_chunks: the list of text chunks found by OpenSearch
                         (filled in by retrieve_node)
    - answer:            Claude's final answer, as a normalized dict
                         (description, header, question_id, references,
                         steps, subheader) - filled in by generate_node
    """

    question: str
    retrieved_chunks: List[str]
    answer: dict


def retrieve_node(state: GraphState) -> dict:
    """
    NODE 1: RETRIEVE
    ------------------
    Reads `question` off the state, turns it into an embedding, searches
    OpenSearch for the top matching chunks, and writes those chunks back
    onto the state under `retrieved_chunks`.
    """
    question = state["question"]

    print(f"\n[retrieve_node] Question received: {question}")

    query_embedding = opensearch_client.get_query_embedding(question)
    chunks = opensearch_client.search_top_chunks(query_embedding)

    print(f"[retrieve_node] Retrieved {len(chunks)} chunk(s) from OpenSearch:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"    {i}. {chunk[:120]!r}")

    return {"retrieved_chunks": chunks}


def generate_node(state: GraphState) -> dict:
    """
    NODE 2: GENERATE
    ------------------
    Reads `question` AND `retrieved_chunks` off the state, sends both to
    Claude, and writes Claude's answer back onto the state under
    `answer`.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    print("[generate_node] Sending question + chunks to Claude...")

    answer = llm.ask_claude(question, chunks)

    print(f"[generate_node] Claude's answer: {answer}\n")

    return {"answer": answer}
