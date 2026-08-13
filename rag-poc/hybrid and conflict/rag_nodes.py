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

import logging
from typing import List, TypedDict

from app.vectordb import opensearch_client

# "as llm" keeps every reference below (llm.ask_claude(...)) identical
# to how this file worked before the folder reorganization.
from app.llm import bedrock_client as llm

logger = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    """
    WHAT THIS IS:
    One chunk coming back from opensearch_client.search_top_chunks() -
    its text PLUS the citation metadata (document/section/page) needed
    to tell Claude exactly where it came from, instead of Claude having
    to guess a reference from the chunk's raw text.
    """

    content: str
    document_id: str
    document_name: str
    page: str
    section: str
    chunk_id: str


class GraphState(TypedDict):
    """
    WHAT THIS IS:
    A TypedDict describing every piece of data that flows through our
    graph.

    FIELDS:
    - question:         the user's original question (set before the
                         graph even starts running)
    - request_id:        a short unique ID generated once per request
                         (see app/api/routes/chat.py), used to tag every
                         log line so you can trace ONE request through
                         retrieve -> generate, even with multiple users
                         hitting the API at the same time
    - retrieved_chunks: the list of chunks (text + citation metadata)
                         found by OpenSearch (filled in by retrieve_node)
    - answer:            Claude's primary answer, as a normalized dict
                         (description, header, subheader, references,
                         followupquestions, steps) - filled in by
                         generate_node
    - answeroption:      a SECOND, independent answer - only populated
                         when Claude identifies a genuine conflict
                         between sources (see the system prompt's
                         "MULTI-SOURCE SYNTHESIS, CONFLICT DETECTION"
                         section). {} in the normal, non-conflict case -
                         filled in by generate_node
    """

    question: str
    request_id: str
    retrieved_chunks: List[RetrievedChunk]
    answer: dict
    answeroption: dict


def retrieve_node(state: GraphState) -> dict:
    """
    NODE 1: RETRIEVE
    ------------------
    Reads `question` off the state, turns it into an embedding, searches
    OpenSearch for the top matching chunks, and writes those chunks back
    onto the state under `retrieved_chunks`.
    """
    question = state["question"]
    request_id = state["request_id"]

    logger.info(f"[{request_id}] [retrieve_node] Question received: {question}")

    query_embedding = opensearch_client.get_query_embedding(question)
    chunks = opensearch_client.search_top_chunks(question, query_embedding, request_id)

    logger.info(
        f"[{request_id}] [retrieve_node] Retrieved {len(chunks)} chunk(s) from OpenSearch"
    )
    for i, chunk in enumerate(chunks, start=1):
        logger.info(
            f"[{request_id}] [retrieve_node]     {i}. "
            f"document={chunk['document_name']!r} page={chunk['page']!r} "
            f"section={chunk['section']!r} - {chunk['content'][:120]!r}"
        )

    return {"retrieved_chunks": chunks}


def generate_node(state: GraphState) -> dict:
    """
    NODE 2: GENERATE
    ------------------
    Reads `question` AND `retrieved_chunks` off the state, sends both to
    Claude, and writes BOTH parts of Claude's response - `answer` and
    `answeroption` - back onto the state. `answeroption` is {} unless
    Claude identified a genuine conflict between sources.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    request_id = state["request_id"]

    logger.info(f"[{request_id}] [generate_node] Sending question + chunks to Claude...")

    answer, answeroption = llm.ask_claude(question, chunks)

    logger.info(f"[{request_id}] [generate_node] Claude's answer: {answer}")
    if answeroption:
        logger.info(
            f"[{request_id}] [generate_node] Conflict detected - answeroption: {answeroption}"
        )

    return {"answer": answer, "answeroption": answeroption}
