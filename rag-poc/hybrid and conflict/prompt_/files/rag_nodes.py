# =============================================================================
# app/graph/nodes/rag_nodes.py
# =============================================================================
# WHY THIS FILE EXISTS:
# LangGraph works by moving a "State" (a shared piece of data) through a
# series of "Nodes" (plain Python functions), connected by "Edges" - some
# fixed, some conditional (branching based on state). This file defines:
#   1. What that State looks like (GraphState)
#   2. The four Nodes that process it (retrieve_node, detect_conflict_node,
#      generate_single_node, generate_conflict_node)
#   3. The routing function that decides which generation node runs
#      (route_after_conflict_detection)
#
# WHY CONFLICT HANDLING IS NOW A GRAPH BRANCH, NOT A PROMPT INSTRUCTION:
# The old design made ONE Claude call, with ONE prompt, asked to BOTH
# detect a conflict AND produce the answer(s) in the same JSON reply.
# That's exactly why "answeroption" was unreliable - a correctly-detected
# conflict still depended on Claude remembering to ALSO populate a second
# field inside one blob. Now conflict detection is its own node with a
# real conditional edge, so producing two answers is structural:
#
#     START -> retrieve -> detect_conflict --(no conflict)--> generate_single -> END
#                                           \--(conflict)----> generate_conflict -> END
#
# See app/graph/graph.py for how these nodes/edges are actually wired,
# and app/prompts/system_prompt_UPDATED.py for the three prompts each
# Claude-calling node uses.
# =============================================================================

import logging
from typing import List, TypedDict

from app.vectordb import opensearch_client

# "as llm" keeps every reference below (llm.detect_conflict(...), etc.)
# identical to how this file worked before the folder reorganization.
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
    - question:          the user's original question (set before the
                          graph even starts running)
    - request_id:         a short unique ID generated once per request
                          (see app/api/routes/chat.py), used to tag every
                          log line so you can trace ONE request through
                          retrieve -> detect_conflict -> generate, even
                          with multiple users hitting the API at once.
    - retrieved_chunks:  the list of chunks (text + citation metadata)
                          found by OpenSearch (filled in by retrieve_node)
    - conflict_detected: whether detect_conflict_node found a genuine
                          conflict between sources for this question -
                          drives the conditional edge to
                          generate_single_node vs generate_conflict_node.
    - answer:             Claude's primary answer, as a normalized dict
                          (description, header, subheader, references,
                          documentids, followupquestions, steps) - filled
                          in by whichever generate node ran.
    - answeroption:       a SECOND, independent answer - {} unless
                          generate_conflict_node ran (i.e.
                          conflict_detected was True).
    """

    question: str
    request_id: str
    retrieved_chunks: List[RetrievedChunk]
    conflict_detected: bool
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


def detect_conflict_node(state: GraphState) -> dict:
    """
    NODE 2: DETECT CONFLICT
    ------------------------
    Reads `question` AND `retrieved_chunks` off the state, makes ONE
    small, focused Claude call (CONFLICT_DETECTION_PROMPT) whose only
    job is a yes/no judgment: do the retrieved sources genuinely,
    irreconcilably conflict on this exact question? Writes the result
    onto state as `conflict_detected` - route_after_conflict_detection
    below reads this to pick the next node.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    request_id = state["request_id"]

    logger.info(f"[{request_id}] [detect_conflict_node] Checking for a genuine conflict...")

    conflict_detected = llm.detect_conflict(question, chunks, request_id)

    return {"conflict_detected": conflict_detected}


def route_after_conflict_detection(state: GraphState) -> str:
    """
    CONDITIONAL EDGE FUNCTION
    --------------------------
    LangGraph calls this after detect_conflict_node runs. Its return
    value must match one of the keys in the `path_map` passed to
    add_conditional_edges(...) in app/graph/graph.py - here, either
    "conflict" or "no_conflict" - which tells LangGraph which node to
    run next.
    """
    if state["conflict_detected"]:
        return "conflict"
    return "no_conflict"


def generate_single_node(state: GraphState) -> dict:
    """
    NODE 3a: GENERATE (SINGLE ANSWER)
    -----------------------------------
    Runs ONLY when detect_conflict_node found conflict_detected=False.
    Makes ONE Claude call (SINGLE_ANSWER_PROMPT) covering every
    non-conflict situation - greetings, vague input, unrelated/
    insufficient context, Case 1 synthesis, Case 3 clarifying questions -
    and writes the result onto state as `answer`, with `answeroption`
    explicitly set to {} (there is no second answer on this path).
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    request_id = state["request_id"]

    logger.info(f"[{request_id}] [generate_single_node] Generating single answer...")

    answer = llm.ask_claude_single(question, chunks, request_id)

    logger.info(f"[{request_id}] [generate_single_node] Claude's answer: {answer}")

    return {"answer": answer, "answeroption": {}}


def generate_conflict_node(state: GraphState) -> dict:
    """
    NODE 3b: GENERATE (CONFLICT - TWO ANSWERS)
    ---------------------------------------------
    Runs ONLY when detect_conflict_node found conflict_detected=True.
    Makes ONE Claude call (CONFLICT_ANSWER_PROMPT) whose only job is
    producing two independent, non-blended answers - one per conflicting
    source - and writes both onto state as `answer` and `answeroption`.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    request_id = state["request_id"]

    logger.info(f"[{request_id}] [generate_conflict_node] Generating two answers (conflict)...")

    answer, answeroption = llm.ask_claude_conflict(question, chunks, request_id)

    logger.info(f"[{request_id}] [generate_conflict_node] answer: {answer}")
    logger.info(f"[{request_id}] [generate_conflict_node] answeroption: {answeroption}")

    return {"answer": answer, "answeroption": answeroption}
