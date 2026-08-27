# =============================================================================
# app/api/routes/chat.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This defines the actual POST /agent/v1/orchestration/chat endpoint. It's an
# APIRouter (not a full FastAPI app) so that main.py (the root entry
# point) can "include" it into the real app - this is the standard
# modular FastAPI pattern: routers define endpoints, main.py assembles
# them into one application.
# =============================================================================

import logging
import time
import uuid

from fastapi import APIRouter

from app.graph.graph import rag_graph
from app.models.schemas import AskRequest, AskResponse
from app.vectordb.opensearch_client import get_query_embedding, search_nearest_feedback

logger = logging.getLogger(__name__)

# APIRouter works like a mini FastAPI app - you define routes on it,
# then main.py "mounts" it onto the real app with app.include_router(...).
router = APIRouter()


@router.post("/agent/v1/orchestration/chat", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Generates a short request_id for this call, so every log line
       from here through retrieve_node/generate_node can be traced back
       to this ONE request - useful once multiple users are hitting the
       API at the same time.
    2. Takes the user's question from the request body.
    3. FEEDBACK CACHE CHECK (new): embeds the question and searches
       config.OPENSEARCH_FEEDBACK_INDEX for the nearest past
       thumbs-upped question. If it's a >=90%-similar match (see
       config.FEEDBACK_SIMILARITY_THRESHOLD), that stored answer is
       returned immediately - retrieval and generation are skipped
       entirely for this request. See
       app/vectordb/opensearch_client.py's search_nearest_feedback() for
       exactly how that score is computed and why it's a real 0-1
       "% similar" number, not an arbitrary threshold.
    4. On a MISS (no past feedback close enough, or none stored yet),
       falls through to the same flow as before: builds the STARTING
       state for our LangGraph graph, calls rag_graph.invoke(...), which
       runs:
           START -> retrieve -> detect_conflict -> generate_single OR
           generate_conflict -> END
       and returns the FINAL state, with `retrieved_chunks`, `answer`,
       and `answeroption` now filled in.
    5. Returns BOTH `answer` and `answeroption` as the JSON response -
       `answeroption` is {} unless detect_conflict_node identified a
       genuine conflict between sources (or this was served from the
       feedback cache, which never has an `answeroption`), in which case
       generate_conflict_node populated it (see
       app/graph/nodes/rag_nodes.py and app/prompts/system_prompt.py).
    """
    # uuid.uuid4() generates a random unique ID; we only keep the first
    # 8 characters since that's plenty to tell requests apart in logs
    # and is much easier to read than the full 36-character UUID.
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] Received question: {request.question!r}")

    query_embedding = get_query_embedding(request.question)

    feedback_match = search_nearest_feedback(query_embedding, request_id)
    if feedback_match is not None:
        duration = time.time() - start_time
        logger.info(
            f"[{request_id}] Served from feedback cache (score="
            f"{feedback_match['score']:.4f}) in {duration:.2f}s - skipped "
            f"retrieval + generation"
        )
        return AskResponse(answer=feedback_match["answer"], answeroption={})

    initial_state = {
        "question": request.question,
        "request_id": request_id,
        "retrieved_chunks": [],
        "conflict_detected": False,
        "answer": {},
        "answeroption": {},
    }

    try:
        final_state = rag_graph.invoke(initial_state)
    except Exception:
        # logger.exception() logs the full error/traceback for us to
        # debug server-side, WITHOUT exposing it to whoever called the
        # API - FastAPI will still return its own generic 500 response.
        duration = time.time() - start_time
        logger.exception(f"[{request_id}] Request FAILED after {duration:.2f}s")
        raise

    duration = time.time() - start_time
    logger.info(f"[{request_id}] Request completed successfully in {duration:.2f}s")

    return AskResponse(
        answer=final_state["answer"],
        answeroption=final_state.get("answeroption", {}),
    )
