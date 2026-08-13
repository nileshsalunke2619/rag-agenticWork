# =============================================================================
# app/api/routes/chat.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This defines the actual POST /internal/v1/orchestrate/chat endpoint. It's an
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

logger = logging.getLogger(__name__)

# APIRouter works like a mini FastAPI app - you define routes on it,
# then main.py "mounts" it onto the real app with app.include_router(...).
router = APIRouter()


@router.post("/internal/v1/orchestrate/chat", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Generates a short request_id for this call, so every log line
       from here through retrieve_node/generate_node can be traced back
       to this ONE request - useful once multiple users are hitting the
       API at the same time.
    2. Takes the user's question from the request body.
    3. Builds the STARTING state for our LangGraph graph.
    4. Calls rag_graph.invoke(...), which runs:
           START -> retrieve -> generate -> END
       and returns the FINAL state, with `retrieved_chunks`, `answer`,
       and `answeroption` now filled in.
    5. Returns BOTH `answer` and `answeroption` as the JSON response -
       `answeroption` is {} unless a genuine conflict between sources
       was identified (see app/prompts/system_prompt.py's "MULTI-SOURCE
       SYNTHESIS, CONFLICT DETECTION" section).
    """
    # uuid.uuid4() generates a random unique ID; we only keep the first
    # 8 characters since that's plenty to tell requests apart in logs
    # and is much easier to read than the full 36-character UUID.
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] Received question: {request.question!r}")

    initial_state = {
        "question": request.question,
        "request_id": request_id,
        "retrieved_chunks": [],
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
