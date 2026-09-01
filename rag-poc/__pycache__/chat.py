# =============================================================================
# app/api/routes/chat.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This defines the actual POST /agent/v1/orchestration/chat endpoint. It's
# an APIRouter (not a full FastAPI app) so that main.py (the root entry
# point) can "include" it into the real app - this is the standard
# modular FastAPI pattern: routers define endpoints, main.py assembles
# them into one application.
# =============================================================================

import json
from fastapi import APIRouter

from app.utils.logger import get_logger
import time
import uuid
from app.graph.graph import rag_graph
from app.models.schemas import AskRequest, AskResponse
from app.vectordb.opensearch_client import get_query_embedding, search_nearest_feedback

# query reframing
from app.llm import bedrock_client as llm

# feedback handling (owned/maintained separately - see
# app/ingestion/feedback_ingest.py and app/tools/feedback_opensearch_client.py -
# not touched by this file beyond wiring the route below)
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.models.feedback_request_model import IngestRequest
from app.ingestion.feedback_ingest import ingest_question


logger = get_logger(__name__)


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
    3. QUERY REFRAMING: if request.lastquestion is non-empty and
       different from the current question, calls llm.reframe_query() to
       rewrite the question into a standalone one BEFORE retrieval -
       e.g. "what will be more steps here" (with lastquestion "How to
       create sales order") becomes "What are the remaining steps to
       create a sales order?". NOTE: this deliberately does NOT gate on
       request.isfollowup anymore (removed per lead's instruction) - it
       now runs on EVERY request that has a lastquestion different from
       the current question, whether or not the caller flags it as a
       follow-up. QUERY_REFRAMING_PROMPT's own rule 3 ("if already
       self-contained, return unchanged") is the only thing preventing a
       genuinely unrelated question from being rewritten - there is no
       longer a deterministic gate backing that up. Every step after
       this uses the REFRAMED question, not the raw one. This is fully
       fail-safe regardless: if reframing fails or returns something
       unusable, reframe_query() itself falls back to the original
       question - this endpoint never has to handle that case specially.
       See app/llm/bedrock_client.py's reframe_query() and
       app/prompts/system_prompt.py's QUERY_REFRAMING_PROMPT.
    4. FEEDBACK CACHE CHECK: embeds the (possibly reframed) question and
       searches config.OPENSEARCH_FEEDBACK_INDEX for the nearest past
       thumbs-upped question. On a >=threshold match, returns that
       stored answer immediately - retrieval and generation are skipped
       entirely. See app/vectordb/opensearch_client.py's
       search_nearest_feedback().
    5. On a MISS, falls through to the normal flow: builds the STARTING
       state for our LangGraph graph, calls rag_graph.invoke(...), which
       runs:
           START -> retrieve -> detect_conflict -> generate_single OR
           generate_conflict -> END
       and returns the FINAL state, with `retrieved_chunks`, `answer`,
       and `answeroption` now filled in.
    6. Returns BOTH `answer` and `answeroption` as the JSON response -
       `answeroption` is {} unless a genuine conflict between sources
       was identified (see app/prompts/system_prompt.py's "MULTI-SOURCE
       SYNTHESIS, CONFLICT DETECTION" section).
    """
    # uuid.uuid4() generates a random unique ID; we only keep the first
    # 8 characters since that's plenty to tell requests apart in logs
    # and is much easier to read than the full 36-character UUID.
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"[{request_id}] Received question: {request.question!r}, "
        f"lastquestion: {request.lastquestion!r}, isfollowup: {request.isfollowup!r}"
    )

    # QUERY REFRAMING - runs whenever a lastquestion was sent AND it
    # differs from the current question (isfollowup is no longer checked
    # here, per lead's instruction). No lastquestion, or lastquestion
    # identical to the current question, skips straight past this and
    # uses the question as-is - nothing to reframe against either way.
    #
    # LOGGED UNCONDITIONALLY (both branches) - this is the ONE line that
    # proves this code path was even reached at all. If this line is
    # missing from the logs entirely, the request never made it this far
    # into ask() (a routing/startup/deployment problem upstream of this
    # function), which is a completely different problem than reframe
    # itself silently doing nothing.
    effective_question = request.question
    if request.lastquestion.strip() and request.lastquestion.strip() != request.question.strip():
        logger.info(f"[{request_id}] [reframe] Gate PASSED - calling reframe_query()")
        effective_question = llm.reframe_query(request.question, request.lastquestion, request_id)
        logger.info(
            f"[{request_id}] [reframe] Result - original: {request.question!r} -> "
            f"effective: {effective_question!r}"
        )
    else:
        reason = "lastquestion is empty" if not request.lastquestion.strip() else "lastquestion == question"
        logger.info(f"[{request_id}] [reframe] Gate SKIPPED ({reason}) - using original question")

    query_embedding = get_query_embedding(effective_question)

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
        "question": effective_question,
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


@router.post("/agent/v1/feedbackconflicthandling")
async def ingest(req: IngestRequest):

    logger.info(
        f"Feedback request received. "
        f"question_id={req.questionid}, "
        f"user_id={req.userid}"
    )

    try:

        result = ingest_question(
            request=req
        )

        safe_result = json.loads(
            json.dumps(
                result,
                ensure_ascii=False
            )
        )

        logger.info(
            f"Feedback ingestion successful. "
            f"question_id={req.questionid}"
        )

        return JSONResponse(
            content={
                "status": "success",
                "result": safe_result
            }
        )

    except Exception as e:

        logger.exception(
            f"Feedback-conflict handling failed. "
            f"question_id={req.questionid}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )
