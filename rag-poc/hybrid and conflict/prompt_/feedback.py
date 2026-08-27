# =============================================================================
# app/api/routes/feedback.py
# =============================================================================
# WHY THIS FILE EXISTS:
# Defines POST /agent/v1/feedback - a SEPARATE endpoint from
# app/api/routes/chat.py's /agent/v1/orchestration/chat, not the same
# route reused for two purposes. Kept separate because the two payload
# shapes are genuinely different (a question vs. a full feedback record
# with userid/questionid/feedback/responseJSON), so branching on payload
# shape inside one handler would be more fragile than just having two
# clearly-named endpoints.
# =============================================================================

import logging
import uuid

from fastapi import APIRouter

from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.vectordb.opensearch_client import get_query_embedding, store_feedback

logger = logging.getLogger(__name__)

router = APIRouter()

# We only ever ACT on this one feedback value - see app/models/schemas.py's
# FeedbackRequest docstring for why the other values (down/notrelavent/
# outdated) are accepted but not validated against a fixed set.
POSITIVE_FEEDBACK_VALUE = "thumsup"


@router.post("/agent/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Generates a request_id for tracing, same convention as
       app/api/routes/chat.py.
    2. We ONLY capture POSITIVE feedback - if request.feedback is
       anything other than "thumsup" (down/notrelavent/outdated), we
       acknowledge the call but store nothing, and return early.
    3. On a thumsup: embeds request.question (the SAME
       get_query_embedding() the chat endpoint uses, so future kNN
       matching compares like-for-like vectors) and stores it, along
       with the full feedback record, into
       config.OPENSEARCH_FEEDBACK_INDEX via store_feedback().
    """
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        f"[{request_id}] [feedback] Received feedback={request.feedback!r} for "
        f"questionid={request.questionid!r}"
    )

    if request.feedback != POSITIVE_FEEDBACK_VALUE:
        logger.info(
            f"[{request_id}] [feedback] feedback={request.feedback!r} is not "
            f"positive - not stored"
        )
        return FeedbackResponse(status="ignored")

    query_embedding = get_query_embedding(request.question)

    store_feedback(
        question=request.question,
        query_embedding=query_embedding,
        payload=request.model_dump(),
        request_id=request_id,
    )

    return FeedbackResponse(status="stored")
