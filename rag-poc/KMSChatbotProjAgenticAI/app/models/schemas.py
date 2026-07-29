# =============================================================================
# app/models/schemas.py
# =============================================================================
# WHY THIS FILE EXISTS:
# These are the Pydantic models describing the JSON shapes our API
# accepts and returns. Keeping them separate from the route logic
# (app/api/routes/chat.py) means the "what does a request/response look
# like" question has one clear home.
# =============================================================================

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON body we expect on
    POST /internal/v1/orchestrate/chat.

    WHY WE NEED IT:
    FastAPI uses this to automatically validate incoming requests - if
    someone sends a request WITHOUT a "question" field, FastAPI will
    reject it with a clear error before our code ever runs.

    Example valid request body:
        { "question": "What is Generative AI?" }
    """

    question: str


class AskResponse(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON we send back.

    WHY `answer` IS A `dict`, NOT A STRING:
    Claude's raw reply is cleaned up by
    app/utils/json_normalizer_dynamic.py's normalize_response() before it
    ever reaches this model, so `answer` is already a well-structured
    dict (description, header, question_id, references, steps,
    subheader) - not text the caller has to parse themselves.

    Example response body:
        {
          "answer": {
            "description": "...",
            "header": "...",
            "question_id": "...",
            "references": ["SOP-124627"],
            "steps": {"step_1": "..."},
            "subheader": "..."
          }
        }
    """

    answer: dict
