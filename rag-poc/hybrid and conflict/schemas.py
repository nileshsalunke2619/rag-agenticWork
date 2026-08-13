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

    WHY `answer` AND `answeroption` ARE BOTH `dict` HERE:
    app/llm/bedrock_client.py's ask_claude() parses Claude's raw JSON
    reply and returns BOTH top-level objects it can contain - "answer"
    (always populated) and "answeroption" (only populated when Claude
    identifies a genuine conflict between two sources - {} otherwise).
    Pydantic response models silently drop any field not declared here,
    so `answeroption` MUST be declared explicitly, or it would never
    reach the actual HTTP response even if every other layer worked.

    Example response body (no conflict):
        {
          "answer": {
            "description": "...",
            "header": "...",
            "subheader": "...",
            "references": ["SOP-124627"],
            "followupquestions": [],
            "steps": []
          },
          "answeroption": {}
        }

    Example response body (genuine conflict between two sources):
        {
          "answer": {
            "description": "...",
            "references": ["SOP-104302"],
            ...
          },
          "answeroption": {
            "description": "...",
            "references": ["SOP-129268"],
            ...
          }
        }
    """

    answer: dict
    answeroption: dict = {}
