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


class FeedbackRequest(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON body we expect on
    POST /agent/v1/feedback.

    Example valid request body:
        {
          "userid": "akhilesh.kumar2@pfizer.com",
          "question": "what is order creation",
          "questionid": "1785603829793_akhilesh.kumar2@pfizer.com",
          "feedback": "thumsup",
          "responseJSON": {"description": "...", "references": ["SOP-124627"]}
        }

    WHY `feedback` IS `str` AND NOT VALIDATED AGAINST A FIXED SET HERE:
    the Java/API layer sends one of "thumsup"/"down"/"notrelavent"/
    "outdated" (spelling as given by that team) - we only ever ACT on
    "thumsup" (see app/api/routes/feedback.py), so there's no need to
    reject the other values at this layer; they're accepted and simply
    not stored.
    """

    userid: str
    question: str
    questionid: str
    feedback: str
    responseJSON: dict


class FeedbackResponse(BaseModel):
    """
    WHAT THIS IS:
    The JSON we send back from POST /agent/v1/feedback - just enough to
    confirm whether this feedback was actually stored into the feedback
    index or acknowledged-but-ignored (any feedback value other than
    "thumsup" - see app/config/settings.py, we only capture POSITIVE
    feedback).
    """

    status: str
