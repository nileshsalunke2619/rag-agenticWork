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

from fastapi import APIRouter

from app.graph.graph import rag_graph
from app.models.schemas import AskRequest, AskResponse

# APIRouter works like a mini FastAPI app - you define routes on it,
# then main.py "mounts" it onto the real app with app.include_router(...).
router = APIRouter()


@router.post("/internal/v1/orchestrate/chat", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Takes the user's question from the request body.
    2. Builds the STARTING state for our LangGraph graph.
    3. Calls rag_graph.invoke(...), which runs:
           START -> retrieve -> generate -> END
       and returns the FINAL state, with `retrieved_chunks` and `answer`
       now filled in.
    4. Returns just the `answer` field as the JSON response.
    """
    initial_state = {
        "question": request.question,
        "retrieved_chunks": [],
        "answer": {},
    }

    final_state = rag_graph.invoke(initial_state)

    return AskResponse(answer=final_state["answer"])
