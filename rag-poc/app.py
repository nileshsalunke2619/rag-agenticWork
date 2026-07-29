# =============================================================================
# app.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the entry point of the whole project - the file you actually
# run to start the API server. It defines a single FastAPI endpoint,
# POST /api/v1/chat/query, which:
#   1. Receives a JSON question from the user
#   2. Runs it through our LangGraph pipeline (retrieve -> generate)
#   3. Returns the final answer as JSON
#
# Run this file with (--host 0.0.0.0 makes it reachable from OTHER
# machines on the network, not just this one - see README for details):
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel

from graph import rag_graph

# Create the FastAPI application. This "app" object is what uvicorn looks
# for when you run `uvicorn app:app`.
app = FastAPI(
    title="RAG POC - Pipeline 2 (Retrieval + Generation)",
    description="A minimal FastAPI + LangGraph service that answers "
    "questions using AWS OpenSearch retrieval and Claude generation.",
)


class AskRequest(BaseModel):
    """
    WHAT THIS IS:
    A Pydantic model describing the JSON body we expect on
    POST /api/v1/chat/query.

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

    WHY `answer` IS A `dict` HERE:
    llm.py's ask_claude() runs Claude's raw reply through
    utils/json_normalizer_dynamic.py's get_json_string() to clean it up,
    then parses that result back into a real Python dict and unwraps it
    once, so `answer` here is already a well-structured, single-nested
    object - description, header, question_id, references, steps (if
    present), subheader - not text the caller has to parse themselves.

    Example response body:
        {
          "answer": {
            "description": "...",
            "header": "...",
            "question_id": "",
            "references": ["SOP-124627"],
            "steps": {"step_1": "..."},
            "subheader": "..."
          }
        }
    """

    answer: dict


@app.post("/api/v1/chat/query", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    WHAT THIS ENDPOINT DOES:
    1. Takes the user's question from the request body.
    2. Builds the STARTING state for our LangGraph graph - at this point
       we only know the question; `retrieved_chunks` and `answer` are
       still empty and will be filled in as the graph runs.
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

    # .invoke() runs the entire graph from START to END and gives us
    # back the final state once every node has finished.
    final_state = rag_graph.invoke(initial_state)

    return AskResponse(answer=final_state["answer"])
