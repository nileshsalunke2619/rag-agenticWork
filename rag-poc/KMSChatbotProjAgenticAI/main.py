# =============================================================================
# main.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the entry point of the whole project - the file you actually
# run to start the API server. In the old flat structure, this used to
# be called "app.py" - it's now "main.py" because there's a FOLDER
# called "app/" in this same directory, and Python cannot have both a
# module named "app.py" and a package named "app/" in the same place
# without ambiguity. "main.py" is also the standard, conventional name
# FastAPI projects use for this file.
#
# All this file does is:
#   1. Create the FastAPI application
#   2. "Mount" our chat router (app/api/routes/chat.py) onto it
#
# Run this file with:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# (Note: "main:app", not "app:app" like before - see above.)
# =============================================================================

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router

# Create the FastAPI application. This "app" object is what uvicorn looks
# for when you run `uvicorn main:app`.
app = FastAPI(
    title="RAG POC - Pipeline 2 (Retrieval + Generation)",
    description="A minimal FastAPI + LangGraph service that answers "
    "questions using AWS OpenSearch retrieval and Claude generation.",
)

# include_router() takes every endpoint defined on chat_router (just
# POST /internal/v1/orchestrate/chat for now) and adds it to our main app. This is
# the standard way to keep route definitions in their own files/folders
# while still ending up with one single running application.
app.include_router(chat_router)
