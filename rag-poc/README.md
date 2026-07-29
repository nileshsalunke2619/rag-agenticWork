# RAG POC — Pipeline 2 (Retrieval + Generation)

A minimal, beginner-friendly Proof of Concept that answers questions using
**Retrieval-Augmented Generation (RAG)**: it retrieves relevant chunks
from AWS OpenSearch and asks Claude to answer using only that retrieved
context.

This is **Pipeline 2** of a two-pipeline project:

- **Pipeline 1** (already built, not part of this repo): ingests
  documents, splits them into chunks, generates Amazon Titan
  Embeddings for each chunk, and stores them in an AWS OpenSearch
  vector index.
- **Pipeline 2** (this repo): takes a user's question, retrieves the
  most relevant chunks from that same OpenSearch index, and asks
  Claude Sonnet 4.6 (via AWS Bedrock) to generate an answer grounded in
  those chunks.

This project is intentionally **simple**: no authentication, no
streaming, no caching, no Docker, no multi-agent orchestration. It is
meant for learning and demoing, not production use.

---

## 1. Project Overview

```
User Question
     |
     v
FastAPI endpoint (POST /ask)
     |
     v
LangGraph starts execution
     |
     v
Node 1: Retrieve top-3 relevant chunks from AWS OpenSearch
     |
     v
Node 2: Send question + chunks to Claude Sonnet 4.6
     |
     v
Claude generates the final answer
     |
     v
Return { "answer": "..." } as JSON
```

---

## 2. Folder Structure

```
rag-poc/
├── app.py                 # FastAPI app + the /ask endpoint
├── graph.py                # Builds the LangGraph graph (retrieve -> generate)
├── nodes.py                 # Defines the State + the two node functions
├── opensearch_client.py     # All AWS OpenSearch + Titan Embeddings calls
├── llm.py                    # All calls to Claude (prompt building + API call)
├── config.py                 # Loads all settings from environment variables
├── requirements.txt          # Python dependencies
├── .env.example               # Template for your local .env file
└── README.md                  # This file
```

---

## 3. Installation

**Prerequisites:** Python 3.10+, and an AWS account with access to
OpenSearch and Bedrock (Titan Embeddings AND Claude — this project calls
Claude through Bedrock, so there's no separate Anthropic API key to set
up). Make sure Claude Sonnet 4.6 model access is enabled in the Bedrock
console for your account/region before running this.

```bash
# 1. Move into the project folder
cd rag-poc

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file from the template
cp .env.example .env
# then open .env and fill in your real values
```

---

## 4. Required Python Packages

Installed automatically via `requirements.txt`:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework that exposes the `/ask` endpoint |
| `uvicorn[standard]` | Server that actually runs the FastAPI app |
| `langgraph` | Orchestrates the retrieve -> generate flow as a graph |
| `boto3` | AWS SDK — used to call Bedrock (Titan Embeddings AND Claude, both via `invoke_model()`) |
| `opensearch-py` | Official client for querying AWS OpenSearch |
| `python-dotenv` | Loads `.env` file contents into environment variables |

---

## 5. Environment Variables

Set these in a `.env` file in the project root (see `.env.example`):

| Variable | Description |
|---|---|
| `TITAN_REGION` | AWS region where your Bedrock Titan Embeddings model runs |
| `OPENSEARCH_REGION` | AWS region where your OpenSearch collection/domain lives |
| `BEDROCK_REGION` | AWS region where your Claude model/inference profile lives — must match the region inside your `ANTHROPIC_MODEL` value |
| `OPENSEARCH_HOST` | OpenSearch endpoint, without `https://` |
| `OPENSEARCH_PORT` | Usually `443` |
| `OPENSEARCH_INDEX` | Name of the index Pipeline 1 wrote chunks/embeddings into |
| `OPENSEARCH_SERVICE` | `aoss` for OpenSearch **Serverless**, `es` for a managed OpenSearch **Service** domain — see note below |
| `VECTOR_FIELD_NAME` | Field name holding the embedding vector in each document |
| `TEXT_FIELD_NAME` | Field name holding the original chunk text in each document |
| `TITAN_EMBEDDING_MODEL_ID` | Titan Embeddings model ID — must match what Pipeline 1 used |
| `ANTHROPIC_MODEL` | Claude model identifier for Bedrock's `invoke_model()` — **account/region-specific, no default.** May be a short name or a full inference profile ARN (see below). |
| `TOP_K` | How many chunks to retrieve per question (default `3`) |

**AWS credentials:** this app authenticates to AWS OpenSearch, Titan
Embeddings, AND Claude (via Bedrock) using boto3's standard credential
lookup (environment variables, an AWS CLI profile, or an IAM role) —
there is no separate Anthropic API key anywhere in `.env`. If
`aws sts get-caller-identity` already works in your terminal, this app
will use those same credentials for everything.

**OpenSearch Serverless vs. OpenSearch Service (`OPENSEARCH_SERVICE`):**
AWS has two different OpenSearch offerings that share a name but sign
requests differently:
- **OpenSearch Serverless** (a "collection", hostname contains `.aoss.`) → `OPENSEARCH_SERVICE=aoss`
- **OpenSearch Service** (a managed "domain", hostname contains `.es.`) → `OPENSEARCH_SERVICE=es`

Signing a request with the wrong one produces a request that *looks*
correctly signed but AWS rejects as `403 Forbidden` — with no message
telling you it's a service-name mismatch. Check your `OPENSEARCH_HOST`
for `.aoss.` vs `.es.` in the hostname to know which one you have. This
project defaults to `aoss` since that's what it was built against.

**Different AWS resources can live in different regions
(`TITAN_REGION` / `OPENSEARCH_REGION` / `BEDROCK_REGION`):** don't
assume one `AWS_REGION` covers everything — that assumption caused a
long-running `403 Forbidden` bug in this project. Titan Embeddings,
OpenSearch, and Claude can each be provisioned in a different AWS
region, and every AWS call in this project must be signed/routed using
the **correct region for that specific resource**, not a shared
default. Confirm all three values with whoever manages your AWS
account before assuming the defaults in `.env.example` are right for
you.

**Finding the correct `ANTHROPIC_MODEL` value:** unlike the other
settings, there's no safe default we can ship — Bedrock model
identifiers are specific to your AWS account/region, and some accounts
require a full **cross-region inference profile ARN** rather than a
short model name. Example of what an ARN-style value looks like:

```
ANTHROPIC_MODEL=arn:aws:bedrock:ap-southeast-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-6
```

Ask your AWS admin/team for the exact value, or find it yourself:

```bash
aws bedrock list-foundation-models --region <region> --by-provider anthropic \
  --query "modelSummaries[].modelId" --output table

aws bedrock list-inference-profiles --region <region> \
  --query "inferenceProfileSummaries[].inferenceProfileId" --output table
```

Your AWS identity also needs `bedrock:InvokeModel` IAM permission, and
Claude Sonnet 4.6 model access must be explicitly enabled in the
Bedrock console (Bedrock → Model access) — it's off by default per
account.

---

## 6. How to Start the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- `app:app` means "in the file `app.py`, use the object named `app`".
- `--host 0.0.0.0` makes the server reachable from OTHER machines on
  your network (not just this one) — leave this off (or use the default
  `127.0.0.1`) if you only need to test locally on this machine.
- `--reload` restarts the server automatically whenever you edit a file
  (handy while learning/demoing — remove it for anything long-running).

You should see output like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Interactive API docs are automatically available at:
`http://127.0.0.1:8000/docs` (or `http://<this-machine's-IP>:8000/docs`
from another machine).

---

## 7. How to Call the API (curl)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Generative AI?"}'
```

### Expected Output

```json
{
  "answer": "Generative AI refers to artificial intelligence systems that can create new content..."
}
```

If OpenSearch has no relevant chunks for the question, Claude is
instructed (via the system prompt) to respond with:

```json
{
  "answer": "I don't have enough information."
}
```

---

## 8. Complete Execution Flow (Request to Response)

1. **You send a request**: `POST /api/v1/chat/query` with `{"question": "..."}`.
2. **FastAPI validates the request** using the `AskRequest` Pydantic
   model in `app.py` — a missing `question` field is rejected
   automatically.
3. **`app.py` builds the starting state** for LangGraph:
   `{"question": "...", "retrieved_chunks": [], "answer": ""}`.
4. **LangGraph starts running** (`rag_graph.invoke(...)`), beginning at
   `START`.
5. **Node 1 — `retrieve_node`** (in `nodes.py`) runs first:
   - Calls `opensearch_client.get_query_embedding(question)`, which
     sends the question to Amazon Titan Embeddings (via Bedrock) and
     gets back a vector.
   - Calls `opensearch_client.search_top_chunks(...)`, which sends a
     k-NN search to AWS OpenSearch and gets back the top 3 most
     similar chunks.
   - Writes those chunks into the state as `retrieved_chunks`.
6. **Node 2 — `generate_node`** (in `nodes.py`) runs next:
   - Calls `llm.ask_claude(question, chunks)`, which builds a prompt
     (system instructions + context + question) and sends it to
     Claude Sonnet 4.6 via AWS Bedrock.
   - Writes Claude's reply into the state as `answer`.
7. **LangGraph reaches `END`** and returns the final state to `app.py`.
8. **`app.py` extracts `final_state["answer"]`** and returns it as
   `{"answer": "..."}` — the JSON response you receive back from curl.

---

## 9. Important Notes / Limitations (by design)

This is a POC, so several things are intentionally left out:

- No authentication on the `/api/v1/chat/query` endpoint
- No streaming responses
- No conversation memory (every question is independent)
- No hybrid search or re-ranking — just a plain top-3 vector search
- No retries, caching, or production monitoring
- `VECTOR_FIELD_NAME` / `TEXT_FIELD_NAME` / `TITAN_EMBEDDING_MODEL_ID`
  **must exactly match** whatever Pipeline 1 used when it built the
  OpenSearch index — a mismatch here is the most common source of
  "no results" or dimension errors.
- `OPENSEARCH_SERVICE` **must match your actual OpenSearch type**
  (`aoss` for Serverless, `es` for a managed Service domain) — getting
  this wrong produces a `403 Forbidden` that looks like a permissions
  problem but is actually a signing mismatch (see §5 above).
- `TITAN_REGION` / `OPENSEARCH_REGION` / `BEDROCK_REGION` **must each be
  set to the correct region for that specific resource** — they are not
  guaranteed to be the same, and using one shared region for all three
  is what caused this project's original, hard-to-diagnose `403
  Forbidden` bug (see §5 above).
