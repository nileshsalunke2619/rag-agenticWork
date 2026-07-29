# =============================================================================
# llm.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to Claude (Anthropic's LLM). It takes
# the user's question + the chunks we retrieved from OpenSearch, builds a
# simple prompt, sends it to Claude, and returns Claude's answer as a
# clean JSON STRING (already filtered/normalized).
#
# Keeping this separate from opensearch_client.py means: if you ever want
# to change the prompt wording, or swap models, you only touch this file.
#
# NOTE ON AUTHENTICATION:
# We call Claude through AMAZON BEDROCK's raw invoke_model() API - the
# exact same low-level pattern used for Titan Embeddings in
# opensearch_client.py, and the same pattern confirmed working in this
# AWS account/region. This means there is NO separate ANTHROPIC_API_KEY
# anywhere in this project - Claude is authenticated using the SAME AWS
# credentials (from your normal AWS setup) that opensearch_client.py
# already uses for OpenSearch and Titan Embeddings. One set of
# credentials for everything.
#
# NOTE ON RESPONSE CLEANUP:
# Claude's raw reply can come back as plain JSON, JSON wrapped in
# markdown code fences (```json ... ```), or other messy variations.
# Rather than handle that ourselves, we hand the raw text to
# utils/json_normalizer_dynamic.py's get_json_string(), which already
# knows how to clean up all of those cases and hand back a ready-to-use
# JSON string. We do NOT modify that file - we only call it.
# =============================================================================

import json

import boto3

import config
from utils.json_normalizer_dynamic import normalize_response, get_json_string

# We create ONE Bedrock Runtime client when this file is first imported,
# and reuse it for every request. This is the same boto3 client type
# used for Titan Embeddings in opensearch_client.py - Bedrock hosts many
# different models (Titan, Claude, etc.) behind this one client, and
# `modelId` tells it which one to actually run.
#
# NOTE: config.BEDROCK_REGION, not TITAN_REGION or OPENSEARCH_REGION -
# Claude's model/inference-profile ARN lives in its own AWS region in
# this project, which may differ from where Titan Embeddings or
# OpenSearch live. This must match the region INSIDE your ANTHROPIC_MODEL
# value (e.g. the "ap-southeast-1" in an inference profile ARN).
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

# This is the "system prompt" - instructions that tell Claude HOW to
# behave, before it even sees the user's question. Keeping Claude
# grounded in "only use the provided context" is what makes this a RAG
# system instead of Claude just answering from its own general knowledge.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only using the provided context. "
    'If the answer is not found, say: "I don\'t have enough information."'
)


def build_user_prompt(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Combines the retrieved chunks and the user's question into a single
    block of text, in the exact "Context / Question" shape Claude expects
    based on our system prompt.

    WHY WE NEED IT:
    Claude doesn't automatically know which chunks we retrieved - we have
    to literally paste them into the message we send it. This function
    just formats that message consistently.
    """
    if chunks:
        # We separate chunks with a blank line so Claude can tell where
        # one chunk ends and the next begins.
        context_text = "\n\n".join(chunks)
    else:
        # If OpenSearch found nothing, we still send a valid prompt -
        # Claude will see there's no context and (per the system prompt)
        # should say it doesn't have enough information.
        context_text = "No context was found for this question."

    return f"Context:\n{context_text}\n\nQuestion:\n{question}"


def get_raw_claude_text(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Sends the question + retrieved chunks to Claude via Bedrock's
    invoke_model() API, and returns Claude's answer EXACTLY as Claude
    wrote it - the raw, un-cleaned-up text, BEFORE it goes anywhere near
    get_json_string(). This might be plain JSON, JSON wrapped in
    markdown fences, or something messier - whatever Claude actually
    sent back.

    WHY THIS IS ITS OWN FUNCTION (separate from ask_claude below):
    Splitting this out lets us (or a test script) inspect exactly what
    Claude sent back, BEFORE normalization, which is useful for
    debugging/testing the normalizer against real Claude output instead
    of a hand-typed sample.
    """
    user_prompt = build_user_prompt(question, chunks)

    # This exact JSON shape is what Bedrock requires for Claude models.
    # A few notes on fields that might look unfamiliar:
    #   - "anthropic_version" is a FIXED constant Bedrock expects on every
    #     request - it is NOT related to which Claude model you're
    #     calling (that's what `modelId` on invoke_model is for below).
    #   - "system" and "messages" are the same shape Claude uses on
    #     Anthropic's own API - Bedrock just wraps them with the extra
    #     "anthropic_version" field.
    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        # config.ANTHROPIC_MODEL holds whatever your AWS account actually
        # needs here - this might be a short model name OR a full
        # "inference profile" ARN, depending on your account/region.
        modelId=config.ANTHROPIC_MODEL,
        body=request_body,
        contentType="application/json",
    )

    # The response body is a stream of bytes containing JSON - we read it
    # and parse it, same as we do for Titan Embeddings.
    response_body = json.loads(response["body"].read())

    # response_body["content"] is a list of "content blocks" - the same
    # shape Claude's Messages API always returns. For a simple text
    # answer (no tools, no thinking), there is normally just one block
    # of type "text" - we grab its text below.
    raw_text = ""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            raw_text = block["text"]
            break

    return raw_text


def ask_claude(question: str, chunks: list) -> dict:
    """
    WHAT THIS FUNCTION DOES:
    Gets Claude's raw answer (via get_raw_claude_text above), then runs
    it through get_json_string() (from utils/json_normalizer_dynamic.py)
    to clean it up, then parses that result back into a Python dict and
    returns just the inner "answer" object - description, header,
    question_id, references, steps (if present), subheader.

    WHY WE PARSE THE STRING BACK INTO A DICT AND UNWRAP IT:
    get_json_string() returns a JSON STRING shaped like
    '{"answer": {...fields...}}'. If we returned that string as-is (or
    even just parsed it without unwrapping), our API's response model
    (which ALSO wraps everything under "answer" - see app.py) would
    double-wrap it into {"answer": {"answer": {...}}}. Parsing it AND
    taking just the inner ["answer"] value here means the final API
    response has exactly ONE "answer" wrapper, matching the expected
    output format.

    WHY WE NEED IT:
    This is the "generation" half of Retrieval-Augmented Generation (RAG).
    Retrieval (opensearch_client.py) found the relevant information;
    this function asks Claude to actually turn that information into an
    answer, then filters that answer through our normalizer before
    handing it back.
    """
    raw_text = get_raw_claude_text(question, chunks)

    # get_json_string() expects {"answer": <raw text from the LLM>} -
    # same input shape as normalize_response() (get_json_string() calls
    # normalize_response() internally, then converts the result to a
    # JSON string). It handles plain JSON, markdown-fenced JSON, escaped
    # JSON, and Python-dict-style strings for us - we don't need to
    # guess which one Claude sent back.
    filtered_json = get_json_string({"answer": raw_text})

    # filtered_json is a STRING - parse it back into a real dict, then
    # unwrap the outer "answer" key (see the docstring above for why).
    parsed = json.loads(filtered_json)

    return parsed["answer"]
