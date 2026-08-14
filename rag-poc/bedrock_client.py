# =============================================================================
# app/llm/bedrock_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to Claude (Anthropic's LLM). It takes
# the user's question + the chunks we retrieved from OpenSearch, builds
# a prompt, sends it to Claude, and returns Claude's answer as a clean,
# normalized dict - ready to hand straight back to the API caller.
#
# NOTE ON AUTHENTICATION:
# We call Claude through AMAZON BEDROCK's raw invoke_model() API - the
# exact same low-level pattern used for Titan Embeddings in
# app/vectordb/opensearch_client.py. This means there is NO separate
# ANTHROPIC_API_KEY anywhere in this project - Claude is authenticated
# using the SAME AWS credentials that app/vectordb/opensearch_client.py
# already uses for OpenSearch and Titan Embeddings.
#
# NOTE ON RESPONSE FORMAT:
# The system prompt (app/prompts/system_prompt.py) instructs Claude to
# return ONLY a raw JSON object shaped like {"answer": {...fields...}},
# with no markdown fences and no extra text. So we parse Claude's raw
# text directly with json.loads() - no normalizer utility involved.
# =============================================================================

import json
import logging
import time

import boto3

# "as config" keeps every reference below (config.BEDROCK_REGION, etc.)
# identical to how this file worked before the folder reorganization.
from app.config import settings as config
from app.prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# We create ONE Bedrock Runtime client when this file is first imported,
# and reuse it for every request.
#
# NOTE: config.BEDROCK_REGION, not TITAN_REGION or OPENSEARCH_REGION -
# Claude's model/inference-profile ARN lives in its own AWS region in
# this project, which may differ from where Titan Embeddings or
# OpenSearch live.
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)


def _format_chunk_with_source(chunk: dict) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Attaches a "[Source - ...]" label in front of a chunk's text, built
    from its citation metadata (document_name/section/page from
    app/vectordb/opensearch_client.py). This is what lets Claude cite a
    REAL source in its "references" field instead of guessing one from
    the chunk's raw body text - see the "SOURCE LABELS AND CITATIONS"
    section of app/prompts/system_prompt.py, which expects EXACTLY this
    "[Source - document: ..., section: ..., page: ...]" format.

    Any piece of metadata Pipeline 1 didn't set on a given chunk (e.g. no
    section) is simply left out of that chunk's label rather than shown
    as blank/missing.
    """
    source_parts = []
    if chunk.get("document_name"):
        source_parts.append(f'document: "{chunk["document_name"]}"')
    if chunk.get("section"):
        source_parts.append(f'section: "{chunk["section"]}"')
    if chunk.get("page"):
        source_parts.append(f'page: {chunk["page"]}')

    if source_parts:
        source_label = "[Source - " + ", ".join(source_parts) + "]"
    else:
        source_label = "[Source - unknown]"

    return f"{source_label}\n{chunk.get('content', '')}"


def build_user_prompt(question: str, chunks: list) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Combines the retrieved chunks and the user's question into a single
    block of text, in the exact "Context / Question" shape Claude
    expects based on our system prompt. Each chunk is labeled with its
    source (see _format_chunk_with_source above) before being joined -
    `chunks` here is the list of dicts returned by
    opensearch_client.search_top_chunks(), not plain strings.
    """
    if chunks:
        context_text = "\n\n".join(_format_chunk_with_source(chunk) for chunk in chunks)
    else:
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
    """
    user_prompt = build_user_prompt(question, chunks)

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            # NOTE: do NOT add "temperature" here. This model/inference
            # profile's Bedrock validation hard-rejects it:
            #   ValidationException: `temperature` is deprecated for
            #   this model.
            # We tried pinning it to reduce wording variance across
            # identical questions and it broke every request with a 500.
            # Consistency has to come from the retrieval side
            # (config.MIN_ABSOLUTE_SCORE / RELEVANCE_SCORE_RATIO) and
            # prompt wording instead, not sampling params, for this model.
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
    )

    start_time = time.time()
    response = bedrock_runtime.invoke_model(
        # config.ANTHROPIC_MODEL holds whatever your AWS account actually
        # needs here - this might be a short model name OR a full
        # "inference profile" ARN, depending on your account/region.
        modelId=config.ANTHROPIC_MODEL,
        body=request_body,
        contentType="application/json",
    )
    duration = time.time() - start_time

    response_body = json.loads(response["body"].read())

    # stop_reason tells us WHY Claude stopped generating. If this is
    # "max_tokens", the answer got cut off before finishing - this is
    # exactly what caused the "Unterminated string" JSON errors we hit
    # before raising max_tokens. Logging it every time means we catch
    # that happening again immediately, instead of debugging it blind.
    stop_reason = response_body.get("stop_reason")
    logger.info(
        f"[bedrock_client] Claude call took {duration:.2f}s, stop_reason={stop_reason!r}"
    )
    if stop_reason == "max_tokens":
        logger.warning(
            "[bedrock_client] Claude's response was CUT OFF (stop_reason=max_tokens) - "
            "the answer may be truncated/invalid JSON. Consider raising max_tokens."
        )

    raw_text = ""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            raw_text = block["text"]
            break

    return raw_text


def _filter_hallucinated_references(answer: dict, chunks: list) -> None:
    """
    WHAT THIS FUNCTION DOES:
    A deterministic backstop against Claude fabricating a citation
    despite the system prompt's rules - modifies answer["references"]
    IN PLACE, keeping only entries that actually start with a real
    document name from the chunks retrieved for THIS request. Prompt
    instructions alone are necessary but not sufficient - a model can
    still slip - so this makes it structurally impossible for a made-up
    reference to reach the caller, regardless of whether Claude followed
    the prompt correctly. Never makes "references" LONGER or changes its
    shape - only ever removes entries that don't check out.
    """
    # Chunks with no document_name at all can't back up ANY reference -
    # excluding them (rather than including "") stops an empty string
    # from matching (and passing) every reference via startswith("").
    known_document_names = {
        chunk["document_name"] for chunk in chunks if chunk.get("document_name")
    }

    original_references = answer.get("references", [])
    kept_references = [
        reference
        for reference in original_references
        if any(reference.startswith(name) for name in known_document_names)
    ]

    dropped_count = len(original_references) - len(kept_references)
    if dropped_count:
        logger.warning(
            f"[bedrock_client] Dropped {dropped_count} reference(s) that did NOT "
            f"match any actually-retrieved document - Claude likely fabricated "
            f"{'them' if dropped_count > 1 else 'it'} despite the prompt's rules. "
            f"Original references: {original_references!r}"
        )

    answer["references"] = kept_references


def _deduplicate_references(answer: dict) -> None:
    """
    WHAT THIS FUNCTION DOES:
    Removes exact-duplicate entries from answer["references"] IN PLACE,
    keeping the first occurrence and preserving order. This is a
    deterministic backstop alongside _filter_hallucinated_references -
    the prompt already instructs Claude to cite each document only once
    (see SOURCE LABELS AND CITATIONS), but that is a request, not a
    guarantee, the same reason the hallucination filter above exists
    independent of the prompt's "don't fabricate" rule.
    """
    references = answer.get("references", [])

    seen = set()
    deduplicated = []
    for reference in references:
        if reference not in seen:
            seen.add(reference)
            deduplicated.append(reference)

    if len(deduplicated) != len(references):
        logger.info(
            f"[bedrock_client] Removed {len(references) - len(deduplicated)} "
            f"duplicate reference(s) - original: {references!r}"
        )

    answer["references"] = deduplicated


def ask_claude(question: str, chunks: list) -> tuple:
    """
    WHAT THIS FUNCTION DOES:
    Gets Claude's raw answer (via get_raw_claude_text above), parses it
    as JSON, strips out any hallucinated reference from BOTH "answer"
    and "answeroption" (see _filter_hallucinated_references above), and
    returns a (answer, answeroption) tuple - answeroption is {} unless
    Claude populated a genuine Case 2 conflict (see the system prompt's
    "MULTI-SOURCE SYNTHESIS, CONFLICT DETECTION" section).

    WHY THIS RETURNS TWO VALUES NOW:
    The system prompt's JSON contract has "answer" and "answeroption" as
    TWO top-level siblings, not one. Earlier this function only ever
    returned parsed["answer"] and silently discarded everything else -
    which meant even a perfectly-generated "answeroption" could never
    reach the caller. Returning both explicitly is what lets
    app/graph/nodes/rag_nodes.py carry both through graph state, and
    app/api/routes/chat.py put both in the final API response.
    """
    raw_text = get_raw_claude_text(question, chunks)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Logging the raw text here is what lets us actually see WHAT
        # Claude sent back when parsing fails, instead of just getting
        # an "Unterminated string" error with no context.
        logger.error(
            f"[bedrock_client] Failed to parse Claude's response as JSON. "
            f"Raw text was:\n{raw_text}"
        )
        raise

    answer = parsed["answer"]
    _filter_hallucinated_references(answer, chunks)
    _deduplicate_references(answer)

    # answeroption is OPTIONAL in Claude's response - {} (no conflict) is
    # the normal case, so we default to {} rather than requiring the key.
    answeroption = parsed.get("answeroption") or {}
    if answeroption:
        _filter_hallucinated_references(answeroption, chunks)
        _deduplicate_references(answeroption)

    return answer, answeroption
