# =============================================================================
# app/llm/bedrock_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to Claude (Anthropic's LLM). It takes
# the user's question + the chunks we retrieved from OpenSearch, builds a
# prompt, sends it to Claude, and returns Claude's answer as a clean,
# normalized dict/tuple - ready to hand straight back to the caller.
#
# NOTE ON AUTHENTICATION:
# We call Claude through AMAZON BEDROCK's raw invoke_model() API - the
# exact same low-level pattern used for Titan Embeddings in
# app/vectordb/opensearch_client.py. This means there is NO separate
# ANTHROPIC_API_KEY anywhere in this project - Claude is authenticated
# using the SAME AWS credentials that app/vectordb/opensearch_client.py
# already uses for OpenSearch and Titan Embeddings.
#
# WHY THREE FUNCTIONS INSTEAD OF ONE ask_claude():
# The old design made ONE Claude call per request, with ONE mega-prompt
# asked to both detect a conflict AND produce the answer(s) in the same
# JSON reply. That's exactly why "answeroption" was unreliable - a
# correctly-detected conflict still depended on Claude remembering to
# ALSO populate a second top-level field inside one blob.
#
# The graph (see app/graph/nodes/rag_nodes.py, app/graph/graph.py) now
# makes conflict handling STRUCTURAL: a dedicated detection call routes
# to one of two independent generation calls, each with a prompt scoped
# to exactly what that call needs to do:
#   - detect_conflict()    -> CONFLICT_DETECTION_PROMPT -> bool
#   - ask_claude_single()  -> SINGLE_ANSWER_PROMPT       -> answer dict
#   - ask_claude_conflict()-> CONFLICT_ANSWER_PROMPT     -> (answer, answeroption)
# See app/prompts/system_prompt_UPDATED.py for the full prompt text and
# reasoning behind the split.
# =============================================================================

import json
import logging
import time

import boto3

# "as config" keeps every reference below (config.BEDROCK_REGION, etc.)
# identical to how this file worked before the folder reorganization.
from app.config import settings as config
from app.prompts.system_prompt import (
    CONFLICT_ANSWER_PROMPT,
    CONFLICT_DETECTION_PROMPT,
    SINGLE_ANSWER_PROMPT,
)

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
    from its citation metadata (document_name/document_id/section/page
    from app/vectordb/opensearch_client.py). This is what lets Claude
    cite a REAL source in "references" and "documentids" instead of
    guessing from the chunk's raw body text - see the "SOURCE LABELS AND
    CITATIONS" section of app/prompts/system_prompt.py, which expects
    EXACTLY this "[Source - document: ..., document_id: ..., section:
    ..., page: ...]" format.

    Any piece of metadata Pipeline 1 didn't set on a given chunk (e.g. no
    section) is simply left out of that chunk's label rather than shown
    as blank/missing.
    """
    source_parts = []
    if chunk.get("document_name"):
        source_parts.append(f'document: "{chunk["document_name"]}"')
    if chunk.get("document_id"):
        source_parts.append(f'document_id: "{chunk["document_id"]}"')
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
    block of text, in the exact "Context / Question" shape every one of
    our system prompts expects. Each chunk is labeled with its source
    (see _format_chunk_with_source above) before being joined - `chunks`
    here is the list of dicts returned by
    opensearch_client.search_top_chunks(), not plain strings.
    """
    if chunks:
        context_text = "\n\n".join(_format_chunk_with_source(chunk) for chunk in chunks)
    else:
        context_text = "No context was found for this question."

    return f"Context:\n{context_text}\n\nQuestion:\n{question}"


def get_raw_claude_text(
    question: str,
    chunks: list,
    system_prompt: str,
    max_tokens: int = 4096,
    request_id: str = "",
) -> str:
    """
    WHAT THIS FUNCTION DOES:
    Sends the question + retrieved chunks to Claude via Bedrock's
    invoke_model() API, using whichever `system_prompt` the caller passes
    in (CONFLICT_DETECTION_PROMPT / SINGLE_ANSWER_PROMPT /
    CONFLICT_ANSWER_PROMPT), and returns Claude's answer EXACTLY as
    Claude wrote it - the raw, un-cleaned-up text. This might be plain
    JSON, JSON wrapped in markdown fences, or something messier -
    whatever Claude actually sent back.

    `max_tokens` defaults to 4096 for the two answer-generation prompts,
    but detect_conflict() passes a much smaller value - its entire
    expected output is `{"conflict_detected": true}`, so there's no
    reason to allow/pay for a long response there.
    """
    user_prompt = build_user_prompt(question, chunks)

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            # NOTE: do NOT add "temperature" here. This model/inference
            # profile's Bedrock validation hard-rejects it:
            #   ValidationException: `temperature` is deprecated for
            #   this model.
            # We tried pinning it to reduce wording variance across
            # identical questions and it broke every request with a 500.
            # Consistency has to come from the retrieval side
            # (config.MIN_ABSOLUTE_SCORE / RELEVANCE_SCORE_RATIO) and
            # prompt wording instead, not sampling params, for this model.
            "system": system_prompt,
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
        f"[{request_id}] [bedrock_client] Claude call took {duration:.2f}s, "
        f"stop_reason={stop_reason!r}"
    )
    if stop_reason == "max_tokens":
        logger.warning(
            f"[{request_id}] [bedrock_client] Claude's response was CUT OFF "
            "(stop_reason=max_tokens) - the answer may be truncated/invalid "
            "JSON. Consider raising max_tokens."
        )

    raw_text = ""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            raw_text = block["text"]
            break

    return raw_text


def _align_document_ids(references: list, document_ids: list, request_id: str = "") -> list:
    """
    WHAT THIS FUNCTION DOES:
    "documentids" is meant to be positionally paired with "references" -
    documentids[i] is the id for references[i]. If Claude ever generates
    the two arrays with different lengths (desync), zip()-ing them
    together in the functions below would silently pair the WRONG id to
    the wrong reference for every entry after the mismatch. This pads
    (with "") or truncates document_ids to exactly match len(references)
    first, so pairing stays correct - a mismatch is logged so it's
    visible, not silently absorbed.
    """
    if len(document_ids) == len(references):
        return document_ids

    logger.warning(
        f"[{request_id}] [bedrock_client] 'documentids' length "
        f"({len(document_ids)}) does not match 'references' length "
        f"({len(references)}) - Claude desynced them. Padding/truncating "
        f"to keep positional pairing correct."
    )
    if len(document_ids) < len(references):
        return document_ids + [""] * (len(references) - len(document_ids))
    return document_ids[: len(references)]


def _filter_hallucinated_references(answer: dict, chunks: list, request_id: str = "") -> None:
    """
    WHAT THIS FUNCTION DOES:
    A deterministic backstop against Claude fabricating a citation
    despite the system prompt's rules - modifies answer["references"]
    AND answer["documentids"] IN PLACE together (as paired entries),
    keeping only entries whose reference actually starts with a real
    document name from the chunks retrieved for THIS request. Prompt
    instructions alone are necessary but not sufficient - a model can
    still slip - so this makes it structurally impossible for a made-up
    reference (or its id) to reach the caller, regardless of whether
    Claude followed the prompt correctly. Never makes either list
    LONGER - only ever removes paired entries that don't check out.
    """
    # Chunks with no document_name at all can't back up ANY reference -
    # excluding them (rather than including "") stops an empty string
    # from matching (and passing) every reference via startswith("").
    known_document_names = {
        chunk["document_name"] for chunk in chunks if chunk.get("document_name")
    }

    original_references = answer.get("references", [])
    original_ids = _align_document_ids(
        original_references, answer.get("documentids", []), request_id
    )

    kept_references = []
    kept_ids = []
    for reference, document_id in zip(original_references, original_ids):
        if any(reference.startswith(name) for name in known_document_names):
            kept_references.append(reference)
            kept_ids.append(document_id)

    dropped_count = len(original_references) - len(kept_references)
    if dropped_count:
        logger.warning(
            f"[{request_id}] [bedrock_client] Dropped {dropped_count} reference(s) "
            f"that did NOT match any actually-retrieved document - Claude likely "
            f"fabricated {'them' if dropped_count > 1 else 'it'} despite the "
            f"prompt's rules. Original references: {original_references!r}"
        )

    answer["references"] = kept_references
    answer["documentids"] = kept_ids


def _deduplicate_references(answer: dict, request_id: str = "") -> None:
    """
    WHAT THIS FUNCTION DOES:
    Removes exact-duplicate entries from answer["references"] IN PLACE,
    keeping the first occurrence and preserving order - and removes the
    matching "documentids" entry at the same position, so the two lists
    stay paired. This is a deterministic backstop alongside
    _filter_hallucinated_references - the prompt already instructs
    Claude to cite each document only once (see SOURCE LABELS AND
    CITATIONS), but that is a request, not a guarantee, the same reason
    the hallucination filter above exists independent of the prompt's
    "don't fabricate" rule.
    """
    references = answer.get("references", [])
    document_ids = _align_document_ids(references, answer.get("documentids", []), request_id)

    seen = set()
    deduplicated_references = []
    deduplicated_ids = []
    for reference, document_id in zip(references, document_ids):
        if reference not in seen:
            seen.add(reference)
            deduplicated_references.append(reference)
            deduplicated_ids.append(document_id)

    if len(deduplicated_references) != len(references):
        logger.info(
            f"[{request_id}] [bedrock_client] Removed "
            f"{len(references) - len(deduplicated_references)} duplicate "
            f"reference(s) - original: {references!r}"
        )

    answer["references"] = deduplicated_references
    answer["documentids"] = deduplicated_ids


def detect_conflict(question: str, chunks: list, request_id: str = "") -> bool:
    """
    WHAT THIS FUNCTION DOES:
    Makes a small, focused Claude call using CONFLICT_DETECTION_PROMPT,
    whose only job is deciding whether the retrieved chunks contain a
    genuine, irreconcilable conflict for this exact question. Returns
    True/False - app/graph/nodes/rag_nodes.py's conditional edge uses
    this to route to generate_single_node or generate_conflict_node.

    FAIL-SAFE ON PARSE ERROR: if Claude's response isn't valid JSON or
    doesn't contain "conflict_detected", we return False (no conflict)
    rather than raising - this routes the request down the single-answer
    path, which still produces a correct, grounded answer even in the
    no-genuine-conflict cases (Case 1/Case 3). Losing the SECOND answer
    on a rare detection hiccup is a much smaller failure than the whole
    request erroring out.
    """
    raw_text = get_raw_claude_text(
        question, chunks, CONFLICT_DETECTION_PROMPT, max_tokens=200, request_id=request_id
    )
    try:
        parsed = json.loads(raw_text)
        conflict_detected = bool(parsed["conflict_detected"])
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error(
            f"[{request_id}] [bedrock_client] Failed to parse conflict-detection "
            f"response - defaulting to conflict_detected=False. Raw text was:\n{raw_text}"
        )
        return False

    logger.info(f"[{request_id}] [bedrock_client] conflict_detected={conflict_detected}")
    return conflict_detected


def ask_claude_single(question: str, chunks: list, request_id: str = "") -> dict:
    """
    WHAT THIS FUNCTION DOES:
    The no-conflict path. Makes a Claude call using SINGLE_ANSWER_PROMPT,
    parses the JSON reply, runs the same reference-hallucination and
    deduplication backstops as before, and returns just the "answer"
    dict. Used by generate_single_node.
    """
    raw_text = get_raw_claude_text(
        question, chunks, SINGLE_ANSWER_PROMPT, request_id=request_id
    )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(
            f"[{request_id}] [bedrock_client] Failed to parse single-answer "
            f"response as JSON. Raw text was:\n{raw_text}"
        )
        raise

    answer = parsed["answer"]
    _filter_hallucinated_references(answer, chunks, request_id)
    _deduplicate_references(answer, request_id)
    return answer


def ask_claude_conflict(question: str, chunks: list, request_id: str = "") -> tuple:
    """
    WHAT THIS FUNCTION DOES:
    The conflict path - only called once detect_conflict() has already
    confirmed a genuine conflict exists. Makes a Claude call using
    CONFLICT_ANSWER_PROMPT, parses the JSON reply, runs the same
    reference-hallucination and deduplication backstops on BOTH "answer"
    and "answeroption", and returns an (answer, answeroption) tuple.
    Used by generate_conflict_node.

    If Claude still returns an empty "answeroption" despite the conflict
    prompt (a prompt failure, not a routing failure - detect_conflict()
    already confirmed a real conflict exists), we log it clearly so it's
    visible in logs/monitoring rather than silently looking identical to
    a normal no-conflict response.
    """
    raw_text = get_raw_claude_text(
        question, chunks, CONFLICT_ANSWER_PROMPT, request_id=request_id
    )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(
            f"[{request_id}] [bedrock_client] Failed to parse conflict-answer "
            f"response as JSON. Raw text was:\n{raw_text}"
        )
        raise

    answer = parsed["answer"]
    _filter_hallucinated_references(answer, chunks, request_id)
    _deduplicate_references(answer, request_id)

    answeroption = parsed.get("answeroption") or {}
    if answeroption:
        _filter_hallucinated_references(answeroption, chunks, request_id)
        _deduplicate_references(answeroption, request_id)
    else:
        logger.warning(
            f"[{request_id}] [bedrock_client] detect_conflict() found a conflict, "
            "but the conflict-answer call returned an empty 'answeroption' - "
            "Claude did not produce a second independent answer as instructed."
        )

    return answer, answeroption
