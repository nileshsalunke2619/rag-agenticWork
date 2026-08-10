# =============================================================================
# app/vectordb/opensearch_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to AWS OpenSearch and AWS Bedrock
# (Titan Embeddings). It lives in its own `vectordb/` package inside
# `app/` since it's specifically "the vector database integration",
# separate from the graph/LLM/API logic elsewhere in the app.
#
# What this file does, in plain English:
#   1. Turns the user's question into a vector using Titan Embeddings.
#   2. Sends that vector to OpenSearch and asks for the top-k closest
#      chunks.
#   3. Returns just the plain text of those chunks.
# =============================================================================

import json
import logging

import boto3
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import TransportError

# "as config" keeps every reference below (config.OPENSEARCH_HOST, etc.)
# identical to how this file worked before the folder reorganization -
# only the import path changed, not how settings are used.
from app.config import settings as config

logger = logging.getLogger(__name__)


def get_opensearch_client() -> OpenSearch:
    """
    WHAT THIS FUNCTION DOES:
    Creates and returns a connected OpenSearch client, authenticated
    using your AWS credentials.

    IMPORTANT - "es" vs "aoss":
    AWS has TWO different OpenSearch offerings that sign requests
    differently - "es" (managed domains) vs "aoss" (Serverless
    collections). config.OPENSEARCH_SERVICE controls which one we sign
    for; getting this wrong produces a "403 Forbidden" that looks like a
    permissions problem but isn't.

    DIAGNOSTIC PRINTS:
    The print() statements below show EXACTLY which AWS identity and
    connection settings are being used for every request, so if
    something is denied, you have concrete values to check against your
    OpenSearch access policy.
    """
    credentials = boto3.Session().get_credentials()

    try:
        identity = boto3.client(
            "sts", region_name=config.OPENSEARCH_REGION
        ).get_caller_identity()
        logger.info(
            f"[opensearch_client] Calling AWS as identity: "
            f"Account={identity.get('Account')} Arn={identity.get('Arn')}"
        )
    except Exception as identity_error:  # noqa: BLE001 - diagnostic only
        logger.warning(
            f"[opensearch_client] Could NOT resolve AWS identity: {identity_error}"
        )

    logger.info(
        f"[opensearch_client] Connection settings: "
        f"HOST={config.OPENSEARCH_HOST} PORT={config.OPENSEARCH_PORT} "
        f"INDEX={config.OPENSEARCH_INDEX} SERVICE={config.OPENSEARCH_SERVICE} "
        f"REGION={config.OPENSEARCH_REGION}"
    )

    # IMPORTANT: this MUST be config.OPENSEARCH_REGION - not a general
    # "AWS_REGION" - because OpenSearch and Bedrock can live in different
    # AWS regions in this project.
    auth = RequestsAWSV4SignerAuth(
        credentials, config.OPENSEARCH_REGION, config.OPENSEARCH_SERVICE
    )

    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client


def get_query_embedding(text: str) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends `text` (the user's question) to Amazon Titan Embeddings (via
    AWS Bedrock) and returns the embedding - a list of numbers that
    represents the MEANING of that text.
    """
    # NOTE: config.TITAN_REGION, not OPENSEARCH_REGION or BEDROCK_REGION -
    # Titan Embeddings may live in a different AWS region than OpenSearch
    # or Claude in this project.
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.TITAN_REGION)

    request_body = json.dumps({"inputText": text})

    response = bedrock_runtime.invoke_model(
        modelId=config.TITAN_EMBEDDING_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


def search_top_chunks(
    question: str, query_embedding: list, request_id: str = "", top_k: int = None
) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends a HYBRID search to OpenSearch - combining TWO ways of matching
    in one query:
      1. Vector (kNN) similarity on VECTOR_FIELD_NAME - finds chunks
         that are semantically similar in MEANING to the question, even
         if they use completely different wording.
      2. Keyword (BM25 text) matching on TEXT_FIELD_NAME - finds chunks
         that share actual WORDS with the question. This catches exact
         terms (form numbers, SOP codes, product names) that vector
         search alone can sometimes under-rank, since embeddings
         represent overall meaning, not exact wording.
    Both are combined inside one "bool" / "should" query, so a chunk
    that scores well on EITHER (or both) can be returned. This is a
    BASIC hybrid search - OpenSearch just adds the two scores together;
    there's no score normalization pipeline here, tunable via
    config.KEYWORD_MATCH_BOOST.

    RELEVANCE FLOOR:
    A plain "should" query has no minimum score - it always returns the
    top `top_k` hits no matter how weak the worst of them is. Without a
    floor, a question with only 1-2 truly relevant chunks in the index
    still gets `top_k` chunks back, and the padding is often a leftover
    chunk from an unrelated topic that happened to share a few keywords.
    That leftover then gets handed to Claude as if it were real context.
    To fix this WITHOUT changing the query shape above, we over-fetch a
    slightly larger candidate pool, then drop any hit scoring below
    config.RELEVANCE_SCORE_RATIO of the top hit's score, before slicing
    back down to `top_k`. See config.RELEVANCE_SCORE_RATIO for why a
    RELATIVE threshold is used instead of a fixed absolute score.

    RETURN SHAPE:
    Returns a list of dicts, NOT a list of plain strings - each dict has
    "content" (the chunk text) plus "document_id", "document_name",
    "page", "section", "chunk_id" (citation metadata from Pipeline 1).
    This is what lets app/llm/bedrock_client.py label each chunk with
    its real source before sending it to Claude, instead of Claude
    having to guess a reference from the chunk's raw text.
    """
    if top_k is None:
        top_k = config.TOP_K

    # Over-fetch so that filtering out weak matches below doesn't starve
    # us of chunks when there ARE enough genuinely relevant ones.
    candidate_pool_size = top_k * 2

    client = get_opensearch_client()

    search_query = {
        "size": candidate_pool_size,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            config.VECTOR_FIELD_NAME: {
                                "vector": query_embedding,
                                "k": top_k,
                            }
                        }
                    },
                    {
                        "match": {
                            config.TEXT_FIELD_NAME: {
                                "query": question,
                                "boost": config.KEYWORD_MATCH_BOOST,
                            }
                        }
                    },
                ]
            }
        },
    }

    logger.info(
        f"[{request_id}] [opensearch_client] Searching index: {config.OPENSEARCH_INDEX!r}"
    )
    logger.info(
        f"[{request_id}] [opensearch_client] Hybrid query: knn on field "
        f"{config.VECTOR_FIELD_NAME!r} (k={top_k}, vector length="
        f"{len(query_embedding)}) + match on field {config.TEXT_FIELD_NAME!r} "
        f"(boost={config.KEYWORD_MATCH_BOOST}) for question={question!r}"
    )

    # DIAGNOSTIC: we wrap this call so that if OpenSearch rejects it, we
    # log the FULL detail AWS actually sent back - not just "403
    # forbidden" - before letting the error continue up (re-raising),
    # so FastAPI still reports the failure like normal.
    try:
        response = client.search(index=config.OPENSEARCH_INDEX, body=search_query)
    except TransportError as search_error:
        logger.error(
            f"[{request_id}] [opensearch_client] SEARCH FAILED - "
            f"full AWS error detail below:"
        )
        logger.error(f"    status_code: {search_error.status_code}")
        logger.error(f"    error:       {search_error.error}")
        try:
            logger.error(f"    info:        {json.dumps(search_error.info, indent=4)}")
        except TypeError:
            logger.error(f"    info:        {search_error.info!r}")
        raise

    hits = response["hits"]["hits"]

    if not hits:
        logger.info(f"[{request_id}] [opensearch_client] Hybrid search returned 0 hits")
        return []

    # OpenSearch returns hits sorted by score, highest first, so hits[0]
    # is always the best match in this pool.
    top_score = hits[0]["_score"]
    logger.info(
        f"[{request_id}] [opensearch_client] Top score in candidate pool: "
        f"{top_score:.4f} (use this to calibrate config.MIN_ABSOLUTE_SCORE - "
        f"NOTE: this is a raw, unbounded score again, not the 0-1 normalized "
        f"score from the brief normalized-hybrid-search experiment, so any "
        f"MIN_ABSOLUTE_SCORE value calibrated against that 0-1 scale is no "
        f"longer valid)"
    )

    # ABSOLUTE floor first: if even the BEST hit in the pool is too weak,
    # nothing here is a real match (e.g. an off-topic question like
    # "what's the temperature today" against an SOP/forms index) - a
    # RATIO filter alone can't catch this, since a pool of uniformly bad
    # matches passes a relative comparison easily. See
    # config.MIN_ABSOLUTE_SCORE for how to calibrate this.
    if config.MIN_ABSOLUTE_SCORE > 0 and top_score < config.MIN_ABSOLUTE_SCORE:
        logger.info(
            f"[{request_id}] [opensearch_client] Top score {top_score:.4f} is below "
            f"MIN_ABSOLUTE_SCORE ({config.MIN_ABSOLUTE_SCORE}) - treating this as "
            f"having NO real match, returning 0 chunks instead of the closest "
            f"available junk"
        )
        return []

    min_acceptable_score = top_score * config.RELEVANCE_SCORE_RATIO

    relevant_hits = [hit for hit in hits if hit["_score"] >= min_acceptable_score]
    dropped_count = len(hits) - len(relevant_hits)
    if dropped_count:
        logger.info(
            f"[{request_id}] [opensearch_client] Dropped {dropped_count} weakly-"
            f"scoring hit(s) below {config.RELEVANCE_SCORE_RATIO:.0%} of the top "
            f"score ({top_score:.4f}) - likely leftover matches from an unrelated "
            f"topic, not genuinely relevant to this question"
        )

    kept_hits = relevant_hits[:top_k]

    # Each chunk carries its citation metadata alongside its text, not
    # just the bare text string - this is what lets Claude cite a real
    # document/section/page instead of guessing a reference from the
    # chunk's raw body text. Every field defaults to "" if Pipeline 1
    # didn't set it on a given chunk, same defensive pattern already used
    # for TEXT_FIELD_NAME above.
    chunks = []
    for i, hit in enumerate(kept_hits):
        source = hit["_source"]

        # DIAGNOSTIC (first hit only, so this doesn't spam the logs every
        # chunk): shows the REAL top-level keys OpenSearch actually
        # returned for this document. If DOCUMENT_NAME_FIELD_NAME etc.
        # don't appear in this list, .get() below will always return ""
        # even if that field name "exists" somewhere in the document -
        # e.g. if it's nested one level down inside a "metadata" object
        # instead of sitting at the top level, .get() won't find it.
        if i == 0:
            logger.info(
                f"[{request_id}] [opensearch_client] DIAGNOSTIC - top-level keys "
                f"actually present on this document: {list(source.keys())}"
            )

        chunks.append(
            {
                "content": source.get(config.TEXT_FIELD_NAME, ""),
                "document_id": source.get(config.DOCUMENT_ID_FIELD_NAME, ""),
                "document_name": source.get(config.DOCUMENT_NAME_FIELD_NAME, ""),
                "page": source.get(config.PAGE_FIELD_NAME, ""),
                "section": source.get(config.SECTION_FIELD_NAME, ""),
                "chunk_id": source.get(config.CHUNK_ID_FIELD_NAME, ""),
            }
        )

    logger.info(
        f"[{request_id}] [opensearch_client] Hybrid search returned {len(chunks)} "
        f"relevant chunk(s) (of {len(hits)} candidate(s) fetched)"
    )

    return chunks
