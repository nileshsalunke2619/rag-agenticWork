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
import re
import time
from typing import Optional

import boto3
from botocore.config import Config
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


def _filter_to_dominant_document(hits: list, request_id: str = "") -> list:
    """
    WHAT THIS FUNCTION DOES:
    Catches cross-document contamination that the relevance floor alone
    doesn't - a single stray chunk from an unrelated document sneaking
    in alongside several chunks that clearly belong together in ONE
    document. Groups the already relevance-filtered hits by
    DOCUMENT_ID_FIELD_NAME; if one document accounts for at least
    config.DOCUMENT_DOMINANCE_RATIO of them, drops every hit from any
    OTHER document, so the dominant document gets first claim on every
    TOP_K slot instead of losing one to noise.

    Hits with no document_id at all are left untouched either way - they
    can't be grouped, so they're neither counted as competition for a
    dominant document nor discarded as an outlier.

    If hits are closer to evenly split across several documents (no
    single one clears the ratio), nothing is filtered - that's treated
    as a legitimate multi-document question, not noise.

    DISABLED BY DEFAULT: config.DOCUMENT_DOMINANCE_RATIO defaults to 0,
    which means this function is a no-op until deliberately enabled and
    calibrated - see the settings.py comment for why.
    """
    if config.DOCUMENT_DOMINANCE_RATIO <= 0:
        return hits

    groupable_hits = [
        hit for hit in hits if hit["_source"].get(config.DOCUMENT_ID_FIELD_NAME)
    ]
    ungroupable_hits = [
        hit for hit in hits if not hit["_source"].get(config.DOCUMENT_ID_FIELD_NAME)
    ]

    if len(groupable_hits) < 2:
        # Nothing to compare - 0 or 1 groupable hits can't have a
        # "minority" document to drop.
        return hits

    hits_by_document = {}
    for hit in groupable_hits:
        document_id = hit["_source"][config.DOCUMENT_ID_FIELD_NAME]
        hits_by_document.setdefault(document_id, []).append(hit)

    if len(hits_by_document) < 2:
        # Every groupable hit already belongs to the same document -
        # nothing to filter out.
        return hits

    dominant_document_id = max(hits_by_document, key=lambda doc_id: len(hits_by_document[doc_id]))
    dominant_share = len(hits_by_document[dominant_document_id]) / len(groupable_hits)

    if dominant_share < config.DOCUMENT_DOMINANCE_RATIO:
        # No single document clearly dominates - treat as a legitimate
        # multi-document question rather than filtering anything.
        return hits

    dropped_hits = [
        hit for hit in groupable_hits if hit not in hits_by_document[dominant_document_id]
    ]
    logger.info(
        f"[{request_id}] [opensearch_client] Document {dominant_document_id!r} "
        f"accounts for {dominant_share:.0%} of relevant hits (>= "
        f"{config.DOCUMENT_DOMINANCE_RATIO:.0%} threshold) - dropping "
        f"{len(dropped_hits)} hit(s) from other document(s) as likely noise"
    )

    # Preserve original score-sorted order: dominant-document hits plus
    # untouched ungroupable hits, filtered from the original list rather
    # than reassembled, so relative ranking isn't disturbed.
    kept_ids = {id(hit) for hit in hits_by_document[dominant_document_id]}
    kept_ids.update(id(hit) for hit in ungroupable_hits)
    return [hit for hit in hits if id(hit) in kept_ids]


# KNOWN DOCUMENT ID PREFIXES IN THIS CORPUS - extend this if the corpus
# ever uses other prefixes. Matches with or without a hyphen/space
# between the prefix and the digits, case-insensitively (e.g.
# "SOP-101111", "sop 101111", "FORM120141" all match).
_DOCUMENT_ID_PATTERN = re.compile(r"\b(SOP|FORM|CD)[-\s]?(\d+)\b", re.IGNORECASE)


def _extract_document_ids(question: str) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Scans the user's question for literal document ID mentions (e.g.
    "SOP-101111", "sop 101111") and returns them normalized to the
    canonical "PREFIX-digits" form actually stored in
    config.DOCUMENT_ID_FIELD_NAME (uppercase prefix, single hyphen, no
    stray whitespace) - e.g. "SOP-101111". Used by search_top_chunks()
    to restrict retrieval to only the named document(s) when the user
    explicitly names one, instead of searching the whole corpus and
    risking unrelated documents (which may share keywords, or simply
    mention this same ID in their own "Related Documents" section)
    getting pulled into the answer.

    Returns an empty list when no document ID is mentioned - the normal
    case - and search_top_chunks() searches the whole corpus as before.
    """
    matches = _DOCUMENT_ID_PATTERN.findall(question)
    return [f"{prefix.upper()}-{digits}" for prefix, digits in matches]


def _strip_document_ids(question: str) -> str:
    """
    DOC-FILTER: returns the question with any literal document IDs
    removed, for the KEYWORD side of the search only. Once retrieval is
    already restricted to the named document, the ID text itself adds
    nothing useful to keyword matching - it only rewards chunks that
    happen to print the ID (cover pages, headers, footers) over the
    section the user actually asked about.
    """
    return " ".join(_DOCUMENT_ID_PATTERN.sub(" ", question).split())


# DOC-FILTER: the two ways of applying the document filter inside the
# search, tried in this order. "knn_filter" is the efficient one but
# only some vector engines accept it; "wide_k" works on any engine.
# Whichever one OpenSearch accepts first is remembered here, so later
# requests go straight to it.
_IN_QUERY_FILTER_MODES = ("knn_filter", "wide_k")
_working_in_query_filter_mode = None


# RERANK: one Bedrock client for the reranker, created on first use and
# reused. Short timeouts on purpose - reranking is an improvement, never
# a requirement, so a slow reranker must not hold up the whole request.
_rerank_client = None


def _get_rerank_client():
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = boto3.client(
            "bedrock-runtime",
            region_name=config.RERANK_REGION,
            config=Config(
                connect_timeout=5,
                read_timeout=20,
                retries={"max_attempts": 2},
            ),
        )
    return _rerank_client


def _rerank_hits(question: str, hits: list, request_id: str = "") -> Optional[list]:
    """
    RERANK - WHAT THIS FUNCTION DOES:
    Sends the question plus the text of every candidate hit to the
    reranker model (config.RERANK_MODEL_ID, via Bedrock invoke_model in
    config.RERANK_REGION) and returns the hits RE-ORDERED best-first by
    how well each one actually answers the question - at most
    config.RERANK_TOP_N of them.

    FAIL-SAFE, BY DESIGN: returns None on ANY problem (access denied,
    throttling, timeout, unexpected response shape) or when there is
    nothing worth reranking. The caller treats None as "use the normal
    OpenSearch order", so a reranker failure can never break a request -
    it only means that one request behaves the way it did before
    reranking existed.
    """
    candidates = []
    for original_rank, hit in enumerate(hits, start=1):
        text = (hit["_source"].get(config.TEXT_FIELD_NAME) or "").strip()
        if text:
            candidates.append((original_rank, hit, text))

    if len(candidates) < 2:
        return None

    top_n = min(config.RERANK_TOP_N, len(candidates))
    request_body = json.dumps(
        {
            "api_version": 2,
            "query": question,
            "documents": [text for _, _, text in candidates],
            "top_n": top_n,
        }
    )

    try:
        start_time = time.time()
        response = _get_rerank_client().invoke_model(
            modelId=config.RERANK_MODEL_ID,
            body=request_body,
            contentType="application/json",
            accept="application/json",
        )
        duration = time.time() - start_time
        results = json.loads(response["body"].read())["results"]

        reranked_hits = []
        seen_indexes = set()
        for new_rank, result in enumerate(results, start=1):
            index = result["index"]
            if index in seen_indexes or not 0 <= index < len(candidates):
                continue
            seen_indexes.add(index)
            original_rank, hit, _ = candidates[index]
            source = hit["_source"]
            logger.info(
                f"[{request_id}] [opensearch_client] [rerank]     {new_rank}. "
                f"rerank_score={result['relevance_score']:.4f} "
                f"(was #{original_rank}) "
                f"document_id={source.get(config.DOCUMENT_ID_FIELD_NAME, '')!r} "
                f"section={source.get(config.SECTION_FIELD_NAME, '')!r}"
            )
            reranked_hits.append(hit)
    except Exception:  # noqa: BLE001 - reranking must never break the request
        logger.exception(
            f"[{request_id}] [opensearch_client] [rerank] Rerank call FAILED - "
            f"falling back to the normal OpenSearch order"
        )
        return None

    if not reranked_hits:
        logger.warning(
            f"[{request_id}] [opensearch_client] [rerank] Reranker returned no "
            f"usable results - falling back to the normal OpenSearch order"
        )
        return None

    logger.info(
        f"[{request_id}] [opensearch_client] [rerank] Reranked {len(candidates)} "
        f"candidate(s) with {config.RERANK_MODEL_ID!r} in {duration:.2f}s - "
        f"keeping the best {len(reranked_hits)}"
    )
    return reranked_hits


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
    Both are combined via OpenSearch's "hybrid" query type + a search
    pipeline (config.SEARCH_PIPELINE_NAME) that normalizes/weights the
    two scores - see the HYBRID-SEARCH-TODO block below.

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

    DOCUMENT ID FILTERING:
    If the question names a specific document ID (see
    _extract_document_ids() above), a "post_filter" restricts the
    RETURNED hits to ONLY that document (or documents, if more than one
    is named) - a real, deterministic restriction, not a prompt request
    Claude could ignore. "post_filter" (not a "bool"/"filter" clause
    nested inside the query) is used deliberately here: the "hybrid"
    query type's own filter support varies by OpenSearch version, while
    "post_filter" is a plain top-level search parameter that has worked
    the same way for years, has no effect on the hybrid query's own
    scoring/normalization pipeline, and (since this endpoint never uses
    aggregations) has none of post_filter's usual "doesn't affect facet
    counts" caveats to worry about - the safest, most version-proof way
    to add this without touching the hybrid/search_pipeline setup above.

    FALLBACK: if a named ID doesn't actually match anything in the index
    (typo, wrong ID), the filtered search returns 0 hits - rather than
    treating that as "no context at all", we retry once WITHOUT the
    filter, falling back to a normal best-effort search across the
    whole corpus. This is a deliberate choice (best-effort over a hard
    "document not found" failure) - revisit if a hard failure is
    preferred instead.

    DOC-FILTER (config.DOCUMENT_FILTER_IN_QUERY, off by default):
    When on, the document ID restriction above is ALSO applied inside
    both halves of the hybrid query, so the vector and keyword searches
    only ever look within the named document(s) - instead of searching
    every document and discarding the others afterwards. The
    "post_filter" stays in place as well, as a second guarantee.

    RERANKING (config.RERANK_ENABLED, off by default):
    When on, the candidate pool is re-ordered by a reranker model (see
    _rerank_hits() above) and only the best config.RERANK_TOP_N chunks
    are returned, instead of the first `top_k` in OpenSearch's order.
    """
    if top_k is None:
        top_k = config.TOP_K

    # Over-fetch so that filtering out weak matches below doesn't starve
    # us of chunks when there ARE enough genuinely relevant ones.
    candidate_pool_size = top_k * 2

    # RERANK: optionally give the reranker a different-sized pool to
    # choose from. 0 (the default) keeps the pool exactly as above.
    if config.RERANK_ENABLED and config.RERANK_CANDIDATES > 0:
        candidate_pool_size = config.RERANK_CANDIDATES

    client = get_opensearch_client()

    detected_document_ids = _extract_document_ids(question)
    if detected_document_ids:
        logger.info(
            f"[{request_id}] [opensearch_client] Detected document_id(s) in "
            f"question: {detected_document_ids} - restricting retrieval to "
            f"only these document(s)"
        )
    else:
        logger.info(
            f"[{request_id}] [opensearch_client] No document_id detected in "
            f"question - searching all documents as before"
        )

    """
    search_query = {
        "size": candidate_pool_size,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            config.VECTOR_FIELD_NAME: {
                                "vector": query_embedding,
                                # BUG FIX: this was "k": top_k, which meant the
                                # vector/semantic side of the search only ever
                                # considered TOP_K nearest neighbors, while the
                                # outer "size" below was widened to
                                # candidate_pool_size (TOP_K * 2) for the
                                # relevance floor's over-fetch. Only the
                                # keyword/BM25 side (uncapped) actually
                                # benefited from that wider pool - the kNN side
                                # was silently stuck at the narrower TOP_K the
                                # whole time. A chunk that's a genuine semantic
                                # match but ranks just outside the top TOP_K by
                                # embedding distance, and doesn't share enough
                                # literal keywords to be rescued by BM25, could
                                # never surface at all. Matching this to
                                # candidate_pool_size actually gives kNN the
                                # wider net that was always intended.
                                "k": candidate_pool_size,
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
    """

    #HYBRID-SEARCH-TODO
    def _build_search_query(
        apply_document_id_filter: bool, in_query_filter_mode: str = None
    ) -> dict:
        query_body = {
            "size": candidate_pool_size,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                config.VECTOR_FIELD_NAME: {
                                    "vector": query_embedding,
                                    # Same fix as before: k must match
                                    # candidate_pool_size, not top_k - otherwise
                                    # the semantic/vector side stays artificially
                                    # narrow while the outer "size" and the BM25
                                    # side both search the wider pool.
                                    "k": candidate_pool_size,
                                }
                            }
                        },
                        {
                            "match": {
                                config.TEXT_FIELD_NAME: {
                                    "query": question,
                                    # NOTE: with the hybrid query type + a
                                    # normalization pipeline (see below), this
                                    # boost becomes far less critical - the
                                    # pipeline's "weights" parameter is now the
                                    # main lever for balancing vector vs.
                                    # keyword influence. You can likely reduce
                                    # KEYWORD_MATCH_BOOST toward 1.0 (neutral)
                                    # once the pipeline weights are tuned, since
                                    # it's no longer compensating for a raw
                                    # scale mismatch.
                                    "boost": config.KEYWORD_MATCH_BOOST,
                                }
                            }
                        },
                    ]
                }
            },
        }
        # DOC-FILTER: restrict BOTH halves of the hybrid query to the
        # named document(s), not just the returned hits. Skipped entirely
        # (query_body untouched) unless in_query_filter_mode is given.
        if apply_document_id_filter and detected_document_ids and in_query_filter_mode:
            document_filter = {
                "terms": {config.DOCUMENT_ID_FIELD_NAME: detected_document_ids}
            }
            sub_queries = query_body["query"]["hybrid"]["queries"]
            knn_clause = sub_queries[0]["knn"][config.VECTOR_FIELD_NAME]
            match_clause = sub_queries[1]["match"][config.TEXT_FIELD_NAME]

            # Keyword side: search for what the user asked ABOUT the
            # document, not for the document's own ID text.
            match_clause["query"] = _strip_document_ids(question) or question
            sub_queries[1] = {
                "bool": {"must": [sub_queries[1]], "filter": [document_filter]}
            }

            if in_query_filter_mode == "knn_filter":
                # Vector side, efficient form: the engine itself only
                # considers chunks from the named document(s).
                knn_clause["filter"] = document_filter
            else:
                # Vector side, works-on-any-engine form: fetch a much
                # larger set of nearest chunks, THEN filter, so the named
                # document's chunks are very unlikely to be missed.
                knn_clause["k"] = max(candidate_pool_size, config.DOCUMENT_FILTER_KNN_K)
                sub_queries[0] = {
                    "bool": {"must": [sub_queries[0]], "filter": [document_filter]}
                }

        if apply_document_id_filter and detected_document_ids:
            query_body["post_filter"] = {
                "terms": {config.DOCUMENT_ID_FIELD_NAME: detected_document_ids}
            }
        return query_body

    def _search_with_in_query_document_filter():
        """
        DOC-FILTER: runs the search with the document filter applied
        inside the query, trying each mode in _IN_QUERY_FILTER_MODES
        until OpenSearch accepts one. Returns the response, or None if
        no mode could be used - the caller then runs the original
        post_filter-only search, exactly as before this switch existed.
        """
        global _working_in_query_filter_mode

        if _working_in_query_filter_mode:
            modes_to_try = [_working_in_query_filter_mode]
        else:
            modes_to_try = list(_IN_QUERY_FILTER_MODES)

        for mode in modes_to_try:
            try:
                filtered_response = client.search(
                    index=config.OPENSEARCH_INDEX,
                    body=_build_search_query(True, in_query_filter_mode=mode),
                    params={"search_pipeline": config.SEARCH_PIPELINE_NAME},
                )
            except TransportError as filter_error:
                logger.warning(
                    f"[{request_id}] [opensearch_client] [doc-filter] In-query "
                    f"document filter mode {mode!r} failed (status_code="
                    f"{filter_error.status_code}, error={filter_error.error})"
                )
                if filter_error.status_code == 400:
                    # OpenSearch rejected the QUERY SHAPE itself (e.g. this
                    # vector engine doesn't accept a filter inside knn) -
                    # try the next mode.
                    continue
                # Anything else (timeout, throttling, auth) is not about
                # the query shape - stop here and let the original search
                # below run and report the problem the usual way.
                return None

            _working_in_query_filter_mode = mode
            logger.info(
                f"[{request_id}] [opensearch_client] [doc-filter] Document filter "
                f"applied inside the search (mode={mode!r}) for "
                f"{detected_document_ids}"
            )
            return filtered_response

        logger.warning(
            f"[{request_id}] [opensearch_client] [doc-filter] No in-query filter "
            f"mode was accepted - using the original post_filter-only search"
        )
        return None

    search_query = _build_search_query(apply_document_id_filter=True)

    logger.info(
        f"[{request_id}] [opensearch_client] Searching index: {config.OPENSEARCH_INDEX!r}"
    )
    logger.info(
        f"[{request_id}] [opensearch_client] Hybrid query: knn on field "
        f"{config.VECTOR_FIELD_NAME!r} (k={candidate_pool_size}, vector length="
        f"{len(query_embedding)}) + match on field {config.TEXT_FIELD_NAME!r} "
        f"(boost={config.KEYWORD_MATCH_BOOST}) for question={question!r}"
    )

    # DOC-FILTER: when switched on AND the question names a document,
    # try the search with the filter applied inside the query first.
    # `response` stays None when the switch is off, no document is named,
    # or no in-query mode worked - in all of those cases the original
    # search below runs unchanged.
    response = None
    if config.DOCUMENT_FILTER_IN_QUERY and detected_document_ids:
        response = _search_with_in_query_document_filter()

    # DIAGNOSTIC: we wrap this call so that if OpenSearch rejects it, we
    # log the FULL detail AWS actually sent back - not just "403
    # forbidden" - before letting the error continue up (re-raising),
    # so FastAPI still reports the failure like normal.
    try:
        #response = client.search(index=config.OPENSEARCH_INDEX, body=search_query)
        #HYBRID-SEARCH-TODO
        if response is None:  # DOC-FILTER: skip if already answered above
            response = client.search(
                index=config.OPENSEARCH_INDEX,
                body=search_query,
                params={"search_pipeline": config.SEARCH_PIPELINE_NAME},   # NEW line
            )


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

    # FALLBACK: a named document ID that doesn't actually exist in the
    # index would otherwise silently return 0 chunks. Retry once WITHOUT
    # the filter - best-effort across the whole corpus - rather than
    # treating this as "no context at all".
    if not hits and detected_document_ids:
        logger.warning(
            f"[{request_id}] [opensearch_client] Filtered search for "
            f"document_id(s) {detected_document_ids} returned 0 hits - falling "
            f"back to an unfiltered search across all documents"
        )
        search_query = _build_search_query(apply_document_id_filter=False)
        try:
            response = client.search(
                index=config.OPENSEARCH_INDEX,
                body=search_query,
                params={"search_pipeline": config.SEARCH_PIPELINE_NAME},
            )
        except TransportError as search_error:
            logger.error(
                f"[{request_id}] [opensearch_client] FALLBACK SEARCH FAILED - "
                f"full AWS error detail below:"
            )
            logger.error(f"    status_code: {search_error.status_code}")
            logger.error(f"    error:       {search_error.error}")
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

    # DISABLED BY DEFAULT: config.RELEVANCE_SCORE_RATIO defaults to 0, so
    # this is a no-op until deliberately enabled and calibrated - see
    # the settings.py comment for why (two confirmed false negatives
    # against real questions with the old 0.5 default).
    if config.RELEVANCE_SCORE_RATIO > 0:
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
    else:
        relevant_hits = hits

    relevant_hits = _filter_to_dominant_document(relevant_hits, request_id)

    # RERANK: when switched on, let the reranker choose the best
    # RERANK_TOP_N chunks from the pool. If it is off, or the rerank call
    # fails for any reason (_rerank_hits() returns None), we keep the
    # first `top_k` in OpenSearch's own order - exactly as before.
    reranked_hits = None
    if config.RERANK_ENABLED:
        reranked_hits = _rerank_hits(question, relevant_hits, request_id)

    if reranked_hits is not None:
        kept_hits = reranked_hits[: config.RERANK_TOP_N]
    else:
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


def search_nearest_feedback(query_embedding: list, request_id: str = "") -> Optional[dict]:
    """
    WHAT THIS FUNCTION DOES:
    Looks up the closest past thumbs-upped question(s) in
    config.OPENSEARCH_FEEDBACK_INDEX, using a PURE kNN search -
    deliberately NOT the hybrid kNN+BM25 query search_top_chunks() above
    uses. This index's "embedding" field is mapped with
    "space_type": "cosinesimil" (confirmed against the LIVE index via
    app/scripts/check_feedback_index_mapping.py - this index already
    existed with the correct mapping, so nothing needed to be created),
    so a plain kNN query's score is OpenSearch's native cosine score,
    normalized to 0-1 - directly comparable to
    config.FEEDBACK_SIMILARITY_THRESHOLD as a real "% similar" number.
    Mixing in a BM25/keyword clause here (like search_top_chunks does)
    would reintroduce an unbounded, uncalibratable score - exactly what
    this function exists to avoid.

    WHY k=5, NOT k=1:
    The live index was found to already contain a handful of manually
    inserted rows with an empty "response" field (see chat history -
    app/scripts/peek_feedback_sample.py surfaced these). A hit on one of
    those with k=1 would return a BLANK answer with no way to recover.
    Pulling a small pool and skipping any hit with an empty/unparseable
    "response" means a single bad legacy row can't break a real request -
    we fall through to the next-nearest candidate, or to a MISS (normal
    retrieval) if none in the pool are usable.

    RETURNS:
    None if the feedback index doesn't exist yet, is empty, has no hits
    above threshold, or every candidate above threshold has an unusable
    "response" (all treated as "no feedback match" - the caller falls
    through to the normal retrieve/generate flow, never errors the
    request). Otherwise a dict: {"score": float, "question": str,
    "answer": dict} - "answer" is the exact answer JSON that was
    thumbs-upped, ready to return as-is on a cache hit.
    """
    client = get_opensearch_client()

    search_query = {
        "size": 5,
        "query": {
            "knn": {
                config.VECTOR_FIELD_NAME: {
                    "vector": query_embedding,
                    "k": 5,
                }
            }
        },
    }

    try:
        response = client.search(index=config.OPENSEARCH_FEEDBACK_INDEX, body=search_query)
    except TransportError as search_error:
        # NOT_FOUND (index doesn't exist yet, or before the first
        # thumbsup has ever been stored) is expected in normal operation,
        # not a real failure - treat it as "no feedback match" and let
        # the caller fall through to the normal flow. Any OTHER error is
        # a real problem and should still surface.
        if search_error.status_code == 404:
            logger.info(
                f"[{request_id}] [opensearch_client] Feedback index "
                f"{config.OPENSEARCH_FEEDBACK_INDEX!r} not found - treating as no "
                f"feedback match"
            )
            return None
        logger.error(
            f"[{request_id}] [opensearch_client] Feedback search FAILED - "
            f"status_code: {search_error.status_code}, error: {search_error.error}"
        )
        raise

    hits = response["hits"]["hits"]
    if not hits:
        logger.info(
            f"[{request_id}] [opensearch_client] Feedback check: no stored feedback "
            f"yet - MISS"
        )
        return None

    top_score = hits[0]["_score"]
    logger.info(
        f"[{request_id}] [opensearch_client] Feedback check: nearest score={top_score:.4f} "
        f"(threshold={config.FEEDBACK_SIMILARITY_THRESHOLD})"
    )

    for hit in hits:
        score = hit["_score"]
        if score < config.FEEDBACK_SIMILARITY_THRESHOLD:
            # hits are score-sorted descending - once we drop below
            # threshold, every remaining hit will too.
            break

        source = hit["_source"]
        raw_response = source.get("response", "")
        if not raw_response:
            logger.warning(
                f"[{request_id}] [opensearch_client] Skipping feedback hit "
                f"(score={score:.4f}) with empty 'response' field - likely stale/"
                f"test data, not a usable cached answer"
            )
            continue

        try:
            answer = json.loads(raw_response)
        except (TypeError, ValueError):
            logger.warning(
                f"[{request_id}] [opensearch_client] Skipping feedback hit "
                f"(score={score:.4f}) - 'response' field is not valid JSON"
            )
            continue

        logger.info(
            f"[{request_id}] [opensearch_client] Feedback check: CACHE HIT "
            f"(score={score:.4f})"
        )
        return {"score": score, "question": source.get("question", ""), "answer": answer}

    logger.info(
        f"[{request_id}] [opensearch_client] Feedback check: MISS, continuing to "
        f"normal retrieval"
    )
    return None


def store_feedback(
    question: str,
    query_embedding: list,
    payload: dict,
    request_id: str = "",
) -> None:
    """
    WHAT THIS FUNCTION DOES:
    Indexes ONE document into config.OPENSEARCH_FEEDBACK_INDEX, called
    ONLY when POST /agent/v1/feedback receives feedback="thumsup" (see
    app/api/routes/feedback.py - negative/other feedback values are
    acknowledged but never reach this function).

    FIELD NAMES MATCH THE INDEX'S EXISTING LIVE MAPPING EXACTLY
    (question, embedding, user_id, question_id, response, option,
    source, document_id, chunk_id, nearest_feedback_score) - this index
    already existed with 14 rows in it before this feature was built
    (confirmed via app/scripts/check_feedback_index_mapping.py /
    peek_feedback_sample.py), so we write into the SAME shape rather
    than inventing a second, incompatible one. No new fields are added:

      - "response" is a "keyword" field, so the full answer object is
        JSON-serialized to a string on write and json.loads() back on
        read (search_nearest_feedback above) - keyword fields store the
        full value in _source regardless of length, we just never search
        BY its content, so this works correctly.
      - "option" is always "" - the feedback payload (see
        app/models/schemas.py's FeedbackRequest) has no separate
        conflict-answer field to put there.
      - "document_id"/"chunk_id" are always "" - deliberately NOT
        parsed out of the answer's "references" strings, since that
        would be fragile string-parsing for metadata the cache lookup
        itself never uses (matching is done purely on the embedding).
      - "nearest_feedback_score" is always 1.0, matching the convention
        already found in the index's existing rows - it plays no role
        in search_nearest_feedback()'s actual matching, which computes
        a fresh kNN score at read time regardless of this stored value.
      - there is no dedicated field for the feedback TYPE ("thumsup") -
        none was needed, since a row only ever gets written here when it
        already is one.
    """
    client = get_opensearch_client()

    document = {
        "question": question,
        config.VECTOR_FIELD_NAME: query_embedding,
        "user_id": payload.get("userid", ""),
        "question_id": payload.get("questionid", ""),
        "response": json.dumps(payload.get("responseJSON", {})),
        "option": "",
        "source": "feedback",
        "document_id": "",
        "chunk_id": "",
        "nearest_feedback_score": 1.0,
    }

    client.index(index=config.OPENSEARCH_FEEDBACK_INDEX, body=document, refresh=True)

    logger.info(
        f"[{request_id}] [opensearch_client] Stored feedback into "
        f"{config.OPENSEARCH_FEEDBACK_INDEX!r} for question_id="
        f"{payload.get('questionid')!r}"
    )

