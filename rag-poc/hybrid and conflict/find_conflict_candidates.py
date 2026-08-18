# =============================================================================
# app/scripts/find_conflict_candidates.py
# =============================================================================
# WHY THIS SCRIPT EXISTS:
# We repeatedly tried to find a genuine Case 2 conflict (two SOPs giving
# different instructions for the same condition) in the real Pfizer document
# set, to properly test the "answeroption" conflict feature. Manually reading
# individual documents and guessing from titles both proved unreliable - the
# only confirmed conflict so far is between two documents that were
# deliberately authored as a matched pair for testing.
#
# This script does NOT find conflicts itself - it can't judge whether two
# chunks actually contradict each other, only a person who knows the real
# business rule can. What it DOES do is the tedious part: scan every chunk in
# the live index and surface pairs from DIFFERENT documents that are
# topically very similar (the necessary precondition for either a Case 1
# synthesis or a genuine Case 2 conflict), ranked by similarity. A human then
# reads the top of that ranked list instead of the entire corpus.
#
# HOW TO RUN:
# From the project root, with the same environment/AWS role the FastAPI app
# already uses:
#     python -m app.scripts.find_conflict_candidates
#
# This is a ONE-OFF diagnostic script - run it once, read the printed report,
# share it for review. It is not wired into the FastAPI app and nothing else
# imports it.
#
# WHY IT DOESN'T CALL BEDROCK/TITAN:
# Every chunk already has its "embedding" vector stored in the index from
# ingestion. Reusing those stored vectors instead of re-embedding text keeps
# this script simple and avoids extra Bedrock calls entirely.
#
# FALLBACK NOTE (not implemented - only needed if this turns out too slow):
# This does one kNN round-trip per chunk, which is fine for a one-off run but
# could be slow on a very large index. If so, switch to sampling ONE
# representative chunk per unique document_name (e.g. via a `terms`
# aggregation on config.DOCUMENT_NAME_FIELD_NAME) instead of every chunk -
# trades completeness for speed.
# =============================================================================

import logging

from app.config import settings as config
from app.vectordb.opensearch_client import get_opensearch_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# How many chunks to pull per page while scanning the whole index.
PAGE_SIZE = 200

# How many nearest neighbors to ask OpenSearch for internally, BEFORE
# excluding same-document hits - needs to be generous, since the closest
# few neighbors of any chunk are usually other chunks from its OWN document
# (adjacent sections of the same SOP), which then get filtered out.
INTERNAL_KNN_K = 15

# How many cross-document matches to actually keep per source chunk.
MATCHES_PER_CHUNK = 1

# How many top-ranked pairs to print in the final report.
TOP_N_TO_REPORT = 30

SOURCE_FIELDS = [
    config.DOCUMENT_NAME_FIELD_NAME,
    config.DOCUMENT_ID_FIELD_NAME,
    config.SECTION_FIELD_NAME,
    config.PAGE_FIELD_NAME,
    config.TEXT_FIELD_NAME,
    config.CHUNK_ID_FIELD_NAME,
    config.VECTOR_FIELD_NAME,
]


def _chunk_from_hit(hit: dict) -> dict:
    source = hit.get("_source", {})
    return {
        "id": hit.get("_id"),
        "document_name": source.get(config.DOCUMENT_NAME_FIELD_NAME, ""),
        "document_id": source.get(config.DOCUMENT_ID_FIELD_NAME, ""),
        "section": source.get(config.SECTION_FIELD_NAME, ""),
        "page": source.get(config.PAGE_FIELD_NAME, ""),
        "content": source.get(config.TEXT_FIELD_NAME, ""),
        "chunk_id": source.get(config.CHUNK_ID_FIELD_NAME, ""),
        "embedding": source.get(config.VECTOR_FIELD_NAME),
    }


def fetch_all_chunks(client) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Paginates through every chunk in the index using search_after (NOT the
    classic _search/scroll API - OpenSearch Serverless doesn't support
    scroll), collecting the fields we need plus each chunk's own stored
    embedding vector.
    """
    chunks = []
    search_after = None

    while True:
        body = {
            "size": PAGE_SIZE,
            "query": {"match_all": {}},
            "sort": [{"_id": "asc"}],
            "_source": SOURCE_FIELDS,
        }
        if search_after is not None:
            body["search_after"] = search_after

        response = client.search(index=config.OPENSEARCH_INDEX, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            chunk = _chunk_from_hit(hit)
            if chunk["embedding"]:
                chunks.append(chunk)

        search_after = hits[-1]["sort"]
        logger.info(f"Fetched {len(chunks)} chunk(s) so far...")

        if len(hits) < PAGE_SIZE:
            break

    return chunks


def find_cross_document_match(client, chunk: dict):
    """
    WHAT THIS FUNCTION DOES:
    Runs a kNN search against config.VECTOR_FIELD_NAME using this chunk's
    OWN stored embedding, excluding hits from this chunk's own document via
    a must_not term filter - so whatever comes back is the most similar
    content from a DIFFERENT document. Returns (matched_chunk, score) for
    the best cross-document match, or None if nothing came back (e.g. a
    single-document index, or every neighbor was from the same document).
    """
    body = {
        "size": MATCHES_PER_CHUNK,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            config.VECTOR_FIELD_NAME: {
                                "vector": chunk["embedding"],
                                "k": INTERNAL_KNN_K,
                            }
                        }
                    }
                ],
                "must_not": [
                    {"term": {config.DOCUMENT_NAME_FIELD_NAME: chunk["document_name"]}}
                ],
            }
        },
        "_source": SOURCE_FIELDS,
    }

    response = client.search(index=config.OPENSEARCH_INDEX, body=body)
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None

    best_hit = hits[0]
    return _chunk_from_hit(best_hit), best_hit.get("_score", 0.0)


def _snippet(text: str, length: int = 300) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:length] + ("..." if len(text) > length else "")


def print_report(pairs: list) -> None:
    pairs_sorted = sorted(pairs, key=lambda p: p[2], reverse=True)
    top = pairs_sorted[:TOP_N_TO_REPORT]

    print("\n" + "=" * 100)
    print(f"TOP {len(top)} CROSS-DOCUMENT TOPICAL MATCHES (highest similarity first)")
    print("This is a triage list, NOT a list of confirmed conflicts - read the")
    print("content of each pair yourself to judge whether it's a genuine")
    print("contradiction (Case 2), a complementary synthesis (Case 1), or an")
    print("unrelated false positive.")
    print("=" * 100)

    for rank, (source_chunk, matched_chunk, score) in enumerate(top, start=1):
        print(f"\n#{rank}  score={score:.4f}")
        print(
            f"  A: {source_chunk['document_name']!r} "
            f"(section={source_chunk['section']!r}, page={source_chunk['page']!r})"
        )
        print(f"     {_snippet(source_chunk['content'])}")
        print(
            f"  B: {matched_chunk['document_name']!r} "
            f"(section={matched_chunk['section']!r}, page={matched_chunk['page']!r})"
        )
        print(f"     {_snippet(matched_chunk['content'])}")


def main() -> None:
    client = get_opensearch_client()

    logger.info(f"Scanning index {config.OPENSEARCH_INDEX!r} for all chunks...")
    chunks = fetch_all_chunks(client)
    logger.info(f"Fetched {len(chunks)} total chunk(s). Finding cross-document matches...")

    pairs = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            result = find_cross_document_match(client, chunk)
        except Exception:
            logger.exception(
                f"Failed to find a match for chunk {chunk['chunk_id']!r} "
                f"in document {chunk['document_name']!r} - skipping."
            )
            continue

        if result is not None:
            matched_chunk, score = result
            pairs.append((chunk, matched_chunk, score))

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(chunks)} chunk(s)...")

    logger.info(f"Done. Found {len(pairs)} cross-document match(es).")
    print_report(pairs)


if __name__ == "__main__":
    main()
