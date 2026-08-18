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
import re

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

# The PDF export/viewer this document set comes from stamps a near-identical
# header/footer template onto EVERY page of EVERY document (title block,
# signature table, "Accessed By / Access Date / Doc Name / Uncontrolled
# printed copy..." footer, etc.). When a chunk's real body text is short,
# this boilerplate can dominate its embedding - which is exactly what
# produced the first run's spurious 1.0000-similarity matches between
# completely unrelated documents (they were matching on shared page
# furniture, not shared content). These phrases are stripped before judging
# whether a chunk has enough REAL content to be worth comparing.
BOILERPLATE_PHRASES = [
    "Pfizer Controlled Document",
    "Document Signatures:",
    "Signature Date/Time (GMT) Signature Reason",
    "Accessed By:",
    "Access Date:",
    "Doc Name",
    "Doc Alias",
    "Effective Date",
    "Effective Effective Date",
    "Site Code / Department",
    "Uncontrolled printed copy valid for up to 24hr",
    "Pfizer Internal Use",
    "Unofficial if printed",
    "Viewed on:",
    "GMT Status",
    "GLNS Version",
    "webviewer_admin",
    "Linked to",
    # Found in the second run: these are TEMPLATE table headers/sentences
    # repeated near-verbatim across many sibling documents - real words,
    # but not real content, same problem as the page header/footer.
    "[TABLE START]",
    "Failure Mode Detection Resolution and owner Escalation",
    "Reference Title",
    "Below table gives an overview of the most common SAP failures that could occur in this process flow.",
    "The VCS team is responsible for a proper resolution.",
    # Signature-block "reason" labels - the names next to them still vary
    # per document, but _is_boilerplate() also has a dedicated section=None
    # + "Signature" check below to catch signature pages outright.
    "Author Approval",
    "Departmental Approval",
    "Quality Approval",
    "Manager Approval",
]

# Date/time/version/ID tokens (e.g. "09-Dec-2024 00:11:08", "CD-60346",
# "SOP-104302", "3.0", "Eastern Time") are the other big chunk of this
# boilerplate - they vary per document/page so BOILERPLATE_PHRASES can't
# catch them, but they carry no real content either, so they're stripped by
# pattern instead of stripping actual body text.
BOILERPLATE_PATTERNS = [
    r"\d{1,2}-[A-Za-z]{3}-\d{4}",  # 09-Dec-2024
    r"\d{1,2}:\d{2}:\d{2}",  # 00:11:08
    r"\b[A-Z]{2,4}-\d{3,7}\b",  # CD-60346, SOP-104302
    r"\bv?\d+\.\d+\b",  # 3.0
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\bEastern Time\b",
    r"\bAM\b|\bPM\b",
    r"\bN/A\b",  # empty "N/A" table filler, repeated across many documents
]

# After stripping all of the above, a chunk needs at least this many
# alphabetic characters left to count as having real content worth
# comparing - pure boilerplate/footer chunks and pure ASCII-table-border
# chunks both fail this threshold.
MIN_ALPHA_CHARS = 60

# Sections that are, by name, never useful for conflict-hunting even
# though their content is real text, not short filler. A "Related
# documents" table lists citation IDs/titles - sibling documents that cite
# the same parent SOP produce identical-looking chunks here, but that's
# shared bookkeeping, not two sources describing (let alone disagreeing on)
# the same rule. Checked as a case-insensitive substring against the
# chunk's section name.
SKIP_SECTION_SUBSTRINGS = [
    "related documents",
]


def _is_boilerplate(chunk: dict) -> bool:
    content = chunk.get("content") or ""
    section = (chunk.get("section") or "").lower()

    if any(skip in section for skip in SKIP_SECTION_SUBSTRINGS):
        return True

    # Cover/signature-block pages (section=None, contains "Signature") are
    # structurally boilerplate regardless of how many real proper names
    # they contain - a list of who signed off and when is not SOP content,
    # and names alone can carry enough alphabetic characters to slip past
    # the length check below even after removing "Author Approval" etc.
    if not chunk.get("section") and "Signature" in content:
        return True

    text = content
    for phrase in BOILERPLATE_PHRASES:
        text = text.replace(phrase, " ")
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)

    alpha_chars = sum(1 for c in text if c.isalpha())
    return alpha_chars < MIN_ALPHA_CHARS


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

        skipped_boilerplate = 0
        for hit in hits:
            chunk = _chunk_from_hit(hit)
            if not chunk["embedding"]:
                continue
            if _is_boilerplate(chunk):
                skipped_boilerplate += 1
                continue
            chunks.append(chunk)

        search_after = hits[-1]["sort"]
        logger.info(
            f"Fetched {len(chunks)} real-content chunk(s) so far "
            f"(skipped {skipped_boilerplate} boilerplate-only chunk(s) this page)..."
        )

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
    # Fetch more than MATCHES_PER_CHUNK - the live index still contains the
    # boilerplate-only chunks we filtered out of our local `chunks` list
    # (fetch_all_chunks only controls what WE compare, not what's actually
    # in the index), so the top hit(s) can still be boilerplate. Pulling a
    # small pool lets us skip past those and find the first hit with real
    # content.
    result_pool_size = max(MATCHES_PER_CHUNK, 5)
    body = {
        "size": result_pool_size,
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

    for hit in hits:
        matched_chunk = _chunk_from_hit(hit)
        if _is_boilerplate(matched_chunk):
            continue
        return matched_chunk, hit.get("_score", 0.0)

    return None


def _snippet(text: str, length: int = 300) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:length] + ("..." if len(text) > length else "")


# A genuine Case 2 conflict shares topic vocabulary (driving similarity up)
# but disagrees on the actual instruction (capping similarity below a
# near-perfect match). Score >= 1.0-ish pairs are almost always either
# leftover boilerplate or two documents saying the literal SAME thing
# (agreement, not conflict) - the moderate band below is where a real
# disagreement is actually more likely to show up.
NEAR_DUPLICATE_MIN_SCORE = 0.98
MODERATE_BAND_MIN_SCORE = 0.75
MODERATE_BAND_MAX_SCORE = 0.98


def _print_pairs(pairs: list, title: str, subtitle_lines: list) -> None:
    print("\n" + "=" * 100)
    print(title)
    for line in subtitle_lines:
        print(line)
    print("=" * 100)

    if not pairs:
        print("\n(none found in this range)")
        return

    for rank, (source_chunk, matched_chunk, score) in enumerate(pairs, start=1):
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


def print_report(pairs: list) -> None:
    pairs_sorted = sorted(pairs, key=lambda p: p[2], reverse=True)

    near_duplicates = [p for p in pairs_sorted if p[2] >= NEAR_DUPLICATE_MIN_SCORE]
    moderate_band = [
        p
        for p in pairs_sorted
        if MODERATE_BAND_MIN_SCORE <= p[2] < MODERATE_BAND_MAX_SCORE
    ]

    _print_pairs(
        moderate_band[:TOP_N_TO_REPORT],
        f"MOST LIKELY CONFLICT CANDIDATES - moderate similarity "
        f"({MODERATE_BAND_MIN_SCORE}-{MODERATE_BAND_MAX_SCORE}), same topic but NOT "
        f"identical wording",
        [
            "Read this section FIRST. Same topic, different wording is the",
            "actual signature of a genuine disagreement between two sources -",
            "still a triage list, not confirmed conflicts, but a much better",
            "place to look than the near-duplicate section below.",
        ],
    )

    _print_pairs(
        near_duplicates[:TOP_N_TO_REPORT],
        f"NEAR-DUPLICATE MATCHES - score >= {NEAR_DUPLICATE_MIN_SCORE} "
        f"(sanity check only, not conflict candidates)",
        [
            "These pairs are near-identical text - either leftover boilerplate,",
            "or two documents stating the literal SAME instruction (agreement,",
            "not a conflict). Included so you can confirm dedup is working, not",
            "because these are conflict candidates.",
        ],
    )


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
