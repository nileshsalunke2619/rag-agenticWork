# =============================================================================
# app/scripts/create_feedback_index.py
# =============================================================================
# WHY THIS FILE EXISTS:
# A one-off setup script, run ONCE against your live OpenSearch
# environment (same pattern as app/scripts/find_conflict_candidates.py) -
# NOT imported by the running FastAPI app. It creates
# config.OPENSEARCH_FEEDBACK_INDEX with the exact mapping the feedback
# feature needs:
#   - "embedding": a knn_vector field mapped with
#     "space_type": "cosinesimil", so a plain kNN query against it
#     returns OpenSearch's native cosine score, normalized 0-1 - the
#     number app/vectordb/opensearch_client.py's search_nearest_feedback()
#     compares directly against config.FEEDBACK_SIMILARITY_THRESHOLD.
#   - "responseJSON" mapped with "enabled": false - this tells OpenSearch
#     NOT to index the fields inside it individually (we never search
#     BY answer content), but the raw JSON is still stored in _source
#     and comes back untouched on a hit. Storing it as a plain "object"
#     without "enabled": false would make OpenSearch try to map every
#     nested field, which can fail or drift as answer JSON shapes vary.
#
# WITHOUT this script, the feedback index would only ever get created by
# OpenSearch's dynamic mapping the first time a document is indexed into
# it - which would guess a DIFFERENT (and wrong) space_type for the
# vector field, silently breaking the ">90%" threshold's meaning. Run
# this BEFORE the first real thumbsup comes in.
#
# ONE THING THIS SCRIPT CANNOT VERIFY FROM HERE: whether your OpenSearch
# Serverless (AOSS) collection is a "VECTORSEARCH" type collection (the
# only kind that supports knn_vector fields at all). If index creation
# fails with a mapping/engine error, that's the first thing to check
# against your AWS console - not something guessable from this
# environment.
#
# RUN WITH:
#   python -m app.scripts.create_feedback_index
# =============================================================================

import logging

from opensearchpy.exceptions import TransportError

from app.config import settings as config
from app.utils.logging_config import setup_logging
from app.vectordb.opensearch_client import get_opensearch_client, get_query_embedding

setup_logging()
logger = logging.getLogger(__name__)


def _infer_embedding_dimension() -> int:
    """
    WHAT THIS FUNCTION DOES:
    Rather than hardcoding a guessed embedding dimension (Titan Text
    Embeddings v2 can be configured to output 256, 512, or 1024 numbers
    depending on account/model settings), this calls the SAME
    get_query_embedding() the live app uses, on a throwaway string, and
    measures the real length it returns - guaranteed to match whatever
    Pipeline 1 / this project actually uses, with no risk of drift
    between a hardcoded number and reality.
    """
    sample_embedding = get_query_embedding("dimension probe")
    dimension = len(sample_embedding)
    logger.info(f"[create_feedback_index] Inferred embedding dimension: {dimension}")
    return dimension


def create_feedback_index() -> None:
    client = get_opensearch_client()

    if client.indices.exists(index=config.OPENSEARCH_FEEDBACK_INDEX):
        logger.info(
            f"[create_feedback_index] Index {config.OPENSEARCH_FEEDBACK_INDEX!r} "
            f"already exists - doing nothing. Delete it manually first if you need "
            f"to recreate it with a different mapping."
        )
        return

    dimension = _infer_embedding_dimension()

    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                config.VECTOR_FIELD_NAME: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "cosinesimil",
                    },
                },
                "question": {"type": "text"},
                "userid": {"type": "keyword"},
                "questionid": {"type": "keyword"},
                "feedback": {"type": "keyword"},
                "responseJSON": {"type": "object", "enabled": False},
            }
        },
    }

    try:
        client.indices.create(index=config.OPENSEARCH_FEEDBACK_INDEX, body=index_body)
    except TransportError as create_error:
        logger.error(
            f"[create_feedback_index] Index creation FAILED - "
            f"status_code: {create_error.status_code}, error: {create_error.error}"
        )
        try:
            import json

            logger.error(f"    info: {json.dumps(create_error.info, indent=4)}")
        except TypeError:
            logger.error(f"    info: {create_error.info!r}")
        raise

    logger.info(
        f"[create_feedback_index] Created index {config.OPENSEARCH_FEEDBACK_INDEX!r} "
        f"with dimension={dimension}, space_type=cosinesimil, engine=faiss"
    )


if __name__ == "__main__":
    create_feedback_index()
