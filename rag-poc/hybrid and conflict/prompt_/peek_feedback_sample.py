# =============================================================================
# app/scripts/peek_feedback_sample.py
# =============================================================================
# WHY THIS FILE EXISTS:
# check_feedback_index_mapping.py showed kms-index-feedback already has 14
# real documents, with field names (question_id, response, user_id,
# option, source, chunk_id, document_id, nearest_feedback_score) that
# don't match what app/vectordb/opensearch_client.py's store_feedback()/
# search_nearest_feedback() currently assume (questionid, responseJSON,
# userid). Before changing that code, we need to see what's ACTUALLY
# stored in a real document - this is a one-off diagnostic (same pattern
# as every other app/scripts/*.py file) that fetches a couple of real
# documents and prints every field EXCEPT "embedding" (a 1024-number
# vector that would just flood the terminal and isn't useful to read).
#
# RUN WITH:
#   python -m app.scripts.peek_feedback_sample
# =============================================================================

import json
import logging

from app.config import settings as config
from app.utils.logging_config import setup_logging
from app.vectordb.opensearch_client import get_opensearch_client

setup_logging()
logger = logging.getLogger(__name__)


def peek_samples(count: int = 3) -> None:
    client = get_opensearch_client()

    response = client.search(
        index=config.OPENSEARCH_FEEDBACK_INDEX,
        body={"size": count, "query": {"match_all": {}}},
    )

    hits = response["hits"]["hits"]
    logger.info(f"[peek_feedback_sample] Fetched {len(hits)} sample document(s)")

    for i, hit in enumerate(hits):
        source = dict(hit["_source"])
        # Drop the raw vector - 1024 floats, not useful to read, just noise.
        source.pop(config.VECTOR_FIELD_NAME, None)
        logger.info(
            f"[peek_feedback_sample] --- Document {i + 1} (_id={hit['_id']}) ---\n"
            f"{json.dumps(source, indent=2, default=str)}"
        )


if __name__ == "__main__":
    peek_samples()
