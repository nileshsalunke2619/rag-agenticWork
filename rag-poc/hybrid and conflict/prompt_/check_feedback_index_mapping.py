# =============================================================================
# app/scripts/check_feedback_index_mapping.py
# =============================================================================
# WHY THIS FILE EXISTS:
# create_feedback_index.py found config.OPENSEARCH_FEEDBACK_INDEX already
# existing BEFORE it ever ran - meaning something else created it earlier,
# with an unknown mapping. This is a one-off diagnostic (same pattern as
# every other app/scripts/*.py file) to print that mapping so we can
# confirm whether "embedding" is actually a knn_vector field with
# space_type=cosinesimil, or whether the index needs to be deleted and
# recreated with the correct mapping before the feedback feature can be
# trusted.
#
# RUN WITH:
#   python -m app.scripts.check_feedback_index_mapping
# =============================================================================

import json
import logging

from app.config import settings as config
from app.utils.logging_config import setup_logging
from app.vectordb.opensearch_client import get_opensearch_client

setup_logging()
logger = logging.getLogger(__name__)


def check_mapping() -> None:
    client = get_opensearch_client()

    mapping = client.indices.get_mapping(index=config.OPENSEARCH_FEEDBACK_INDEX)
    logger.info(
        f"[check_feedback_index_mapping] Full mapping for "
        f"{config.OPENSEARCH_FEEDBACK_INDEX!r}:\n{json.dumps(mapping, indent=2)}"
    )

    try:
        doc_count = client.count(index=config.OPENSEARCH_FEEDBACK_INDEX)["count"]
        logger.info(f"[check_feedback_index_mapping] Document count: {doc_count}")
    except Exception as count_error:  # noqa: BLE001 - diagnostic only
        logger.warning(f"[check_feedback_index_mapping] Could not get count: {count_error}")


if __name__ == "__main__":
    check_mapping()
