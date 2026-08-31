import logging

from app.utils.logger import get_logger
logger = get_logger(__name__)

def min_max_normalize(scores: list) -> list:
    if not scores:
        return []
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

def normalize_results(results: list, score_field: str = "_score") -> list:
    try:
        scores = [hit[score_field] for hit in results]
        normalized = min_max_normalize(scores)
        for hit, norm in zip(results, normalized):
            hit["normalized_score"] = norm
        logger.info("Normalized scores successfully")
        return results
    except Exception as e:
        logger.error(f"Failed to normalize results: {e}")
        raise

def merge_results(keyword_results: list, semantic_results: list, alpha: float = 0.5) -> list:
    try:
        keyword_results = normalize_results(keyword_results)
        semantic_results = normalize_results(semantic_results)

        merged = {}
        for hit in keyword_results:
            doc_id = hit["_id"]
            merged[doc_id] = {
                "source": hit["_source"],
                "keyword_score": hit["normalized_score"],
                "semantic_score": 0.0
            }

        for hit in semantic_results:
            doc_id = hit["_id"]
            if doc_id not in merged:
                merged[doc_id] = {
                    "source": hit["_source"],
                    "keyword_score": 0.0,
                    "semantic_score": hit["normalized_score"]
                }
            else:
                merged[doc_id]["semantic_score"] = hit["normalized_score"]

        for doc_id, data in merged.items():
            data["hybrid_score"] = alpha * data["keyword_score"] + (1 - alpha) * data["semantic_score"]

        logger.info("Merged keyword and semantic results successfully")
        return sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
    except Exception as e:
        logger.error(f"Failed to merge results: {e}")
        raise