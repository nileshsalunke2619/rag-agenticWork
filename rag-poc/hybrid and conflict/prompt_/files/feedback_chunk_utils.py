import uuid
import logging

from app.utils.logger import get_logger
logger = get_logger(__name__)

def generate_chunk_id(question_id: str) -> str:
    try:
        chunk_id = f"{question_id}_chunk_{uuid.uuid4().hex[:6]}"
        logger.info(f"Generated chunk_id: {chunk_id}")
        return chunk_id
    except Exception as e:
        logger.error(f"Failed to generate chunk_id: {e}")
        raise

def generate_chunk_ids(question_id: str, num_chunks: int) -> list:
    try:
        ids = [f"{question_id}_chunk_{uuid.uuid4().hex[:6]}" for _ in range(num_chunks)]
        logger.info(f"Generated {num_chunks} chunk_ids for {question_id}")
        return ids
    except Exception as e:
        logger.error(f"Failed to generate multiple chunk_ids: {e}")
        raise