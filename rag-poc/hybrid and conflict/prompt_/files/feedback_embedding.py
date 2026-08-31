import boto3
import json
import logging
from dotenv import load_dotenv
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


region = os.getenv("TITAN_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)
embedding_model_id = os.getenv("TITAN_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

#Generates a vector embedding for the give text using aws bedrock embedding model
def get_titan_embedding(text: str) -> list:
    try:
        response = bedrock.invoke_model(
            modelId=embedding_model_id,
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response["body"].read())
        embedding = result.get("embedding", [])
        if not embedding:
            raise ValueError("Empty embedding returned")
        logger.info("Generated Titan v2 embedding successfully")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
