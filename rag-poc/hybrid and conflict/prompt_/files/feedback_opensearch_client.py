import boto3
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

from app.utils.logger import get_logger
logger = get_logger(__name__)


load_dotenv()
host = os.getenv("OPENSEARCH_HOST")
region = os.getenv("OPENSEARCH_REGION", "ap-southeast-1")
service = os.getenv("OPENSEARCH_SERVICE", "aoss")
index_name = os.getenv("OPENSEARCH_FEEDBACK_INDEX", "kms-index-feedback")

session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=120,
    max_retries=3,
    retry_on_timeout=True
)

def ensure_feedback_index_exists():
    try:
        mapping = {
                    "mappings": {
                        "properties": {
                            "question_id": {"type": "keyword"},
                            "question": {"type": "text"},
                            "chunk_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 1024,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "parameters": {
                                        "ef_construction": 512,
                                        "m": 16
                                    }
                                }
                            },
                            "response": {"type": "text"},
                            "source": {"type": "keyword"},
                            "option": {"type": "keyword"},
                            "nearest_feedback_score": {"type": "float"},
                        }
                    }
                }

        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists.")
    except Exception as e:
        logger.exception(f"Failed to ensure index exists: {e}")
        raise