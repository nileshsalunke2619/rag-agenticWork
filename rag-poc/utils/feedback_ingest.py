
import json
import boto3
import os
import traceback
from app.tools.feedback_opensearch_client import client
from app.vectordb.feedback_embedding import get_titan_embedding
from app.utils.feedback_chunk_utils import generate_chunk_id
from app.utils.text_utils import clean_response_text,extract_reference_documents,build_response_json,normalize_response
from app.models.feedback_request_model import IngestRequest
#from app.models.feedback_request_model import FeedbackRequest
from app.utils.json_utils import (
    sanitize_payload,
    validate_json_payload
)

from app.utils.logger import get_logger
logger = get_logger(__name__)


def ingest_question(request: IngestRequest):
    try:

        nearest_feedback_score = 0.0

        # First check if any docs exist with same document_id
        index_name = os.getenv("OPENSEARCH_FEEDBACK_INDEX", "kms-index-feedback")
        #ensure_feedback_index_exists() 

        embedding = get_titan_embedding(request.question)
        chunk_id  = generate_chunk_id(request.questionid)             
               
        safe_response = request.response
        response_for_extraction = safe_response

        if isinstance(safe_response, str):
            safe_response = clean_response_text(
                safe_response
            )
            try:
                parsed_response = json.loads(
                    safe_response
                )
                response_for_extraction = parsed_response
            except Exception:
                response_for_extraction = safe_response
        elif isinstance(safe_response, (dict, list)):
            response_for_extraction = safe_response
            safe_response = json.dumps(
                safe_response,
                ensure_ascii=False
            )


        document_id = (
            request.documentid
            or extract_reference_documents(
                response_for_extraction
            )
        )
        
        #document_id = request.documentid or extract_reference_documents(safe_response) 
        

        print("from feedback_ingest.py the parameter passes as:  \n",safe_response)
        logger.info(
                f"Response Type: {type(safe_response).__name__}"
            )
        logger.info(f"Extracted document_id: {document_id}")


        # GETTING nearest_feedback_score
        
        try:
            
            knn_response = client.search(
                index=index_name,
                body={
                    "size": 1,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": 1
                            }
                        }
                    }
                }
            )

            myhits = knn_response["hits"]["hits"]

            if myhits:

                nearest_feedback_score = float(
                    myhits[0]["_score"]
                )

                nearest_question = (
                        myhits[0]["_source"]
                        .get(
                            "question_id",
                            myhits[0]["_source"].get(
                                "questionid"
                            )
                        )
                    )

                logger.info(
                    f"Nearest Question={nearest_question}, "
                    f"Score={nearest_feedback_score}"
                )

        except Exception as e:

            logger.warning(
                f"Failed to calculate nearest feedback score: {e}"
            )


        
        # First check if any docs exist with same document_id
        existing = client.search(
            index=index_name,
            body={
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }
        )

        #total_hits = existing["hits"]["total"]["value"]
        total_hits = existing["hits"]["total"]

        if isinstance(total_hits, dict):
            total_hits = total_hits.get("value", 0)

        logger.info(
                    f"Found existing {total_hits} documents in vector index "
                    f"for document_id={document_id}"
        )

        if total_hits > 0:
            for hit in existing["hits"]["hits"]:
                client.update(
                    index=index_name,
                    id=hit["_id"],
                    body={
                        "doc": {
                            "response": "",
                            "option": ""
                        }
                    }
                )

        
        doc = {
            "question"   : request.question,
            "question_id": request.questionid,
            "document_id": document_id,
            "user_id"    : request.userid,
            "embedding"  : embedding,
            "response"   : safe_response,
            "chunk_id"   : chunk_id,
            "source"     : request.source,
            "option"     : request.option,
            "nearest_feedback_score": nearest_feedback_score
        }

        doc = sanitize_payload(doc)

        is_valid, validation_result = validate_json_payload(
            doc
        )

        if not is_valid:
            raise ValueError(
                f"Invalid JSON payload: {validation_result}"
            )

        logger.info(
            "JSON validation successful"
        )
        
        result = client.index(index=index_name, body=doc, id=request.questionid)
        logger.info(
            f"Document indexed successfully: "
            f"{json.dumps(result, ensure_ascii=False, indent=2)}"
        )
        return result
    
    except Exception as e:

        logger.exception(
            f"Failed to ingest document: {e}"
        )       

        raise