# =============================================================================
# app/vectordb/opensearch_client.py
# =============================================================================
# WHY THIS FILE EXISTS:
# This is the ONLY file that talks to AWS OpenSearch and AWS Bedrock
# (Titan Embeddings). It lives in its own `vectordb/` package inside
# `app/` since it's specifically "the vector database integration",
# separate from the graph/LLM/API logic elsewhere in the app.
#
# What this file does, in plain English:
#   1. Turns the user's question into a vector using Titan Embeddings.
#   2. Sends that vector to OpenSearch and asks for the top-k closest
#      chunks.
#   3. Returns just the plain text of those chunks.
# =============================================================================

import json

import boto3
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import TransportError

# "as config" keeps every reference below (config.OPENSEARCH_HOST, etc.)
# identical to how this file worked before the folder reorganization -
# only the import path changed, not how settings are used.
from app.config import settings as config


def get_opensearch_client() -> OpenSearch:
    """
    WHAT THIS FUNCTION DOES:
    Creates and returns a connected OpenSearch client, authenticated
    using your AWS credentials.

    IMPORTANT - "es" vs "aoss":
    AWS has TWO different OpenSearch offerings that sign requests
    differently - "es" (managed domains) vs "aoss" (Serverless
    collections). config.OPENSEARCH_SERVICE controls which one we sign
    for; getting this wrong produces a "403 Forbidden" that looks like a
    permissions problem but isn't.

    DIAGNOSTIC PRINTS:
    The print() statements below show EXACTLY which AWS identity and
    connection settings are being used for every request, so if
    something is denied, you have concrete values to check against your
    OpenSearch access policy.
    """
    credentials = boto3.Session().get_credentials()

    try:
        identity = boto3.client(
            "sts", region_name=config.OPENSEARCH_REGION
        ).get_caller_identity()
        print("[opensearch_client] Calling AWS as identity:")
        print(f"    Account: {identity.get('Account')}")
        print(f"    Arn:     {identity.get('Arn')}")
    except Exception as identity_error:  # noqa: BLE001 - diagnostic only
        print(f"[opensearch_client] Could NOT resolve AWS identity: {identity_error}")

    print("[opensearch_client] Connection settings being used:")
    print(f"    OPENSEARCH_HOST:    {config.OPENSEARCH_HOST}")
    print(f"    OPENSEARCH_PORT:    {config.OPENSEARCH_PORT}")
    print(f"    OPENSEARCH_INDEX:   {config.OPENSEARCH_INDEX}")
    print(f"    OPENSEARCH_SERVICE: {config.OPENSEARCH_SERVICE}")
    print(f"    OPENSEARCH_REGION:  {config.OPENSEARCH_REGION}")

    # IMPORTANT: this MUST be config.OPENSEARCH_REGION - not a general
    # "AWS_REGION" - because OpenSearch and Bedrock can live in different
    # AWS regions in this project.
    auth = RequestsAWSV4SignerAuth(
        credentials, config.OPENSEARCH_REGION, config.OPENSEARCH_SERVICE
    )

    client = OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client


def get_query_embedding(text: str) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends `text` (the user's question) to Amazon Titan Embeddings (via
    AWS Bedrock) and returns the embedding - a list of numbers that
    represents the MEANING of that text.
    """
    # NOTE: config.TITAN_REGION, not OPENSEARCH_REGION or BEDROCK_REGION -
    # Titan Embeddings may live in a different AWS region than OpenSearch
    # or Claude in this project.
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.TITAN_REGION)

    request_body = json.dumps({"inputText": text})

    response = bedrock_runtime.invoke_model(
        modelId=config.TITAN_EMBEDDING_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


def search_top_chunks(query_embedding: list, top_k: int = None) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends a k-NN search to OpenSearch: given a query vector, find the
    `top_k` stored documents whose vectors are closest to it.

    KEEPING IT SIMPLE (on purpose): no filters, no hybrid search, no
    re-ranking - just a plain k-NN vector search for the top_k closest
    chunks.
    """
    if top_k is None:
        top_k = config.TOP_K

    client = get_opensearch_client()

    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                config.VECTOR_FIELD_NAME: {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        },
    }

    print(f"[opensearch_client] Searching index: {config.OPENSEARCH_INDEX!r}")
    print(
        f"[opensearch_client] Query shape: knn on field "
        f"{config.VECTOR_FIELD_NAME!r}, k={top_k}, vector length="
        f"{len(query_embedding)}"
    )

    # DIAGNOSTIC: we wrap this call so that if OpenSearch rejects it, we
    # print the FULL detail AWS actually sent back - not just "403
    # forbidden" - before letting the error continue up (re-raising),
    # so FastAPI still reports the failure like normal.
    try:
        response = client.search(index=config.OPENSEARCH_INDEX, body=search_query)
    except TransportError as search_error:
        print("[opensearch_client] SEARCH FAILED - full AWS error detail below:")
        print(f"    status_code: {search_error.status_code}")
        print(f"    error:       {search_error.error}")
        try:
            print(f"    info:        {json.dumps(search_error.info, indent=4)}")
        except TypeError:
            print(f"    info:        {search_error.info!r}")
        raise

    hits = response["hits"]["hits"]
    chunks = [hit["_source"].get(config.TEXT_FIELD_NAME, "") for hit in hits]

    return chunks
