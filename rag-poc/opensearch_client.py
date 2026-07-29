import json

import boto3
from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import TransportError

import config


def get_opensearch_client() -> OpenSearch:
    """
    WHAT THIS FUNCTION DOES:
    Creates and returns a connected OpenSearch client, authenticated using
    your AWS credentials (the same credentials boto3 already knows about,
    e.g. from environment variables, an AWS profile, or an IAM role).

    WHY WE NEED IT:
    AWS OpenSearch requires requests to be "SigV4 signed" - a way of
    proving the request really comes from someone with valid AWS
    permissions. RequestsAWSV4SignerAuth handles that signing for us.

    IMPORTANT - "es" vs "aoss":
    AWS actually has TWO different OpenSearch offerings that both use the
    word "OpenSearch", but are different services under the hood:
      - AWS OpenSearch SERVICE (managed domains) - service name "es"
      - AWS OpenSearch SERVERLESS (collections)   - service name "aoss"
    They look similar but sign requests differently - using the wrong one
    ("es" against a Serverless collection, or vice versa) produces a
    request that LOOKS validly signed but gets rejected as "forbidden",
    which is exactly what was happening before this was set to "aoss" to
    match this project's actual OpenSearch Serverless collection.
    config.OPENSEARCH_SERVICE controls which one we sign for.

    DIAGNOSTIC PRINTS:
    The print() statements below show EXACTLY which AWS identity and
    which connection settings are being used for every request - this is
    so if something is denied, you have concrete values (not guesses) to
    show whoever manages the OpenSearch access policy.
    """
    # boto3.Session().get_credentials() grabs whatever AWS credentials are
    # already configured on this machine (env vars, ~/.aws/credentials, or
    # an IAM role if running on AWS).
    credentials = boto3.Session().get_credentials()

    # Print exactly which AWS identity (IAM user/role) is about to be
    # used - this is the identity that needs to be allow-listed in the
    # OpenSearch Serverless data access policy.
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
    # AWS regions in this project. Signing with the wrong region produces
    # a "403 Forbidden" that looks like a permissions problem but isn't.
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

    WHY WE NEED IT:
    OpenSearch can only compare vectors to vectors. Our documents are
    already stored as vectors (from Pipeline 1). So before we can search,
    we must convert the user's plain-text question into a vector using
    the SAME embedding model, so the two vectors "speak the same language".
    """
    # bedrock-runtime is the AWS client used to actually RUN a model
    # (as opposed to "bedrock", which is used to manage/list models).
    # NOTE: config.TITAN_REGION, not OPENSEARCH_REGION or BEDROCK_REGION -
    # Titan Embeddings may live in a different AWS region than OpenSearch
    # or Claude in this project.
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.TITAN_REGION)

    # Titan Embeddings expects a JSON body shaped like: {"inputText": "..."}
    request_body = json.dumps({"inputText": text})

    response = bedrock_runtime.invoke_model(
        modelId=config.TITAN_EMBEDDING_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    # The response body is a stream of bytes containing JSON - we read it
    # and parse it to get the actual embedding list out.
    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


def search_top_chunks(query_embedding: list, top_k: int = None) -> list:
    """
    WHAT THIS FUNCTION DOES:
    Sends a "k-NN" (k-Nearest-Neighbors) search to OpenSearch: given a
    query vector, find the `top_k` stored documents whose vectors are
    closest (most similar in meaning) to it.

    WHY WE NEED IT:
    This is the actual "retrieval" step of Retrieval-Augmented Generation
    (RAG) - it's how we find the chunks of your original documents that
    are most relevant to the user's question.

    KEEPING IT SIMPLE (on purpose, per the project requirements):
    - No filters
    - No hybrid search (text search + vector search combined)
    - No re-ranking
    Just a plain k-NN vector search for the top_k closest chunks.
    """
    if top_k is None:
        top_k = config.TOP_K

    client = get_opensearch_client()

    # This is the simplest possible OpenSearch k-NN query shape:
    # "search the vector field for the k closest vectors to mine."
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
        # .info is normally a dict with AWS's detailed reason - print it
        # fully formatted so it's easy to read and easy to copy/paste.
        try:
            print(f"    info:        {json.dumps(search_error.info, indent=4)}")
        except TypeError:
            # .info isn't always JSON-serializable (e.g. a raw exception) -
            # fall back to printing it as-is rather than crashing here.
            print(f"    info:        {search_error.info!r}")
        raise

    # response["hits"]["hits"] is a list of matching documents.
    # Each one has a "_source" dict containing the original fields we
    # stored in Pipeline 1 - we just want the text field out of each.
    hits = response["hits"]["hits"]
    chunks = [hit["_source"].get(config.TEXT_FIELD_NAME, "") for hit in hits]

    return chunks
