# =============================================================================
# app/config/settings.py
# =============================================================================
# WHY THIS FILE EXISTS:
# Every other file in this project needs settings like "which AWS region",
# "which OpenSearch index", "which Claude model". Instead of typing these
# values directly inside app/vectordb/opensearch_client.py or app/llm, we keep
# ALL settings in ONE place: this file.
#
# The actual secret values live in a ".env" file (never committed to git).
# python-dotenv reads that file and loads the values as environment
# variables, and this file just reads them with os.getenv().
# =============================================================================

import os
from dotenv import load_dotenv

# load_dotenv() looks for a file named ".env" in the project's working
# directory and loads every "KEY=value" line inside it into the
# environment. We call this ONCE, here, so every other file can just
# use os.getenv() without needing to load the .env file itself.
load_dotenv()

# -----------------------------------------------------------------------
# AWS region settings
# -----------------------------------------------------------------------
# IMPORTANT: this project talks to AWS resources that live in DIFFERENT
# regions from each other - there is no single region that's correct for
# everything. Each AWS call uses its OWN, separately-configured region.
TITAN_REGION = os.getenv("TITAN_REGION", "us-east-1")
OPENSEARCH_REGION = os.getenv("OPENSEARCH_REGION", "ap-southeast-1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-southeast-1")

# -----------------------------------------------------------------------
# AWS OpenSearch settings
# -----------------------------------------------------------------------
# The OpenSearch endpoint, WITHOUT "https://" in front.
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")

# OpenSearch over HTTPS almost always uses port 443.
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "443"))

# The name of the index where Pipeline 1 stored your chunks + embeddings.
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")

# The AWS service name used to SigV4-sign requests:
#   - AWS OpenSearch SERVICE (managed domains)    -> "es"
#   - AWS OpenSearch SERVERLESS (collections)     -> "aoss"
# Signing with the wrong one produces a request AWS rejects as
# "forbidden" even though the signature itself looks valid.
OPENSEARCH_SERVICE = os.getenv("OPENSEARCH_SERVICE", "aoss")

# The field name INSIDE each OpenSearch document that holds the vector
# (the embedding). This must match whatever field name Pipeline 1 used
# when it stored the embeddings.
VECTOR_FIELD_NAME = os.getenv("VECTOR_FIELD_NAME", "embedding")

# The field name INSIDE each OpenSearch document that holds the original
# chunk text. This must also match what Pipeline 1 used.
TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "content")

# -----------------------------------------------------------------------
# Amazon Titan Embeddings settings
# -----------------------------------------------------------------------
# We call this SAME embedding model to turn the user's QUESTION into a
# vector, so it can be compared against the vectors already stored in
# OpenSearch. Must match what Pipeline 1 used, including the ":0"
# version suffix for v2.
TITAN_EMBEDDING_MODEL_ID = os.getenv(
    "TITAN_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# -----------------------------------------------------------------------
# Anthropic Claude settings (via AWS Bedrock)
# -----------------------------------------------------------------------
# We call Claude through AWS Bedrock's invoke_model() API, so there is
# NO separate Anthropic API key here - authentication uses the same AWS
# credentials as OpenSearch and Titan Embeddings above.
#
# There is NO safe default for this value - it depends entirely on your
# AWS account/region, and may be a short model name OR a full
# "inference profile" ARN. You MUST set ANTHROPIC_MODEL in your .env.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")

# -----------------------------------------------------------------------
# Retrieval settings
# -----------------------------------------------------------------------
# How many chunks to fetch from OpenSearch for every question.
TOP_K = int(os.getenv("TOP_K", "3"))
