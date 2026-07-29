import os
from dotenv import load_dotenv

# load_dotenv() looks for a file named ".env" in the project folder and
# loads every "KEY=value" line inside it into the environment.
# We call this ONCE, here, so every other file can just use os.getenv().
load_dotenv()

# -----------------------------------------------------------------------
# AWS region settings
# -----------------------------------------------------------------------
# IMPORTANT: this project talks to AWS resources that live in DIFFERENT
# regions from each other. There is NO single "AWS_REGION" that's
# correct for everything - using one shared region for all of them was
# the actual root cause of a long-running "403 Forbidden" bug: OpenSearch
# requests were being SigV4-signed for the wrong region, which produces
# a request that LOOKS validly signed but AWS silently rejects.
#
# Each AWS call in this project now uses its OWN, separately-configured
# region - set each one to match what your AWS admin/team confirms:
#   - TITAN_REGION      -> where your Bedrock Titan Embeddings model runs
#   - OPENSEARCH_REGION -> where your OpenSearch collection/domain lives
#   - BEDROCK_REGION    -> where your Claude model / inference profile
#                          lives (check the region inside your Claude
#                          model ID/ARN - it must match this)
TITAN_REGION = os.getenv("TITAN_REGION", "us-east-1")
OPENSEARCH_REGION = os.getenv("OPENSEARCH_REGION", "ap-southeast-1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-southeast-1")

# -----------------------------------------------------------------------
# AWS OpenSearch settings
# -----------------------------------------------------------------------
# The OpenSearch domain endpoint, WITHOUT "https://" in front.
# Example: "search-my-domain-abc123.us-east-1.es.amazonaws.com"
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")

# OpenSearch over HTTPS almost always uses port 443.
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "443"))

# The name of the index where Pipeline 1 stored your chunks + embeddings.
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")

# The AWS service name used to SigV4-sign requests. AWS has two different
# OpenSearch offerings under one name:
#   - AWS OpenSearch SERVICE (managed domains)    -> "es"
#   - AWS OpenSearch SERVERLESS (collections)     -> "aoss"
# Signing with the wrong one produces a request that AWS rejects as
# "forbidden" even though the signature itself looks valid - this must
# match whichever one Pipeline 1's OpenSearch actually is. Defaulting to
# "aoss" (Serverless) since that's what this project's OpenSearch turned
# out to be.
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
# OpenSearch. If Pipeline 1 used a different Titan model, change this to
# match it exactly - otherwise the vectors won't be comparable.
# This project uses Titan Embeddings v2 (confirmed - lives in
# TITAN_REGION=us-east-1). Note ":0" at the end - that's part of the
# real model ID for v2, unlike v1 which has no version suffix.
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
# IMPORTANT: There is NO safe default for this value - it depends
# entirely on your AWS account/region. Depending on how your account is
# set up, this might be:
#   - a short model name, e.g. "anthropic.claude-sonnet-4-6", OR
#   - a full "inference profile" ARN, e.g.
#     "arn:aws:bedrock:ap-southeast-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-6"
#
# Ask your AWS admin/team for the exact value, or find it yourself with:
#   aws bedrock list-foundation-models --region <region> --by-provider anthropic
#   aws bedrock list-inference-profiles --region <region>
# You MUST set ANTHROPIC_MODEL in your .env file - there is no fallback
# default here on purpose, since a wrong guess just produces a confusing
# "invalid model identifier" error instead of a clear "not set" error.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")

# -----------------------------------------------------------------------
# Retrieval settings
# -----------------------------------------------------------------------
# How many chunks to fetch from OpenSearch for every question.
TOP_K = int(os.getenv("TOP_K", "3"))
