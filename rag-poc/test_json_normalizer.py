# =============================================================================
# test_json_normalizer.py
# =============================================================================
# WHY THIS FILE EXISTS:
# A small, standalone script that runs the SAME two real steps our full
# API does - retrieve real chunks from OpenSearch, then get a real
# answer from Claude - and prints every stage to your terminal, so you
# can see EXACTLY what get_json_string() receives and returns, without
# needing Postman or the FastAPI server running.
#
# IMPORTANT: this makes REAL calls to AWS (OpenSearch + Bedrock), so it
# needs your real .env file - the same setup the main app already uses.
#
# HOW TO RUN IT:
#   python test_json_normalizer.py
#
# (Run it from the project root - the same folder that contains llm.py,
# opensearch_client.py, and the "utils" folder.)
# =============================================================================

import opensearch_client
from llm import get_raw_claude_text
from utils.json_normalizer_dynamic import get_json_string


def main():
    """
    WHAT THIS FUNCTION DOES, STEP BY STEP:
    1. Takes a real question.
    2. Retrieves REAL chunks from OpenSearch for that question - the
       exact same call retrieve_node makes in nodes.py.
    3. Sends the question + those REAL chunks to Claude via
       get_raw_claude_text() - a REAL Bedrock call, same as the full
       app makes - and prints Claude's answer exactly as Claude wrote
       it, before any cleanup.
    4. Passes THAT real LLM response into get_json_string() and prints
       the cleaned-up result - this is "the llm response" going into
       get_json_string, not the question.
    """
    # Change this to whatever real question you want to test against
    # your actual OpenSearch index.
    question = "How to fill out FORM 7 for batch certification"

    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print()
    print("STEP 1: Retrieving real chunks from OpenSearch...")
    query_embedding = opensearch_client.get_query_embedding(question)
    chunks = opensearch_client.search_top_chunks(query_embedding)
    print(f"Retrieved {len(chunks)} chunk(s).")

    print()
    print("=" * 80)
    print("STEP 2: Calling Claude via Bedrock (get_raw_claude_text)...")
    print("=" * 80)
    # THIS is "the llm response" - Claude's real answer, based on the
    # real retrieved chunks above.
    llm_response = get_raw_claude_text(question, chunks)

    print()
    print("RAW LLM RESPONSE (exactly what Claude sent back, before any cleanup):")
    print("-" * 80)
    print(llm_response)

    print()
    print("=" * 80)
    print("STEP 3: Passing the LLM RESPONSE through get_json_string...")
    print("=" * 80)
    # We pass the LLM's actual response here - NOT the question - into
    # get_json_string(), the same way llm.py's ask_claude() does.
    filtered_json = get_json_string({"answer": llm_response})

    print()
    print("OUTPUT (what get_json_string returned):")
    print("-" * 80)
    print(filtered_json)


# This "if __name__ == ...:" guard means main() only runs when you
# execute this file directly (python test_json_normalizer.py) - not if
# this file ever gets imported from somewhere else.
if __name__ == "__main__":
    main()
