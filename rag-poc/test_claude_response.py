# =============================================================================
# test_claude_response.py
# =============================================================================
# WHY THIS FILE EXISTS:
# A small, standalone script to test ONE real question against the real
# OpenSearch + real Claude (via Bedrock) pipeline, and print exactly what
# comes back - including whether Claude's answer got CUT OFF (truncated)
# before finishing, which is what causes "Unterminated string" errors when
# we try to json.loads() it.
#
# HOW TO RUN IT:
#   python test_claude_response.py
#
# (Run it from the KMSChatbotProjAgenticAI folder - the same folder that
# contains main.py - so the "app" package can be found. Make sure your
# .env file is in this same folder too.)
# =============================================================================

import json

from app.vectordb import opensearch_client
from app.llm.bedrock_client import get_raw_claude_text, bedrock_runtime, build_user_prompt
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.config import settings as config


def main():
    # Change this to whatever question you want to test.
    question = "How to create order?"

    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print("\nSTEP 1: Retrieving real chunks from OpenSearch...")
    query_embedding = opensearch_client.get_query_embedding(question)
    chunks = opensearch_client.search_top_chunks(query_embedding)
    print(f"Retrieved {len(chunks)} chunk(s).")

    print("\n" + "=" * 80)
    print("STEP 2: Calling Claude via Bedrock directly (so we can see stop_reason)...")
    print("=" * 80)

    # This duplicates get_raw_claude_text()'s request, but calls
    # invoke_model() directly here so we can also read response_body["stop_reason"] -
    # get_raw_claude_text() itself only returns the text, not the full response.
    user_prompt = build_user_prompt(question, chunks)
    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,  # <- deliberately kept at 1024 to REPRODUCE the issue
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    )
    response = bedrock_runtime.invoke_model(
        modelId=config.ANTHROPIC_MODEL,
        body=request_body,
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())

    raw_text = ""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            raw_text = block["text"]
            break

    print(f"\nstop_reason: {response_body.get('stop_reason')!r}")
    print(f"raw_text length (characters): {len(raw_text)}")

    print("\nRAW CLAUDE TEXT:")
    print("-" * 80)
    print(raw_text)
    print("-" * 80)

    print("\n" + "=" * 80)
    print("STEP 3: Trying json.loads(raw_text)...")
    print("=" * 80)
    try:
        parsed = json.loads(raw_text)
        print("SUCCESS - parsed into a dict:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"FAILED: {e}")
        if response_body.get("stop_reason") == "max_tokens":
            print(
                "\n>>> CONFIRMED: Claude's answer was CUT OFF because it hit the "
                "max_tokens limit (1024) before finishing its JSON response. "
                "This is why json.loads() fails with 'Unterminated string'. "
                "Fix: raise max_tokens in app/llm/bedrock_client.py's "
                "get_raw_claude_text() function (e.g. to 4096)."
            )
        else:
            print(
                "\n>>> stop_reason is NOT 'max_tokens', so this is NOT a "
                "truncation issue - the JSON is malformed for some other "
                "reason (e.g. Claude included literal unescaped newlines "
                "inside a string value). Share this raw text so we can "
                "look at exactly what Claude wrote."
            )


if __name__ == "__main__":
    main()
