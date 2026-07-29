# =============================================================================
# app/prompts/system_prompt.py
# =============================================================================
# WHY THIS FILE EXISTS:
# The system prompt - instructions that tell Claude HOW to behave before
# it even sees the user's question - is kept in its own file/folder,
# separate from the code that CALLS Claude (app/llm/bedrock_client.py).
# This means anyone who wants to tweak the wording doesn't need to touch
# any calling logic, and vice versa.
# =============================================================================

# Keeping Claude grounded in "only use the provided context" is what
# makes this a RAG system instead of Claude just answering from its own
# general knowledge.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only using the provided context. "
    'If the answer is not found, say: "I don\'t have enough information."'
)
