from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


def count_message_tokens(messages: list) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def trim_context(
    messages: List[Dict], max_tokens: int = 6000, keep_recent: int = 10
) -> Tuple[List[Dict], bool]:
    """
    Trim old messages if context is too large.
    Always keeps system message and most recent messages.
    Returns (trimmed_messages, was_trimmed)
    """
    if not messages:
        return messages, False

    total_tokens = count_message_tokens(messages)

    if total_tokens <= max_tokens:
        return messages, False

    # Always keep system message (first) and recent messages
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    recent_messages = messages[-keep_recent:]

    # Calculate tokens for what we're keeping
    kept_tokens = count_message_tokens(recent_messages)
    if system_msg:
        kept_tokens += estimate_tokens(system_msg.get("content", ""))

    # Build trimmed context
    trimmed = []
    if system_msg:
        trimmed.append(system_msg)

    # Add a summary message about what was trimmed
    num_trimmed = len(messages) - len(recent_messages) - (1 if system_msg else 0)
    if num_trimmed > 0:
        summary = {
            "role": "system",
            "content": f"[Context trimmed: {num_trimmed} older messages removed to stay within token limit]",
        }
        trimmed.append(summary)

    trimmed.extend(recent_messages)

    logger.info(
        f"Context trimmed: {len(messages)} -> {len(trimmed)} messages, {total_tokens} -> {kept_tokens} tokens"
    )

    return trimmed, True
