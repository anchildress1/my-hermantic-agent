from src.core.utils import count_message_tokens, trim_context


def test_token_estimations_and_trim():
    msgs = []
    # system message always present
    msgs.append({"role": "system", "content": "system prompt"})
    # Add 30 messages with content to exceed token thresholds
    for _ in range(30):
        msgs.append({"role": "user", "content": "x" * 500})

    total = count_message_tokens(msgs)
    assert total > 0

    trimmed, was_trimmed = trim_context(msgs, max_tokens=500, keep_recent=5)
    assert was_trimmed
    assert len(trimmed) <= 1 + 5 + 1  # system + summary + recent
