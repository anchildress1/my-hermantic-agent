from unittest.mock import patch

from src.services.llm.ollama_service import OllamaService


def test_check_connection_success():
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"model": "llama3.2"}]}
        service = OllamaService(model="llama3.2")
        assert service.check_connection() is True


def test_check_connection_failure():
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": []}
        service = OllamaService(model="llama3.2")
        assert service.check_connection() is False


def test_check_connection_handles_exceptions():
    with patch("ollama.list", side_effect=RuntimeError("offline")):
        service = OllamaService(model="llama3.2")
        assert service.check_connection() is False


def test_check_connection_allows_partial_model_matches():
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"model": "llama3.2:latest"}]}
        service = OllamaService(model="llama3.2")
        assert service.check_connection() is True


def test_chat_forwards_payload_to_ollama():
    expected = iter([{"message": {"content": "ok"}}])
    with patch("ollama.chat", return_value=expected) as mock_chat:
        service = OllamaService(model="llama3.2", parameters={"temperature": 0.1})
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "x"}]
        result = service.chat(messages=messages, tools=tools, stream=True)

    assert result is expected
    mock_chat.assert_called_once_with(
        model="llama3.2",
        messages=messages,
        options={"temperature": 0.1},
        stream=True,
        tools=tools,
    )
