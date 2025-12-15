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
