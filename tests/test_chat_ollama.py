from src.agent import chat


def test_check_ollama_connection_model_found(monkeypatch):
    # Simulate ollama.list returning available models
    def fake_list():
        return {"models": [{"model": "llama3.2"}, {"model": "gpt-like:latest"}]}

    monkeypatch.setattr(
        chat, "ollama", type("O", (), {"list": staticmethod(fake_list)})
    )

    assert chat.check_ollama_connection("llama3.2") is True
    assert chat.check_ollama_connection("gpt-like") is True


def test_check_ollama_connection_service_down(monkeypatch):
    def raise_exc():
        raise RuntimeError("not running")

    monkeypatch.setattr(
        chat, "ollama", type("O", (), {"list": staticmethod(raise_exc)})
    )

    assert chat.check_ollama_connection("llama3.2") is False
