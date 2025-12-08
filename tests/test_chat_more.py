import json
import pytest
from unittest.mock import MagicMock
from src.agent import chat


def test_check_ollama_partial_match(monkeypatch):
    # available model has tag, should match partial
    monkeypatch.setattr("src.agent.chat.ollama", MagicMock())
    oll = __import__("src.agent.chat", fromlist=["ollama"]).ollama
    oll.list.return_value = {"models": [{"model": "llama3.2:abc"}, {"model": "other"}]}

    assert chat.check_ollama_connection("llama3.2") is True


def test_check_ollama_model_not_found(monkeypatch, capsys):
    monkeypatch.setattr("src.agent.chat.ollama", MagicMock())
    oll = __import__("src.agent.chat", fromlist=["ollama"]).ollama
    oll.list.return_value = {"models": [{"model": "other"}]}

    res = chat.check_ollama_connection("llama3.2")
    out = capsys.readouterr().out
    assert res is False
    assert "Model 'llama3.2' not found" in out or "Available models" in out


def test_print_context_truncation(capsys):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 500},
    ]
    chat.print_context(msgs, show_full=False)
    out = capsys.readouterr().out
    assert "..." in out
    assert "Total messages" in out


def test_trim_context_no_system_message():
    msgs = []
    for _ in range(20):
        msgs.append({"role": "user", "content": "x" * 200})

    trimmed, was_trimmed = chat.trim_context(msgs, max_tokens=500, keep_recent=5)
    assert was_trimmed
    # first message should be the summary with role system
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) <= 1 + 5 + 1


def test_archive_memory_snapshot_missing(tmp_path):
    missing = tmp_path / "nope.json"
    res = chat.archive_memory_snapshot(str(missing), prefix="t")
    assert res is None


def test_load_memory_no_file(tmp_path, capsys):
    path = tmp_path / "missing.json"
    res = chat.load_memory(str(path))
    out = capsys.readouterr().out
    assert res == []
    # load_memory logs info when file missing; it does not print to stdout
    assert out == ""


def test_load_memory_corrupted_no_backup(tmp_path, capsys):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text("{ not json")

    res = chat.load_memory(str(mem_file))
    out = capsys.readouterr().out
    assert res == []
    assert "Memory file corrupted and no backup available" in out


def test_load_memory_open_exception(tmp_path):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps({"messages": []}))

    # Simulate unexpected exception when opening
    def raise_exc(*a, **k):
        raise RuntimeError("boom")

    import builtins

    orig_open = builtins.open
    try:
        builtins.open = raise_exc
        res = chat.load_memory(str(mem_file))
        assert res == []
    finally:
        builtins.open = orig_open


def test_print_help_without_memory_store(capsys):
    chat.print_help(None)
    out = capsys.readouterr().out
    assert "Memory Commands" not in out


def test_print_context_show_full(capsys):
    msgs = [{"role": "user", "content": "x" * 200}]
    chat.print_context(msgs, show_full=True)
    out = capsys.readouterr().out
    assert "..." not in out
