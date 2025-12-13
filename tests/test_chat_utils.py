import json
from pathlib import Path
import yaml
import pytest

from src.agent import chat


def test_load_template_success(tmp_path):
    p = tmp_path / "template.yaml"
    data = {
        "model": "llama3.2",
        "system": "You are helpful",
        "parameters": {"num_ctx": 1024},
    }
    p.write_text(yaml.safe_dump(data))

    loaded = chat.load_template(p)
    assert loaded["model"] == "llama3.2"
    assert loaded["system"] == "You are helpful"


def test_load_template_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        chat.load_template(tmp_path / "nope.yaml")


def test_save_and_load_memory_roundtrip(tmp_path, capsys):
    mem_file = str(tmp_path / "memory.json")
    messages = [
        {"role": "system", "content": "init"},
        {"role": "user", "content": "hello"},
    ]

    # Save first time
    chat.save_memory(messages, memory_file=mem_file)
    assert Path(mem_file).exists()

    # Save second time to cause backup
    chat.save_memory(messages, memory_file=mem_file)
    backup = Path(mem_file).with_suffix(".json.bak")
    assert backup.exists()

    loaded = chat.load_memory(mem_file)
    assert isinstance(loaded, list)
    assert loaded[0]["role"] == "system"


def test_load_memory_corrupted_uses_backup(tmp_path, capsys):
    mem_file = tmp_path / "memory.json"
    backup = tmp_path / "memory.json.bak"

    mem_file.write_text("{ this is not json")
    backup.write_text(
        json.dumps(
            {
                "timestamp": "now",
                "messages": [{"role": "user", "content": "from backup"}],
            }
        )
    )

    msgs = chat.load_memory(str(mem_file))
    assert msgs == [{"role": "user", "content": "from backup"}]


def test_archive_memory_snapshot(tmp_path):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))

    archived = chat.archive_memory_snapshot(str(mem_file), prefix="testclear")
    assert archived is not None
    assert archived.exists()


def test_load_template_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("::: not yaml :::")
    with pytest.raises(Exception):
        chat.load_template(p)


def test_save_memory_failure(monkeypatch, tmp_path):
    mem_file = str(tmp_path / "memory.json")
    messages = [{"role": "user", "content": "hi"}]

    # Simulate tempfile failing
    def raise_tmp(*a, **k):
        raise RuntimeError("no tmp")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", raise_tmp)

    with pytest.raises(RuntimeError):
        chat.save_memory(messages, memory_file=mem_file)


def test_archive_memory_snapshot_failure(monkeypatch, tmp_path):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text("{}")

    # Make copy2 raise
    def raise_copy(*a, **k):
        raise RuntimeError("copy fail")

    monkeypatch.setattr("shutil.copy2", raise_copy)
    res = chat.archive_memory_snapshot(str(mem_file), prefix="x")
    assert res is None


def test_print_help_with_memory_store(capsys):
    class Dummy:
        pass

    chat.print_help(Dummy())
    out = capsys.readouterr().out
    assert "Memory Commands" in out


def test_setup_logging_creates_dir(tmp_path):
    # Change working dir to tmp to avoid touching repo logs
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        chat.setup_logging()
        assert (tmp_path / "logs").exists()
    finally:
        os.chdir(cwd)


def test_token_estimations_and_trim():
    msgs = []
    # system message always present
    msgs.append({"role": "system", "content": "system prompt"})
    # Add 30 messages with content to exceed token thresholds
    for _ in range(30):
        msgs.append({"role": "user", "content": "x" * 500})

    total = chat.count_message_tokens(msgs)
    assert total > 0

    trimmed, was_trimmed = chat.trim_context(msgs, max_tokens=500, keep_recent=5)
    assert was_trimmed
    assert len(trimmed) <= 1 + 5 + 1  # system + summary + recent
