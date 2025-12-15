import pytest
import json
from pathlib import Path
from src.services.memory.file_storage import (
    save_chat_history,
    load_chat_history,
    archive_chat_history,
)


def test_save_and_load_memory_roundtrip(tmp_path):
    mem_file = str(tmp_path / "memory.json")
    messages = [
        {"role": "system", "content": "init"},
        {"role": "user", "content": "hello"},
    ]

    # Save first time
    save_chat_history(messages, file_path=mem_file)
    assert Path(mem_file).exists()

    # Save second time to cause backup
    save_chat_history(messages, file_path=mem_file)
    backup = Path(mem_file).with_suffix(".json.bak")
    assert backup.exists()

    loaded = load_chat_history(mem_file)
    assert isinstance(loaded, list)
    assert loaded[0]["role"] == "system"


def test_load_memory_corrupted_uses_backup(tmp_path):
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

    msgs = load_chat_history(str(mem_file))
    assert msgs == [{"role": "user", "content": "from backup"}]


def test_archive_memory_snapshot(tmp_path):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))

    archived = archive_chat_history(str(mem_file), prefix="testclear")
    assert archived is not None
    assert archived.exists()


def test_save_memory_failure(monkeypatch, tmp_path):
    mem_file = str(tmp_path / "memory.json")
    messages = [{"role": "user", "content": "hi"}]

    # Simulate tempfile failing
    def raise_tmp(*a, **k):
        raise RuntimeError("no tmp")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", raise_tmp)

    with pytest.raises(RuntimeError):
        save_chat_history(messages, file_path=mem_file)


def test_archive_memory_snapshot_failure(monkeypatch, tmp_path):
    mem_file = tmp_path / "memory.json"
    mem_file.write_text("{}")

    # Make copy2 raise
    def raise_copy(*a, **k):
        raise RuntimeError("copy fail")

    monkeypatch.setattr("shutil.copy2", raise_copy)
    res = archive_chat_history(str(mem_file), prefix="x")
    assert res is None
