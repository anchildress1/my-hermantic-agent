import pytest
import yaml
from src.core.config import load_config


def test_load_template_success(tmp_path):
    p = tmp_path / "template.yaml"
    data = {
        "model": "llama3.2",
        "system": "You are helpful",
        "parameters": {"num_ctx": 1024},
    }
    p.write_text(yaml.safe_dump(data))

    loaded = load_config(p)
    assert loaded.model == "llama3.2"
    assert loaded.system == "You are helpful"


def test_load_template_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_template_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("::: not yaml :::")
    with pytest.raises(Exception):
        load_config(p)
