from src.core.logging import setup_logging


def test_setup_logging_creates_dir(tmp_path):
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        setup_logging()
        assert (tmp_path / "logs").exists()
    finally:
        os.chdir(cwd)
