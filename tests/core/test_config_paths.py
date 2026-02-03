from pathlib import Path
from unittest.mock import patch, MagicMock
from src.core.config import get_config_path, Settings


def test_get_config_path_explicit_profile(tmp_path, monkeypatch):
    """Test explicit profile takes highest priority."""
    # Create a test config file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    test_config = config_dir / "heretic.yaml"
    test_config.write_text("model: test")

    monkeypatch.chdir(tmp_path)

    # Mock settings
    mock_settings = MagicMock(spec=Settings)
    mock_settings.template_config = Path("config/template.yaml")
    mock_settings.environment = "development"

    result = get_config_path(mock_settings, profile="heretic")
    assert result == Path("config/heretic.yaml")


def test_get_config_path_explicit_profile_not_found(tmp_path, monkeypatch):
    """Test fallback when explicit profile doesn't exist."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    template = config_dir / "template.yaml"
    template.write_text("model: test")

    monkeypatch.chdir(tmp_path)

    mock_settings = MagicMock(spec=Settings)
    mock_settings.template_config = Path("config/template.yaml")
    mock_settings.environment = "development"

    result = get_config_path(mock_settings, profile="nonexistent")
    assert result == Path("config/template.yaml")


def test_get_config_path_custom_template_config(tmp_path, monkeypatch):
    """Test TEMPLATE_CONFIG environment variable takes priority."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    custom = config_dir / "custom.yaml"
    custom.write_text("model: test")

    monkeypatch.chdir(tmp_path)

    mock_settings = MagicMock(spec=Settings)
    mock_settings.template_config = Path("config/custom.yaml")
    mock_settings.environment = "development"

    result = get_config_path(mock_settings)
    assert result == Path("config/custom.yaml")


def test_get_config_path_environment_based(tmp_path, monkeypatch):
    """Test environment-based config selection."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    template = config_dir / "template.yaml"
    template.write_text("model: test")

    monkeypatch.chdir(tmp_path)

    mock_settings = MagicMock(spec=Settings)
    mock_settings.template_config = Path("config/template.yaml")
    mock_settings.environment = "development"

    result = get_config_path(mock_settings)
    assert result == Path("config/template.yaml")


def test_get_config_path_default_fallback(tmp_path, monkeypatch):
    """Test final fallback to template.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    template = config_dir / "template.yaml"
    template.write_text("model: test")

    monkeypatch.chdir(tmp_path)

    mock_settings = MagicMock(spec=Settings)
    mock_settings.template_config = Path("config/template.yaml")
    mock_settings.environment = "unknown_env"

    result = get_config_path(mock_settings)
    assert result == Path("config/template.yaml")


def test_get_config_path_no_settings_provided(tmp_path, monkeypatch):
    """Test that get_config_path loads settings when none provided."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    template = config_dir / "template.yaml"
    template.write_text("model: test")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("src.core.config.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_settings.template_config = Path("config/template.yaml")
        mock_settings.environment = "development"
        mock_get_settings.return_value = mock_settings

        result = get_config_path()
        mock_get_settings.assert_called_once()
        assert result == Path("config/template.yaml")


def test_settings_environment_field_default():
    """Test Settings has environment field with default value."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
        settings = Settings()
        assert settings.environment == "development"


def test_settings_environment_field_custom():
    """Test Settings environment field can be customized via env var."""
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "ENVIRONMENT": "production"},
        clear=True,
    ):
        settings = Settings()
        assert settings.environment == "production"
