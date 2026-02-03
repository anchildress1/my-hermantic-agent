import yaml
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Global application settings and environment variables."""

    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")
    memory_db_url: Optional[str] = Field(
        None, description="TimescaleDB connection string"
    )
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    template_config: Path = Path("config/template.yaml")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings()


class ModelParameters(BaseModel):
    """LLM generation parameters."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 2048
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    num_ctx: int = 4096


class AgentConfig(BaseModel):
    """Main agent configuration."""

    model: str
    system: str
    parameters: ModelParameters = Field(default_factory=ModelParameters)


def load_config(config_path: Path) -> AgentConfig:
    """Load configuration from YAML file and validate with Pydantic."""
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        config = AgentConfig(**raw_config)
        logger.info(f"Loaded and validated config from {config_path}")
        return config

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config: {e}")
        raise
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        raise
