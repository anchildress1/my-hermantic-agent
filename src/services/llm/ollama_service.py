import ollama
import logging
from typing import List, Dict, Any, Generator

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self, model: str, parameters: Dict[str, Any] = None):
        self.model = model
        self.parameters = parameters or {}

    def check_connection(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            model_list = ollama.list()
            available_models = [m["model"] for m in model_list.get("models", [])]

            # Check for exact match or partial match (model might have :tag)
            model_found = any(
                self.model in m or m in self.model for m in available_models
            )

            if not model_found:
                logger.error(
                    f"Model '{self.model}' not found. Available: {available_models}"
                )
                print(f"❌ Model '{self.model}' not found")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   Run: ollama pull {self.model}")
                return False

            logger.info(f"Ollama connection verified, model '{self.model}' available")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            print(f"❌ Failed to connect to Ollama: {e}")
            print("   Make sure Ollama is running (ollama serve)")
            return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any] = None,
        stream: bool = True,
    ) -> Generator[Dict, None, None] | Dict:
        """Send chat request to Ollama."""
        return ollama.chat(
            model=self.model,
            messages=messages,
            options=self.parameters,
            stream=stream,
            tools=tools,
        )
