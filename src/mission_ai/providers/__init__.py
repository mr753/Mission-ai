from mission_ai.providers.base import AIProvider
from mission_ai.providers.gemini import GeminiProvider
from mission_ai.providers.ollama import OllamaProvider

__all__ = [
    "AIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "create_provider",
]


def create_provider(config) -> AIProvider:
    """Build the AI provider named by config.ai_provider using config settings.

    Raises ValueError for unknown provider names. Actual API calls are
    deferred until first use (Gemini) or performed lazily against the local
    Ollama endpoint, so constructing a provider never requires network access.
    """
    name = (getattr(config, "ai_provider", "") or "").strip().lower()

    if name == "gemini":
        from mission_ai.providers.gemini import GeminiProvider
        return GeminiProvider(model_name=getattr(config, "gemini_model", "gemini-1.5-flash"))

    if name == "ollama":
        from mission_ai.providers.ollama import OllamaProvider
        return OllamaProvider(model_name=getattr(config, "ollama_model", "llama3"))

    raise ValueError(
        f"Unknown AI provider: {name!r}. Supported providers: gemini, ollama."
    )
