import os
from mission_ai.providers.base import AIProvider
from mission_ai.providers.gemini import GeminiProvider
from mission_ai.providers.ollama import OllamaProvider

__all__ = ["AIProvider", "GeminiProvider", "OllamaProvider", "create_provider"]

class FallbackProvider:
    """Use Gemini when Ollama is unavailable."""
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    def _call(self, name, *args):
        provider = self.fallback if self._use_fallback else self.primary
        try:
            return getattr(provider, name)(*args)
        except RuntimeError as e:
            if not self._use_fallback and ("Connection refused" in str(e) or "urlopen error" in str(e)):
                self._use_fallback = True
                return getattr(self.fallback, name)(*args)
            raise

    def analyze_image(self, image_path, mission_context):
        return self._call("analyze_image", image_path, mission_context)

    def generate_caption(self, image_analysis, mission_context, platform):
        return self._call("generate_caption", image_analysis, mission_context, platform)

    def generate_voiceover_script(self, image_analysis, mission_context):
        return self._call("generate_voiceover_script", image_analysis, mission_context)

    def select_best_image(self, analyses, mission_context):
        if not hasattr(self.fallback, "select_best_image"):
            raise RuntimeError("No AI provider supports image selection")
        return self.fallback.select_best_image(analyses, mission_context)

def create_provider(config) -> AIProvider:
    name = (getattr(config, "ai_provider", "") or "").strip().lower()
    if name == "gemini":
        return GeminiProvider(model_name=getattr(config, "gemini_model", "gemini-3.8-flash"))
    if name == "ollama":
        primary = OllamaProvider(model_name=getattr(config, "ollama_model", "llama3"))
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return FallbackProvider(primary, GeminiProvider(model_name=getattr(config, "gemini_model", "gemini-3.8-flash")))
        return primary
    raise ValueError(f"Unknown AI provider: {name!r}. Supported providers: gemini, ollama.")
