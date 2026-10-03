import os
import json
import time
import random
from PIL import Image
from mission_ai.models import ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider

class GeminiProvider(AIProvider):
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("Gemini provider requires google-genai, which is not installed.")
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _is_transient_error(self, e: Exception) -> bool:
        err_str = str(e).lower()
        non_transient = [
            "401", "403", "unauthenticated", "permission denied",
            "invalid api key", "invalid_argument", "not_found", "400"
        ]
        if any(nt in err_str for nt in non_transient):
            return False

        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if code in {401, 403, 400}:
            return False
        if code in {503, 429, 500, 502, 504}:
            return True

        transient_keywords = [
            "503", "unavailable", "resource_exhausted", "rate limit",
            "429", "timed out", "timeout", "deadline exceeded",
            "internal", "server error", "high demand", "temporarily unavailable"
        ]
        return any(kw in err_str for kw in transient_keywords)

    def _call_with_retry(self, func, *args, **kwargs):
        max_attempts = 3
        base_delay = 1.0
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1 or not self._is_transient_error(e):
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        client = self._get_client()
        image = Image.open(image_path)
        prompt = (f"Analyze this image based on the mission: {mission_context.instructions}. "
                  "Describe only visible facts. Do not invent people, locations, organizations, numbers, events, or claims. "
                  "Return valid JSON with keys: summary, visible_subjects, visual_context, relevant_details.")
        
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=[prompt, image]
        )
        text = response.text if response and hasattr(response, "text") else ""
        if not text or not text.strip():
            raise ValueError("Gemini response is empty or contains no text.")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        if not cleaned:
            raise ValueError("Gemini response is empty after removing markdown fences.")

        data = json.loads(cleaned)
        return ImageAnalysis(**data)

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        client = self._get_client()
        prompt = (f"Generate a {platform} caption grounded in this analysis: {image_analysis.summary}. "
                  f"Mission context: {mission_context.main_message}. Return text only.")
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        return (response.text or "").strip()

    def generate_voiceover_script(self, image_analysis: ImageAnalysis, mission_context: MissionContext) -> str:
        client = self._get_client()
        prompt = (
            f"Buatkan script voice-over singkat dalam Bahasa Indonesia untuk konten sosial media berdasarkan informasi berikut:\n"
            f"Pesan Utama Mission: {mission_context.main_message}\n"
            f"Instruksi Mission: {mission_context.instructions}\n"
            f"Analisis Visual Image: {image_analysis.summary} (Detail: {', '.join(image_analysis.relevant_details)})\n\n"
            "Prinsip Penting:\n"
            "1. Gunakan Bahasa Indonesia yang natural saat dibacakan oleh TTS.\n"
            "2. Relevan dengan pesan utama mission dan detail visual image tersebut.\n"
            "3. Jangan mengarang fakta visual yang tidak ada di analisis.\n"
            "4. Jangan gunakan markdown, hashtag, caption format, atau instruksi teknis.\n"
            "5. Singkat, padat, dan cocok untuk voice-over.\n"
            "Keluarkan teks script saja."
        )
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        return (response.text or "").strip()
