import os
import json
from PIL import Image
from mission_ai.models import ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider

class GeminiProvider(AIProvider):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
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

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        client = self._get_client()
        image = Image.open(image_path)
        prompt = (f"Analyze this image based on the mission: {mission_context.instructions}. "
                  "Describe only visible facts. Do not invent people, locations, organizations, numbers, events, or claims. "
                  "Return valid JSON with keys: summary, visible_subjects, visual_context, relevant_details.")
        
        response = client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image]
        )
        return ImageAnalysis(**json.loads(response.text))

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        client = self._get_client()
        prompt = (f"Generate a {platform} caption grounded in this analysis: {image_analysis.summary}. "
                  f"Mission context: {mission_context.main_message}. Return text only.")
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()

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
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()
