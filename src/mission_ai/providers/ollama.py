import urllib.request
import urllib.error
import json
import os
import base64
from typing import Dict, Any
from mission_ai.models import ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider

class OllamaProvider(AIProvider):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = (
            f"Analyze this image based on the mission: {mission_context.instructions}. "
            "Describe only visible facts. Do not invent people, locations, organizations, numbers, events, or claims. "
            "Separate observation from interpretation. "
            "Return valid JSON with keys: summary, visible_subjects, visual_context, relevant_details."
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [encoded_image],
            "format": "json",
            "stream": False
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                analysis_data = json.loads(data['response'])
                return ImageAnalysis(**analysis_data)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Ollama provider failed: {e}")

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        prompt = (
            f"Generate a {platform} caption grounded in this analysis: {image_analysis.summary}. "
            f"Mission context: {mission_context.main_message}. "
            "Return text only."
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data['response'].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama provider failed: {e}")
