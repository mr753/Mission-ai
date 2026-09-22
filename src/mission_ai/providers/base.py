from typing import Protocol, List
from mission_ai.models import ImageAnalysis, MissionContext

class AIProvider(Protocol):
    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        ...

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        ...
