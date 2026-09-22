import os
from PIL import Image
from mission_ai.models import ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}

def analyze_image(image_path: str, mission_context: MissionContext, provider: AIProvider) -> ImageAnalysis:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}")
        
    # Get metadata
    with Image.open(image_path) as img:
        width, height = img.size
        format = img.format
        
    # Call provider
    return provider.analyze_image(image_path, mission_context)
