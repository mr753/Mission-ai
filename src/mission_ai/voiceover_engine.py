from mission_ai.models import ImageAnalysis, MissionContext, VoiceOverScript
from mission_ai.providers.base import AIProvider

def generate_voiceover_script(image_analysis: ImageAnalysis, mission_context: MissionContext, 
                                provider: AIProvider, image_id: str = "") -> VoiceOverScript:
    if not hasattr(provider, "generate_voiceover_script") or not callable(getattr(provider, "generate_voiceover_script")):
        raise AttributeError(f"Provider {type(provider).__name__} does not implement generate_voiceover_script")
    
    script_text = provider.generate_voiceover_script(image_analysis, mission_context)
    return VoiceOverScript(
        image_id=image_id,
        script_text=script_text,
        language="id"
    )
