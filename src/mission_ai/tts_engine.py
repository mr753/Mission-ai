import os
from mission_ai.models import VoiceOverScript
from mission_ai.providers.tts_base import TTSProvider

def generate_audio_for_script(script: VoiceOverScript, output_path: str, provider: TTSProvider) -> str:
    if not script or not script.script_text or not script.script_text.strip():
        raise ValueError("VoiceOverScript contains empty or invalid script text.")

    if not provider:
        raise ValueError("TTSProvider is required for audio generation.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    result_path = provider.synthesize(script.script_text, output_path)

    if not result_path or not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        raise RuntimeError(f"Audio generation failed: output file not found or empty at {output_path}")

    script.audio_path = result_path
    return result_path
