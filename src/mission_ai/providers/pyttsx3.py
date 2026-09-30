import os
import pyttsx3
from mission_ai.providers.tts_base import TTSProvider

class PyTTSX3Provider(TTSProvider):
    def __init__(self, voice_id: str = None, rate: int = 150):
        self.voice_id = voice_id
        self.rate = rate

    def synthesize(self, text: str, output_path: str) -> str:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text script.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            if self.voice_id:
                engine.setProperty('voice', self.voice_id)

            engine.save_to_file(text, output_path)
            engine.runAndWait()

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"TTS synthesis failed to create output audio file at {output_path}")

            return output_path
        except Exception as e:
            raise RuntimeError(f"PyTTSX3Provider synthesis failed: {e}")
