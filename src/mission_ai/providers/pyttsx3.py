import os
import shutil
import subprocess

import pyttsx3

from mission_ai.providers.tts_base import TTSProvider


class PyTTSX3Provider(TTSProvider):
    def __init__(self, voice_id: str = None, rate: int = 150):
        self.voice_id = voice_id
        self.rate = rate

    def _synthesize_with_espeak(self, text: str, output_path: str) -> str:
        """Synthesize speech with a slower Indonesian eSpeak voice and natural pauses."""
        espeak = shutil.which("espeak")
        if not espeak:
            raise RuntimeError("eSpeak executable not found. Install it with: pkg install espeak")

        voice = self.voice_id
        if not voice:
            try:
                voices = subprocess.run(
                    [espeak, "--voices"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                if any(line.split()[-1] == "id" for line in voices.splitlines() if line.strip()):
                    voice = "id"
            except Exception:
                voice = None

        # Scale speaking rate to the script length so narration stays near 30-35 seconds.
        # eSpeak uses words-per-minute; target about 32 seconds and keep the rate natural.
        word_count = len(text.split())
        rate = max(120, min(180, round(word_count * 60 / 32)))
        cmd = [
            espeak,
            "-w", output_path,
            "-s", str(rate),
            "-p", "45",
            "-a", "160",
            "-g", "8",
            "-m",
        ]
        if voice and ":" not in voice and os.path.sep not in voice:
            cmd[1:1] = ["-v", voice]
        cmd.append(text)
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown eSpeak error").strip()
            raise RuntimeError(f"eSpeak synthesis failed: {detail}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"eSpeak did not create output audio file at {output_path}")
        return output_path

    def synthesize(self, text: str, output_path: str) -> str:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text script.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Termux commonly has the eSpeak binary available even when the
        # pyttsx3 Python backend cannot load its native eSpeak library.
        if shutil.which("espeak"):
            try:
                return self._synthesize_with_espeak(text, output_path)
            except Exception as espeak_error:
                espeak_failure = espeak_error
            else:
                espeak_failure = None
        else:
            espeak_failure = None

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)

            engine.save_to_file(text, output_path)
            engine.runAndWait()

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"TTS synthesis failed to create output audio file at {output_path}")

            return output_path
        except Exception as e:
            if espeak_failure is not None:
                raise RuntimeError(f"eSpeak fallback failed: {espeak_failure}; pyttsx3 failed: {e}")
            raise RuntimeError(f"PyTTSX3Provider synthesis failed: {e}")
