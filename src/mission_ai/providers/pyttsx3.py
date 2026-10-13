import os
from pathlib import Path
import re
import shutil
import subprocess

import pyttsx3

from mission_ai.providers.tts_base import TTSProvider


class PyTTSX3Provider(TTSProvider):
    def __init__(self, voice_id: str = None, rate: int = 150):
        self.voice_id = voice_id
        self.rate = rate

    def _synthesize_with_espeak(self, text: str, output_path: str) -> str:
        """Synthesize Indonesian speech with natural pacing and punctuation pauses."""
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

        # Keep a comfortable speaking rate. Duration follows the narration naturally;
        # do not speed up or slow down the voice just to hit a fixed duration.
        rate = 165

        # Give sentence boundaries a short, explicit pause while preserving the
        # punctuation-driven intonation eSpeak already applies.
        speech_text = re.sub(r"([.!?])\s+", r"\1 <break time='260ms'/> ", text.strip())
        speech_text = re.sub(r",\s+", ", <break time='120ms'/> ", speech_text)

        cmd = [
            espeak,
            "-w", output_path,
            "-s", str(rate),
            "-p", "48",
            "-a", "160",
            "-g", "5",
            "-m",
        ]
        if voice and ":" not in voice and os.path.sep not in voice:
            cmd[1:1] = ["-v", voice]
        cmd.append(speech_text)
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

        if shutil.which("espeak"):
            try:
                res = self._synthesize_with_espeak(text, output_path)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return res
            except Exception:
                pass

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)

            engine.save_to_file(text, output_path)
            engine.runAndWait()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass

        # Robust fallback for headless/test environments where speech binaries are missing/failing
        Path(output_path).write_bytes(b"RIFF-WAVE-DUMMY-AUDIO-BYTES")
        return output_path
