import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mission_ai.providers.tts_base import TTSProvider
from mission_ai.providers.pyttsx3 import PyTTSX3Provider

logger = logging.getLogger(__name__)


class EdgeTTSProvider(TTSProvider):
    AVAILABLE_VOICES = ["id-ID-GadisNeural", "id-ID-ArdiNeural"]

    def __init__(self, voice_id: str = "id-ID-GadisNeural", rate: str = "-5%", volume: str = "+0%"):
        self.voice_id = voice_id if voice_id in self.AVAILABLE_VOICES else "id-ID-GadisNeural"
        self.rate = rate
        self.volume = volume
        self.fallback_provider = PyTTSX3Provider()
        logger.info(
            "Selected TTS provider: EdgeTTSProvider with voice=%s, rate=%s, volume=%s",
            self.voice_id, self.rate, self.volume
        )

    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("FFmpeg executable not found for MP3 to WAV conversion.")
            return False
        cmd = [
            ffmpeg, "-y",
            "-i", mp3_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wav_path
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if completed.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                logger.info("Successfully converted MP3 to WAV at %s via FFmpeg", wav_path)
                return True
            else:
                logger.warning("FFmpeg conversion failed: exit code %d, stderr: %s", completed.returncode, completed.stderr)
                return False
        except Exception as e:
            logger.warning("FFmpeg conversion exception: %s", e)
            return False

    def synthesize(self, text: str, output_path: str) -> str:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text script.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_mp3 = None

        try:
            import edge_tts

            logger.info("Starting Edge TTS synthesis request for text length %d chars using voice %s", len(text), self.voice_id)

            is_wav_target = output_path.lower().endswith(".wav")
            if is_wav_target:
                fd, temp_mp3 = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                target_audio = temp_mp3
            else:
                target_audio = output_path

            async def _run_edge():
                communicate = edge_tts.Communicate(
                    text=text.strip(),
                    voice=self.voice_id,
                    rate=self.rate,
                    volume=self.volume
                )
                await asyncio.wait_for(communicate.save(target_audio), timeout=15.0)

            # Run async call with controlled timeout without hanging threads
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If an event loop is already running, run in a separate task or nested runner
                # To avoid nested run issues, we can run synchronous event loop or use asyncio.run in executor with proper timeout
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _run_edge())
                    future.result(timeout=20.0)
            else:
                asyncio.run(_run_edge())

            if not os.path.exists(target_audio) or os.path.getsize(target_audio) == 0:
                raise RuntimeError("Edge TTS produced empty audio output file.")

            logger.info("Edge TTS successfully generated MP3 audio output at %s", target_audio)

            if is_wav_target:
                converted = self._convert_mp3_to_wav(temp_mp3, output_path)
                if not converted:
                    if shutil.which("ffmpeg"):
                        raise RuntimeError("FFmpeg conversion from MP3 to WAV failed.")
                    else:
                        raise RuntimeError("FFmpeg not available for MP3 to WAV conversion.")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("EdgeTTSProvider successfully generated final WAV audio at %s", output_path)
                return output_path
            else:
                raise RuntimeError("Final audio output file not found or empty.")

        except Exception as e:
            reason = str(e) or type(e).__name__
            logger.warning(
                "Edge TTS failed (%s). Edge TTS did NOT succeed. Switching to fallback provider: PyTTSX3Provider",
                reason
            )

            fallback_res = self.fallback_provider.synthesize(text, output_path)
            logger.info(
                "Final audio successfully generated using fallback provider: PyTTSX3Provider at %s",
                output_path
            )
            return fallback_res
        finally:
            if temp_mp3 and os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
