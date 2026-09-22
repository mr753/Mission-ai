import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    music_directory: Path = Path(os.getenv("MUSIC_DIR", "./assets/music"))
    default_duration: int = int(os.getenv("VIDEO_DURATION", 15))
    video_width: int = int(os.getenv("VIDEO_WIDTH", 1080))
    video_height: int = int(os.getenv("VIDEO_HEIGHT", 1920))
    fps: int = int(os.getenv("VIDEO_FPS", 30))
    ai_provider: str = os.getenv("AI_PROVIDER", "gemini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    output_directory: Path = Path(os.getenv("OUTPUT_DIR", "./output"))

def load_config() -> AppConfig:
    return AppConfig()
