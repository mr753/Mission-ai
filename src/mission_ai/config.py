import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_key: Optional[str] = os.getenv("SUPABASE_KEY")
    enable_supabase_sink: bool = os.getenv("ENABLE_SUPABASE_SINK", "false").lower() == "true"


def load_config() -> AppConfig:
    return AppConfig()
