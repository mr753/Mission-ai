import os
from pathlib import Path
from typing import Optional

def select_music(mood: str, music_dir: Path) -> Optional[Path]:
    if not music_dir.exists():
        return None
    
    # Simple logic: find mp3/wav in subdir matching mood
    mood_dir = music_dir / mood
    if mood_dir.exists():
        files = [f for f in mood_dir.iterdir() if f.suffix in {'.mp3', '.wav', '.m4a'}]
        if files:
            return files[0]
            
    return None
