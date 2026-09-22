from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class Mission:
    id: str
    content: str
    key_points: List[str]
    hashtags: List[str]
    max_hashtags: int
    drive_url: Optional[str]

class MissionParser:
    @staticmethod
    def parse(file_path: Path) -> Mission:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Simple parser logic, can be improved to regex
        mission = Mission(id=file_path.stem, content="", key_points=[], hashtags=[], max_hashtags=5, drive_url=None)
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith("MISSION:"):
                current_section = "MISSION"
                mission.content = line.replace("MISSION:", "").strip()
            elif line.startswith("KEY POINTS:"):
                current_section = "KEY"
            elif line.startswith("HASHTAG:"):
                current_section = "TAG"
            elif line.startswith("MAX_HASHTAGS:"):
                mission.max_hashtags = int(line.replace("MAX_HASHTAGS:", "").strip())
            elif line.startswith("DRIVE:"):
                mission.drive_url = line.replace("DRIVE:", "").strip()
            elif current_section == "KEY" and line.startswith("-"):
                mission.key_points.append(line.replace("-", "").strip())
            elif current_section == "TAG" and line.startswith("#"):
                mission.hashtags.append(line.strip())
                
        return mission
