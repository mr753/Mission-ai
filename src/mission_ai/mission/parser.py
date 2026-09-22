import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from mission_ai.models import MissionContext

class MissionParser:
    @staticmethod
    def parse_json_file(file_path: str) -> MissionContext:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Mission file not found: {file_path}")
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Malformed JSON: {file_path}")
            
        return MissionParser.validate_and_create(data)

    @staticmethod
    def validate_and_create(data: Dict[str, Any]) -> MissionContext:
        # Required fields validation
        required = ["mission_id", "main_message", "max_hashtags"]
        for field in required:
            if field not in data or not str(data[field]).strip():
                raise ValueError(f"Missing or empty required field: {field}")
        
        # Strict max_hashtags validation: must be int, not bool, not float
        max_h = data["max_hashtags"]
        if not isinstance(max_h, int) or isinstance(max_h, bool):
            raise ValueError("max_hashtags must be an integer")
        if max_h < 0:
            raise ValueError("max_hashtags must be a non-negative integer")
        
        # Deadline validation
        if "deadline" in data:
            try:
                # Normalizing 'Z' to '+00:00' for fromisoformat compatibility in some Python versions
                dt_str = data["deadline"].replace("Z", "+00:00")
                datetime.fromisoformat(dt_str)
            except (ValueError, TypeError):
                raise ValueError("deadline must be in ISO 8601 format")

        # Type validation
        if not isinstance(data.get("key_points", []), list):
            raise ValueError("key_points must be a list")
        if not isinstance(data.get("platforms", []), list):
            raise ValueError("platforms must be a list")
        if not isinstance(data.get("required_hashtags", []), list):
            raise ValueError("required_hashtags must be a list")
        
        return MissionContext(
            mission_id=str(data["mission_id"]),
            instructions=str(data.get("instructions", "")),
            main_message=str(data["main_message"]),
            key_points=data.get("key_points", []),
            platforms=data.get("platforms", []),
            max_hashtags=max_h,
            required_hashtags=data.get("required_hashtags", [])
        )
