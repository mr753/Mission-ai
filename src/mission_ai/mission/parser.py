import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import re
import hashlib
from mission_ai.models import MissionContext
from mission_ai.runner import is_google_drive_url

def extract_google_drive_url(text: str) -> Optional[str]:
    if not text:
        return None
    urls = re.findall(r"https?://[^\s<>\"']+", text)
    for url in urls:
        cleaned = url.rstrip(".,)]:;\"'")
        if is_google_drive_url(cleaned):
            return cleaned
    return None

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
    def parse_mission_text(text: str) -> MissionContext:
        if not text or not text.strip():
            raise ValueError("Mission text is required.")

        urls = re.findall(r"https?://[^\s<>\"']+", text)
        cleaned_urls = [u.rstrip(".,)]:;\"'") for u in urls]
        drive_mentions = [u for u in cleaned_urls if "drive.google.com" in u]

        if not drive_mentions:
            raise ValueError("No supported Google Drive link was found in the mission text.")

        drive_url = drive_mentions[0]
        if not is_google_drive_url(drive_url):
            raise ValueError("Google Drive link is invalid or unsupported.")

        try:
            from mission_ai.sources.google_drive import GoogleDriveFolderResolver
            resolver = GoogleDriveFolderResolver(download_dir=Path("./input"))
            resolver._extract_folder_id(drive_url)
        except Exception:
            raise ValueError("Google Drive link is invalid or unsupported.")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        mission_id = "MISSION-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:8].upper()

        main_message = lines[0] if lines else "Mission Posting"
        for i, line in enumerate(lines):
            if "pesan utama" in line.lower() or "main message" in line.lower():
                if i + 1 < len(lines):
                    main_message = lines[i + 1]
                    break

        key_points = []
        capture_points = False
        for line in lines:
            if "poin penting" in line.lower() or "key points" in line.lower():
                capture_points = True
                continue
            if capture_points:
                if any(kw in line.lower() for kw in ["aturan", "source", "tanggal", "batas"]) and not line.startswith("-") and not line.startswith("*"):
                    capture_points = False
                if capture_points and (line.startswith("-") or line.startswith("*") or (line[0].isdigit() and "." in line[:3])):
                    key_points.append(line.lstrip("-* 0123456789.").strip())

        if not key_points:
            for line in lines:
                if line.startswith("-") or line.startswith("*"):
                    key_points.append(line.lstrip("-* ").strip())

        return MissionContext(
            mission_id=mission_id,
            instructions=text,
            main_message=main_message,
            key_points=key_points,
            platforms=["instagram"],
            max_hashtags=5,
            required_hashtags=[],
            image_source_url=drive_url,
        )

    @staticmethod
    def validate_and_create(data: Dict[str, Any]) -> MissionContext:
        required = ["mission_id", "main_message", "max_hashtags"]
        for field in required:
            if field not in data or not str(data[field]).strip():
                raise ValueError(f"Missing or empty required field: {field}")

        max_h = data["max_hashtags"]
        if not isinstance(max_h, int) or isinstance(max_h, bool):
            raise ValueError("max_hashtags must be an integer")
        if max_h < 0:
            raise ValueError("max_hashtags must be a non-negative integer")

        if "deadline" in data:
            try:
                dt_str = data["deadline"].replace("Z", "+00:00")
                datetime.fromisoformat(dt_str)
            except (ValueError, TypeError):
                raise ValueError("deadline must be in ISO 8601 format")

        if not isinstance(data.get("key_points", []), list):
            raise ValueError("key_points must be a list")
        if not isinstance(data.get("platforms", []), list):
            raise ValueError("platforms must be a list")
        if not isinstance(data.get("required_hashtags", []), list):
            raise ValueError("required_hashtags must be a list")

        image_source_url = data.get("image_source_url") or None

        return MissionContext(
            mission_id=str(data["mission_id"]),
            instructions=str(data.get("instructions", "")),
            main_message=str(data["main_message"]),
            key_points=data.get("key_points", []),
            platforms=data.get("platforms", []),
            max_hashtags=max_h,
            required_hashtags=data.get("required_hashtags", []),
            image_source_url=str(image_source_url) if image_source_url else None,
        )
