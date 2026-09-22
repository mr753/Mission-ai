import pytest
import json
import os
from pathlib import Path
from mission_ai.mission.parser import MissionParser

def test_valid_mission(tmp_path):
    data = {
        "mission_id": "M1",
        "main_message": "Msg",
        "max_hashtags": 5,
        "key_points": ["p"],
        "platforms": ["ig"],
        "deadline": "2026-09-22T23:59:59+09:00"
    }
    file = tmp_path / "m.json"
    with open(file, "w") as f:
        json.dump(data, f)
        
    ctx = MissionParser.parse_json_file(str(file))
    assert ctx.mission_id == "M1"
    assert ctx.max_hashtags == 5

def test_max_hashtags_types(tmp_path):
    invalid_values = ["5", 5.5, True]
    for val in invalid_values:
        data = {"mission_id": "M1", "main_message": "Msg", "max_hashtags": val}
        file = tmp_path / f"m_{val}.json"
        with open(file, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="max_hashtags must be an integer"):
            MissionParser.parse_json_file(str(file))

def test_negative_max_hashtags(tmp_path):
    data = {"mission_id": "M1", "main_message": "Msg", "max_hashtags": -1}
    file = tmp_path / "m.json"
    with open(file, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="max_hashtags must be a non-negative integer"):
        MissionParser.parse_json_file(str(file))

def test_deadline_validation(tmp_path):
    # Valid
    for d in ["2026-09-22T23:59:59+09:00", "2026-09-22T14:59:59Z"]:
        data = {"mission_id": "M1", "main_message": "Msg", "max_hashtags": 5, "deadline": d}
        file = tmp_path / f"m_{d.replace(':', '')}.json"
        with open(file, "w") as f:
            json.dump(data, f)
        MissionParser.parse_json_file(str(file))
    
    # Invalid
    for d in ["tomorrow", "22/09/2026", "2026-99-99"]:
        data = {"mission_id": "M1", "main_message": "Msg", "max_hashtags": 5, "deadline": d}
        file = tmp_path / f"m_{d.replace('/', '')}.json"
        with open(file, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="deadline must be in ISO 8601 format"):
            MissionParser.parse_json_file(str(file))

def test_missing_field(tmp_path):
    data = {"mission_id": "M1"}
    file = tmp_path / "m.json"
    with open(file, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="Missing or empty required field"):
        MissionParser.parse_json_file(str(file))

def test_malformed_json(tmp_path):
    file = tmp_path / "m.json"
    with open(file, "w") as f:
        f.write("{ invalid json")
    with pytest.raises(ValueError, match="Malformed JSON"):
        MissionParser.parse_json_file(str(file))

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        MissionParser.parse_json_file("nonexistent.json")
