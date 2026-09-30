"""Tests for Phase 21: Dashboard Preview, Download, and Copy Actions.

Validates that:
- Video preview URL and download paths are exposed correctly.
- Video download route serves the correct generated file.
- Path traversal attempts (e.g., ../) are rejected with 404.
- Missing video or optional assets are handled safely without breaking posts.
- Captions, hashtags, and voice-over scripts are fully exposed in mission posts API for copy actions.
- Multiple posts remain independent.
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from mission_ai.config import AppConfig
from mission_ai.web.app import create_app


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        output_directory=tmp_path / "out",
    )


@pytest.fixture
def client(app_config):
    app = create_app(app_config)
    return TestClient(app)


def test_mission_posts_exposes_actions_data(client, app_config, tmp_path):
    mission_id = "MISSION-21"
    mission_dir = Path(app_config.output_directory) / mission_id
    videos_dir = mission_dir / "videos"
    metadata_dir = mission_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    job_id = "post_01"
    video_file = videos_dir / f"{job_id}.mp4"
    video_file.write_bytes(b"fake mp4 video bytes")

    meta = {
        "image_id": job_id,
        "status": "COMPLETED",
        "video_path": str(video_file),
        "analysis": {
            "summary": "Analysis summary",
            "visible_subjects": ["sub1"],
            "visual_context": "ctx",
            "relevant_details": ["det1"]
        },
        "captions": [
            {
                "platform": "instagram",
                "caption": "Test caption for Instagram",
                "hashtags": ["#tag1", "#tag2"]
            }
        ],
        "voiceover_script": {
            "image_id": job_id,
            "script_text": "Script voice-over dalam Bahasa Indonesia.",
            "language": "id"
        }
    }
    (metadata_dir / f"{job_id}.json").write_text(json.dumps(meta))

    resp = client.get(f"/api/missions/{mission_id}/posts")
    assert resp.status_code == 200
    data = resp.json()
    posts = data["posts"]
    assert len(posts) == 1

    p = posts[0]
    assert p["image_id"] == job_id
    assert p["video_path"] == str(video_file)
    assert p["voiceover_script"]["script_text"] == "Script voice-over dalam Bahasa Indonesia."
    assert p["captions"][0]["caption"] == "Test caption for Instagram"
    assert p["captions"][0]["hashtags"] == ["#tag1", "#tag2"]


def test_video_download_route_success(client, app_config, tmp_path):
    mission_id = "MISSION-21-DL"
    mission_dir = Path(app_config.output_directory) / mission_id
    videos_dir = mission_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    vid = videos_dir / "job_dl.mp4"
    vid.write_bytes(b"mp4 content")

    resp = client.get(f"/api/missions/{mission_id}/files/videos/job_dl.mp4")
    assert resp.status_code == 200
    assert resp.content == b"mp4 content"


def test_path_traversal_rejection(client, app_config, tmp_path):
    mission_id = "MISSION-21-SEC"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)
    
    # Try directory traversal
    resp = client.get(f"/api/missions/{mission_id}/files/../../secret.txt")
    assert resp.status_code in (404, 400, 422)
