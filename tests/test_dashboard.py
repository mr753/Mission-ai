"""Tests for Phase 20: Mission AI Local Web Dashboard.

Validates that:
- Dashboard can load mission results and list missions.
- Multiple images/jobs appear independently.
- Completed job displays its generated outputs (video, script, captions, hashtags).
- Failed job does not crash the dashboard and displays error status.
- Missing optional output is handled safely.
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


def test_dashboard_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_loads_mission_results_and_posts(client, app_config, tmp_path):
    mission_id = "MISSION-TEST-20"
    mission_dir = Path(app_config.output_directory) / mission_id
    videos_dir = mission_dir / "videos"
    metadata_dir = mission_dir / "metadata"
    captions_dir = mission_dir / "captions"
    voiceovers_dir = mission_dir / "voiceovers"

    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)
    voiceovers_dir.mkdir(parents=True, exist_ok=True)

    job1_id = "job_001"
    job2_id = "job_002"

    checkpoint_data = {
        "state": {
            job1_id: "COMPLETED",
            job2_id: "FAILED"
        }
    }
    (mission_dir / "checkpoint.json").write_text(json.dumps(checkpoint_data))

    meta1 = {
        "image_id": job1_id,
        "source_path": str(tmp_path / "img1.png"),
        "status": "COMPLETED",
        "video_path": str(videos_dir / f"{job1_id}.mp4"),
        "analysis": {
            "summary": "Summary 1",
            "visible_subjects": ["subject1"],
            "visual_context": "context1",
            "relevant_details": ["detail1"]
        },
        "captions": [
            {
                "platform": "instagram",
                "caption": "Caption 1",
                "hashtags": ["#tag1", "#tag2"]
            }
        ],
        "voiceover_script": {
            "image_id": job1_id,
            "script_text": "Script 1 in Indonesian",
            "language": "id"
        }
    }
    (metadata_dir / f"{job1_id}.json").write_text(json.dumps(meta1))
    (videos_dir / f"{job1_id}.mp4").write_bytes(b"fake mp4")
    (captions_dir / f"{job1_id}_caption.txt").write_text("Caption 1\n#tag1 #tag2")
    (voiceovers_dir / f"{job1_id}_voiceover.txt").write_text("Script 1 in Indonesian")

    meta2 = {
        "image_id": job2_id,
        "source_path": str(tmp_path / "img2.png"),
        "status": "FAILED",
        "error": "Analysis timed out"
    }
    (metadata_dir / f"{job2_id}.json").write_text(json.dumps(meta2))

    resp_missions = client.get("/api/missions")
    assert resp_missions.status_code == 200
    missions = resp_missions.json()
    assert any(m["mission_id"] == mission_id for m in missions)

    resp_posts = client.get(f"/api/missions/{mission_id}/posts")
    assert resp_posts.status_code == 200
    data = resp_posts.json()
    assert data["mission_id"] == mission_id
    posts = data["posts"]
    assert len(posts) == 2

    post1 = next(p for p in posts if p["image_id"] == job1_id)
    assert post1["status"] == "COMPLETED"
    assert post1["analysis"]["summary"] == "Summary 1"
    assert post1["voiceover_script"]["script_text"] == "Script 1 in Indonesian"
    assert len(post1["captions"]) == 1

    post2 = next(p for p in posts if p["image_id"] == job2_id)
    assert post2["status"] == "FAILED"
    assert post2.get("error") == "Analysis timed out"


def test_dashboard_missing_optional_output_safely(client, app_config):
    mission_id = "MISSION-PARTIAL"
    mission_dir = Path(app_config.output_directory) / mission_id
    metadata_dir = mission_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    job_id = "job_partial"
    meta = {
        "image_id": job_id,
        "status": "COMPLETED",
        "analysis": {"summary": "Partial", "visible_subjects": [], "visual_context": "", "relevant_details": []},
        "captions": []
    }
    (metadata_dir / f"{job_id}.json").write_text(json.dumps(meta))

    resp = client.get(f"/api/missions/{mission_id}/posts")
    assert resp.status_code == 200
    posts = resp.json()["posts"]
    assert len(posts) == 1
    assert posts[0]["image_id"] == job_id
    assert posts[0].get("voiceover_script") is None


def test_dashboard_accounts_api(client):
    response = client.get("/api/accounts")
    assert response.status_code == 200
    data = response.json()
    assert "accounts" in data
    accounts = data["accounts"]
    platforms = {a["platform"] for a in accounts}
    assert {"youtube", "instagram", "facebook", "x", "threads"}.issubset(platforms)
    for a in accounts:
        assert "status" in a
        assert "has_credential_ref" in a
        assert "credential_ref" not in a
        assert a["status"] == "disconnected"


def test_mission_deletion_success(client, app_config):
    mission_id = "MISSION-TO-DELETE"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)
    (mission_dir / "checkpoint.json").write_text('{"job_1": "COMPLETED"}')

    assert mission_dir.exists()
    resp = client.delete(f"/api/missions/{mission_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert not mission_dir.exists()


def test_mission_deletion_not_found(client):
    resp = client.delete("/api/missions/NON-EXISTENT-MISSION")
    assert resp.status_code == 404


def test_mission_deletion_path_traversal(client):
    resp = client.delete("/api/missions/..%2Fetc")
    assert resp.status_code in (400, 404, 422)


def test_mission_deletion_processing_refused(client, app_config):
    mission_id = "MISSION-PROCESSING"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)
    (mission_dir / "checkpoint.json").write_text('{"job_1": "PROCESSING"}')

    resp = client.delete(f"/api/missions/{mission_id}")
    assert resp.status_code == 400
    assert "processing" in resp.json()["detail"].lower()
    assert mission_dir.exists()
    import shutil
    shutil.rmtree(mission_dir)


def test_mission_activity_log(client, app_config):
    mission_id = "MISSION-ACTIVITY"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)
    log_data = {
        "mission_id": mission_id,
        "stages": [
            {"stage_name": "Mission Initialization", "status": "COMPLETED", "timestamp": "2026-10-10T00:00:00Z", "message": "Initialized"}
        ]
    }
    (mission_dir / "activity_log.json").write_text(json.dumps(log_data))

    resp = client.get(f"/api/missions/{mission_id}/activity")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mission_id"] == mission_id
    assert len(data["stages"]) == 1
    assert data["stages"][0]["stage_name"] == "Mission Initialization"
    import shutil
    shutil.rmtree(mission_dir)


def test_dashboard_processing_job_without_metadata(client, app_config):
    mission_id = "MISSION-PROC-TEST"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)
    (mission_dir / "checkpoint.json").write_text('{"job_active": "PROCESSING"}')

    resp = client.get(f"/api/missions/{mission_id}/posts")
    assert resp.status_code == 200
    posts = resp.json()["posts"]
    assert len(posts) == 1
    assert posts[0]["image_id"] == "job_active"
    assert posts[0]["status"] == "PROCESSING"
    assert "currently processing" in posts[0]["error"].lower()

    import shutil
    shutil.rmtree(mission_dir)


def test_dashboard_worker_failure_logs_failed_stage(client, app_config):
    mission_id = "MISSION-FAIL-TEST"
    mission_dir = Path(app_config.output_directory) / mission_id
    mission_dir.mkdir(parents=True, exist_ok=True)

    log_path = mission_dir / "activity_log.json"
    log_path.write_text(json.dumps({
        "mission_id": mission_id,
        "stages": [
            {"stage_name": "Mission Initialization", "status": "COMPLETED", "timestamp": "2026-10-15T00:00:00Z", "message": "parsed"},
            {"stage_name": "Pipeline Execution", "status": "FAILED", "timestamp": "2026-10-15T00:00:01Z", "message": "PipelineError: Input path does not exist"}
        ]
    }))

    resp = client.get(f"/api/missions/{mission_id}/activity")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mission_id"] == mission_id
    assert len(data["stages"]) == 2
    assert data["stages"][1]["status"] == "FAILED"
    assert "PipelineError" in data["stages"][1]["message"]

    import shutil
    shutil.rmtree(mission_dir)
