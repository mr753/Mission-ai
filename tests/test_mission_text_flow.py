"""Tests for Mission Text → Google Drive flow and parser extraction.

Validates:
- Google Drive URL extraction from mission text (including punctuation handling).
- Mission text parsing into MissionContext.
- Validation errors for empty mission, missing Drive URL, and invalid Drive URL.
- Dashboard API endpoint handling of mission_text.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from mission_ai.mission.parser import MissionParser, extract_google_drive_url
from mission_ai.config import AppConfig
from mission_ai.web.app import create_app


def test_extract_google_drive_url_variants():
    # Plain URL
    text1 = "Source: https://drive.google.com/drive/folders/abc123"
    assert extract_google_drive_url(text1) == "https://drive.google.com/drive/folders/abc123"

    # Trailing period
    text2 = "Check https://drive.google.com/drive/folders/abc123."
    assert extract_google_drive_url(text2) == "https://drive.google.com/drive/folders/abc123"

    # Trailing parenthesis
    text3 = "Folder (https://drive.google.com/drive/folders/abc123)"
    assert extract_google_drive_url(text3) == "https://drive.google.com/drive/folders/abc123"

    # Trailing comma
    text4 = "Link https://drive.google.com/drive/folders/abc123, please review"
    assert extract_google_drive_url(text4) == "https://drive.google.com/drive/folders/abc123"

    # Multiple Google Drive URLs (picks first deterministically)
    text5 = "First: https://drive.google.com/drive/folders/first123 and second: https://drive.google.com/drive/folders/second456"
    assert extract_google_drive_url(text5) == "https://drive.google.com/drive/folders/first123"

    # Non-Google URL
    text6 = "Visit https://example.com/docs"
    assert extract_google_drive_url(text6) is None


def test_parse_mission_text_success():
    mission_text = """
    MISI POSTING KONTEN
    Tanggal: 1 Oktober 2026

    1. PESAN UTAMA
    Dukung kampanye lingkungan bersih.

    2. POIN PENTING
    - Buat video edukatif
    - Sertakan pesan semangat
    - Gunakan gambar dari folder berikut:
    https://drive.google.com/drive/folders/testfolder789?usp=sharing
    """
    ctx = MissionParser.parse_mission_text(mission_text)
    assert ctx.mission_id.startswith("MISSION-")
    assert ctx.main_message == "Dukung kampanye lingkungan bersih."
    assert ctx.image_source_url == "https://drive.google.com/drive/folders/testfolder789?usp=sharing"
    assert len(ctx.key_points) >= 2


def test_parse_mission_text_validation_errors():
    # Empty mission text
    with pytest.raises(ValueError, match="Mission text is required"):
        MissionParser.parse_mission_text("")

    with pytest.raises(ValueError, match="Mission text is required"):
        MissionParser.parse_mission_text("   ")

    # Missing Google Drive URL
    with pytest.raises(ValueError, match="No supported Google Drive link was found"):
        MissionParser.parse_mission_text("This is mission text without any drive link.")

    # Invalid / unsupported Drive URL
    with pytest.raises(ValueError, match="Google Drive link is invalid or unsupported"):
        MissionParser.parse_mission_text("Check out https://drive.google.com/file/notafolder/abc")


@pytest.fixture
def client(tmp_path):
    config = AppConfig(output_directory=tmp_path / "out")
    app = create_app(config)
    return TestClient(app)


def test_dashboard_api_mission_text_success(client):
    valid_text = """
    Mission Title
    Source: https://drive.google.com/drive/folders/18Nb0sC1z3KB28h0oegTQOUknQOB0UzuU
    """
    response = client.post("/api/missions", data={"mission_text": valid_text})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "mission_id" in data


def test_dashboard_api_mission_text_errors(client):
    # Empty mission text
    resp1 = client.post("/api/missions", data={"mission_text": ""})
    assert resp1.status_code == 400
    assert resp1.json()["detail"] == "Mission text is required."

    # Missing Drive URL
    resp2 = client.post("/api/missions", data={"mission_text": "Just text, no link"})
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "No supported Google Drive link was found in the mission text."

    # Invalid Drive URL
    resp3 = client.post("/api/missions", data={"mission_text": "Invalid link https://drive.google.com/file/notafolder/abc"})
    assert resp3.status_code == 400
    assert resp3.json()["detail"] == "Google Drive link is invalid or unsupported."
