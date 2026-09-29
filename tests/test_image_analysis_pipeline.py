"""Tests for Phase 16: Mission Context + Individual Images Analysis Pipeline.

Validates that:
- One mission + one image works correctly.
- One mission + many images produces separate analyses per image.
- Mission context is passed to each image.
- AI provider is called for each image independently.
- Image 01 analysis does not influence image 02 analysis except through shared MissionContext.
- Failure on one image is isolated and does not silently succeed.
- Compatible with Google Drive downloaded image paths.
- Mocked AI provider (no real network or paid API).
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.models import ImageAnalysis, MissionContext, JobStatus
from mission_ai.image_analyzer import analyze_image
from mission_ai.runner import MissionRunner, RunSummary
from mission_ai.sources.google_drive import GoogleDriveFolderResolver


class MockAIProvider:
    def __init__(self, fail_paths=None):
        self.calls = []
        self.fail_paths = fail_paths or set()

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        self.calls.append((image_path, mission_context))
        if image_path in self.fail_paths:
            raise RuntimeError(f"Analysis failed for {image_path}")
        return ImageAnalysis(
            summary=f"Analysis of {Path(image_path).name} for mission {mission_context.mission_id}",
            visible_subjects=["subject_in_" + Path(image_path).stem],
            visual_context=mission_context.main_message,
            relevant_details=[f"Detail for {image_path}"],
            confidence="high"
        )

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        return f"{platform} caption"


@pytest.fixture
def app_config(tmp_path):
    return AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        output_directory=tmp_path / "out",
    )


@pytest.fixture
def sample_mission(tmp_path):
    data = {
        "mission_id": "MISSION-16",
        "instructions": "Analyze product features carefully.",
        "main_message": "Eco-friendly packaging initiative",
        "key_points": ["Recyclable", "Biodegradable"],
        "platforms": ["instagram"],
        "max_hashtags": 3,
        "required_hashtags": ["#green"],
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_one_mission_one_image_analysis(app_config, sample_mission, tmp_path):
    img = tmp_path / "product.png"
    Image.new("RGB", (32, 32), color="green").save(img)

    provider = MockAIProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)
    
    summary = runner.run(sample_mission, str(img), str(app_config.output_directory))
    
    assert summary.total_jobs == 1
    assert summary.completed == 1
    assert len(provider.calls) == 1
    call_path, call_mission = provider.calls[0]
    assert call_path == str(img)
    assert call_mission.mission_id == "MISSION-16"


def test_one_mission_many_images_independent_analyses(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "gallery"
    img_dir.mkdir()
    img1 = img_dir / "img_01.png"
    img2 = img_dir / "img_02.png"
    img3 = img_dir / "img_03.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)
    Image.new("RGB", (32, 32), color="yellow").save(img3)

    provider = MockAIProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)
    
    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))
    
    assert summary.total_jobs == 3
    assert summary.completed == 3
    assert len(provider.calls) == 3

    # Verify each image produced a separate analysis passed with the same mission context
    analyzed_paths = [call[0] for call in provider.calls]
    assert sorted(analyzed_paths) == sorted([str(img1), str(img2), str(img3)])

    for path, mission in provider.calls:
        assert mission.mission_id == "MISSION-16"
        assert mission.main_message == "Eco-friendly packaging initiative"


def test_failure_isolation_per_image(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "gallery"
    img_dir.mkdir()
    img1 = img_dir / "img_01.png"
    img2 = img_dir / "img_02.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    # Fail on img1, succeed on img2
    provider = MockAIProvider(fail_paths={str(img1)})
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)
    
    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))
    
    assert summary.total_jobs == 2
    assert summary.completed == 1
    assert summary.failed == 1
    assert len(provider.calls) == 2


def test_google_drive_input_compatibility(app_config, sample_mission, tmp_path):
    drive_img = tmp_path / "drive_download.png"
    Image.new("RGB", (32, 32), color="purple").save(drive_img)

    provider = MockAIProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    with patch("mission_ai.runner.is_google_drive_url", return_value=True), \
         patch.object(GoogleDriveFolderResolver, "resolve", return_value=[drive_img]):
        summary = runner.run(sample_mission, "https://drive.google.com/drive/folders/dummy123", str(app_config.output_directory))

    assert summary.total_jobs == 1
    assert summary.completed == 1
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == str(drive_img)
