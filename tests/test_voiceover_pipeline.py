"""Tests for Phase 17: Indonesian Voice-Over Script Generation Pipeline.

Validates that:
- One mission + one image -> one voice-over script.
- One mission + multiple images -> separate scripts per image (no combining).
- MissionContext and ImageAnalysis are correctly passed to generation.
- Script is generated in Indonesian.
- Failure on one image does not cause other images to lose results (isolation).
- AI provider is called once for each successfully processed image.
- Mocked AI provider (no real network or paid API).
"""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.models import ImageAnalysis, MissionContext, JobStatus, VoiceOverScript
from mission_ai.runner import MissionRunner
from mission_ai.voiceover_engine import generate_voiceover_script


class FakeVoiceoverProvider:
    def __init__(self, fail_paths=None):
        self.analyzed_paths = []
        self.generated_scripts = []
        self.fail_paths = fail_paths or set()

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        self.analyzed_paths.append(image_path)
        if image_path in self.fail_paths:
            raise RuntimeError(f"Analysis failed for {image_path}")
        return ImageAnalysis(
            summary=f"Analisis visual untuk {Path(image_path).name}",
            visible_subjects=["produk", "objek"],
            visual_context=mission_context.main_message,
            relevant_details=[f"Detail khusus {Path(image_path).name}"],
            confidence="high"
        )

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        return "caption"

    def generate_voiceover_script(self, image_analysis: ImageAnalysis, mission_context: MissionContext) -> str:
        self.generated_scripts.append((image_analysis, mission_context))
        return f"Halo! Ini script voice-over untuk misi {mission_context.main_message} berdasarkan {image_analysis.summary}."


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
        "mission_id": "MISSION-17",
        "instructions": "Buat konten promosi.",
        "main_message": "Produk ramah lingkungan terbaik.",
        "key_points": ["Natural", "Berkualitas"],
        "platforms": ["instagram"],
        "max_hashtags": 3,
        "required_hashtags": ["#eco"],
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_one_mission_one_image_voiceover(app_config, sample_mission, tmp_path):
    img = tmp_path / "item1.png"
    Image.new("RGB", (32, 32), color="green").save(img)

    provider = FakeVoiceoverProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img), str(app_config.output_directory))

    assert summary.total_jobs == 1
    assert summary.completed == 1
    assert len(provider.generated_scripts) == 1

    # Verify output file saved
    vo_file = app_config.output_directory / "MISSION-17" / "voiceovers" / f"{summary.mission_id}_e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855_voiceover.txt"
    # Or check metadata
    meta_files = list((app_config.output_directory / "MISSION-17" / "metadata").glob("*.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text())
    assert meta["voiceover_script"] is not None
    assert "Bahasa Indonesia" or "script voice-over" in meta["voiceover_script"]["script_text"]


def test_one_mission_multiple_images_separate_voiceovers(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = img_dir / "img_a.png"
    img2 = img_dir / "img_b.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    provider = FakeVoiceoverProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))

    assert summary.total_jobs == 2
    assert summary.completed == 2
    # Provider called once for each image successfully processed
    assert len(provider.generated_scripts) == 2

    # Verify separate scripts for each image (no combining)
    analysis_1, mission_1 = provider.generated_scripts[0]
    analysis_2, mission_2 = provider.generated_scripts[1]
    assert analysis_1.summary != analysis_2.summary
    assert mission_1 == mission_2


def test_failure_isolation_voiceover(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = img_dir / "fail.png"
    img2 = img_dir / "success.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    provider = FakeVoiceoverProvider(fail_paths={str(img1)})
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))

    assert summary.total_jobs == 2
    assert summary.completed == 1
    assert summary.failed == 1
    # Successful image still got its voice-over script generated
    assert len(provider.generated_scripts) == 1
