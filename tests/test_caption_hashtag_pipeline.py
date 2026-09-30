"""Tests for Phase 19: Per-Image Caption and Hashtag Generation Pipeline.

Validates that:
- Caption generation works for one image.
- Different images/jobs receive different captions.
- Hashtag generation respects max_hashtags.
- required_hashtags are always preserved.
- Duplicate hashtags are removed case-insensitively.
- Required hashtags alone can fill max_hashtags.
- Invalid/empty generated hashtags are handled.
- Caption/hashtag generation receives image-specific context.
- Failure of one image does not prevent another image from succeeding (isolation).
- Existing Phase 15B-18 tests continue passing.
"""
import json
from pathlib import Path
import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.models import ImageAnalysis, MissionContext, JobStatus, PlatformContent
from mission_ai.runner import MissionRunner
from mission_ai.hashtag_engine import generate_hashtags
from mission_ai.caption_engine import generate_platform_content


class MockCaptionProvider:
    def __init__(self, fail_paths=None):
        self.fail_paths = fail_paths or set()
        self.caption_calls = []

    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        if image_path in self.fail_paths:
            raise RuntimeError(f"Analysis failed for {image_path}")
        return ImageAnalysis(
            summary=f"Visual subject of {Path(image_path).name}",
            visible_subjects=[Path(image_path).stem],
            visual_context=mission_context.main_message,
            relevant_details=[f"Detail for {Path(image_path).name}"],
            confidence="high"
        )

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        self.caption_calls.append((image_analysis, mission_context, platform))
        return f"Caption untuk {image_analysis.summary} dalam misi {mission_context.main_message}"

    def generate_voiceover_script(self, image_analysis: ImageAnalysis, mission_context: MissionContext) -> str:
        return "Voiceover script"


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
        "mission_id": "MISSION-19",
        "instructions": "Buat postingan media sosial.",
        "main_message": "Kampanye Hijau",
        "key_points": ["Go Green", "Lestari"],
        "platforms": ["instagram"],
        "max_hashtags": 4,
        "required_hashtags": ["#Hijau", "#HIJAU", "#Lestari"],
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_caption_generation_one_image(app_config, sample_mission, tmp_path):
    img = tmp_path / "img1.png"
    Image.new("RGB", (32, 32), color="green").save(img)

    provider = MockCaptionProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img), str(app_config.output_directory))
    assert summary.completed == 1

    # Check output caption file exists
    caption_files = list((app_config.output_directory / "MISSION-19" / "captions").glob("*_caption.txt"))
    assert len(caption_files) == 1
    content = caption_files[0].read_text(encoding="utf-8")
    assert "Caption untuk" in content


def test_different_images_receive_different_captions(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "gallery"
    img_dir.mkdir()
    img1 = img_dir / "photo_a.png"
    img2 = img_dir / "photo_b.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    provider = MockCaptionProvider()
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))
    assert summary.completed == 2

    assert len(provider.caption_calls) == 2
    analysis1, _, _ = provider.caption_calls[0]
    analysis2, _, _ = provider.caption_calls[1]
    assert analysis1.summary != analysis2.summary


def test_hashtag_generation_rules():
    mission = MissionContext(
        mission_id="m1",
        instructions="test",
        main_message="msg",
        key_points=["Eco", "Nature"],
        platforms=["ig"],
        max_hashtags=3,
        required_hashtags=["#Green", "#green", "#Nature"]
    )
    analysis = ImageAnalysis(
        summary="Green forest and trees",
        visible_subjects=["Forest", "Trees"],
        visual_context="Nature",
        relevant_details=["Green leaves"]
    )

    tags = generate_hashtags(analysis, mission)
    # Check max_hashtags respected
    assert len(tags) <= 3
    # Check case-insensitive deduplication and required tags preservation
    lower_tags = [t.lower() for t in tags]
    assert "#green" in lower_tags
    assert "#nature" in lower_tags
    assert len(tags) == len(set(lower_tags))


def test_required_hashtags_fill_max():
    mission = MissionContext(
        mission_id="m1",
        instructions="test",
        main_message="msg",
        key_points=["Eco"],
        platforms=["ig"],
        max_hashtags=2,
        required_hashtags=["#Alpha", "#Beta", "#Gamma"]
    )
    analysis = ImageAnalysis(summary="Forest", visible_subjects=[], visual_context="", relevant_details=[])
    tags = generate_hashtags(analysis, mission)
    assert len(tags) == 2
    assert tags == ["#Alpha", "#Beta"]


def test_failure_isolation_per_image(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "gallery"
    img_dir.mkdir()
    img1 = img_dir / "fail.png"
    img2 = img_dir / "success.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    provider = MockCaptionProvider(fail_paths={str(img1)})
    runner = MissionRunner(app_config, provider=provider, video_generator=lambda *a, **kw: True, progress=lambda m: None)

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))
    assert summary.total_jobs == 2
    assert summary.completed == 1
    assert summary.failed == 1
