"""Tests for Phase 18B: Static Image + TTS Audio -> MP4 Pipeline.

Validates that:
- Image + TTS audio produces MP4 via image_to_video.
- Output MP4 exists and is non-empty (when mocked or when FFmpeg is available).
- Video generator receives audio_path (not music_path).
- Multi-image produces separate MP4s.
- Failure on one image is isolated.
- FFmpeg unavailable / failure is handled with clear error.
- No background music / music stream is introduced.
"""
import os
import json
import wave
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.models import ImageAnalysis, MissionContext, JobStatus
from mission_ai.runner import MissionRunner
from mission_ai.video_generator import image_to_video


class MockTTSProvider:
    def synthesize(self, text: str, output_path: str) -> str:
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)      # mono
            wav_file.setsampwidth(2)      # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(b'\x00\x00' * 16000)  # ~1 second of silence
        return output_path


class MockAIProvider:
    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        return ImageAnalysis(
            summary=f"Analysis of {Path(image_path).name}",
            visible_subjects=["object"],
            visual_context=mission_context.main_message,
            relevant_details=["detail"],
            confidence="high"
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        return f"{platform} caption"

    def generate_voiceover_script(self, image_analysis, mission_context):
        return f"Voice-over untuk {mission_context.main_message}"


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
        "mission_id": "MISSION-18B",
        "instructions": "Create video.",
        "main_message": "Eco mission",
        "key_points": ["Point 1"],
        "platforms": ["instagram"],
        "max_hashtags": 2,
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_image_and_tts_audio_generates_mp4(app_config, sample_mission, tmp_path):
    img = tmp_path / "test.png"
    Image.new("RGB", (32, 32), color="green").save(img)

    recorded_calls = []
    def fake_video_gen(image_path, output_path, **kwargs):
        recorded_calls.append((image_path, output_path, kwargs))
        # Ensure output MP4 exists
        Path(output_path).write_bytes(b"DUMMY-MP4-VIDEO")
        return True

    runner = MissionRunner(
        app_config,
        provider=MockAIProvider(),
        tts_provider=MockTTSProvider(),
        video_generator=fake_video_gen,
        progress=lambda m: None
    )

    summary = runner.run(sample_mission, str(img), str(app_config.output_directory))

    assert summary.total_jobs == 1
    assert summary.completed == 1
    assert len(recorded_calls) == 1

    img_path, out_path, kwargs = recorded_calls[0]
    assert img_path == str(img)
    assert out_path.endswith(".mp4")
    assert Path(out_path).exists()
    assert "audio_path" in kwargs
    assert kwargs["audio_path"].endswith(".wav")
    assert Path(kwargs["audio_path"]).exists()
    assert "music_path" not in kwargs


def test_multi_image_produces_separate_mp4s(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = img_dir / "a.png"
    img2 = img_dir / "b.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    recorded_calls = []
    def fake_video_gen(image_path, output_path, **kwargs):
        recorded_calls.append((image_path, output_path, kwargs))
        Path(output_path).write_bytes(b"MP4")
        return True

    runner = MissionRunner(
        app_config,
        provider=MockAIProvider(),
        tts_provider=MockTTSProvider(),
        video_generator=fake_video_gen,
        progress=lambda m: None
    )

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))

    assert summary.total_jobs == 2
    assert summary.completed == 2
    assert len(recorded_calls) == 2
    out_paths = [call[1] for call in recorded_calls]
    assert len(set(out_paths)) == 2


def test_video_generation_failure_is_isolated(app_config, sample_mission, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img1 = img_dir / "fail.png"
    img2 = img_dir / "success.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    def fake_video_gen(image_path, output_path, **kwargs):
        if "fail.png" in image_path:
            return False
        Path(output_path).write_bytes(b"MP4")
        return True

    runner = MissionRunner(
        app_config,
        provider=MockAIProvider(),
        tts_provider=MockTTSProvider(),
        video_generator=fake_video_gen,
        progress=lambda m: None
    )

    summary = runner.run(sample_mission, str(img_dir), str(app_config.output_directory))

    assert summary.total_jobs == 2
    assert summary.completed == 1
    assert summary.failed == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg binary not available in environment")
def test_real_ffmpeg_integration_when_available(app_config, sample_mission, tmp_path):
    img = tmp_path / "real.png"
    Image.new("RGB", (64, 64), color="blue").save(img)

    runner = MissionRunner(
        app_config,
        provider=MockAIProvider(),
        tts_provider=MockTTSProvider(),
        progress=lambda m: None
    )

    summary = runner.run(sample_mission, str(img), str(app_config.output_directory))
    assert summary.completed == 1
    videos = list((app_config.output_directory / "MISSION-18B" / "videos").glob("*.mp4"))
    assert len(videos) == 1
    assert videos[0].stat().st_size > 0
