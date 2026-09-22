"""Tests for checkpoint reliability and video generator integration hooks."""
import json
from pathlib import Path

import pytest
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.models import JobStatus
from mission_ai.video_generator import image_to_video


def test_checkpoint_basic_roundtrip(tmp_path):
    ckpt = CheckpointManager(tmp_path / "state.json")
    ckpt.update("job1", JobStatus.COMPLETED)
    ckpt.update("job2", JobStatus.FAILED)
    assert ckpt.is_completed("job1")
    assert not ckpt.is_completed("job2")
    # Reload from disk.
    reloaded = CheckpointManager(tmp_path / "state.json")
    assert reloaded.is_completed("job1")
    assert reloaded.status_of("job2") == JobStatus.FAILED
    assert reloaded.failed_ids() == ["job2"]


def test_checkpoint_creates_parent_dirs(tmp_path):
    ckpt = CheckpointManager(tmp_path / "deep" / "nested" / "state.json")
    ckpt.update("job1", JobStatus.COMPLETED)
    assert (tmp_path / "deep" / "nested" / "state.json").exists()


def test_checkpoint_survives_corrupt_state(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{ not valid json !!")
    ckpt = CheckpointManager(state_file)
    # State resets instead of crashing, and writing still works.
    ckpt.update("job1", JobStatus.COMPLETED)
    reloaded = CheckpointManager(state_file)
    assert reloaded.is_completed("job1")


def test_checkpoint_recovers_from_backup(tmp_path):
    state_file = tmp_path / "state.json"
    ckpt = CheckpointManager(state_file)
    ckpt.update("job1", JobStatus.COMPLETED)
    # Second write archives the previous good state (job1 only) to .bak.
    ckpt.update("job2", JobStatus.COMPLETED)
    backup = state_file.with_suffix(".json.bak")
    assert backup.exists()
    # Corrupt the main file; the backup restores the last good state.
    state_file.write_text("corrupt")
    reloaded = CheckpointManager(state_file)
    assert reloaded.is_completed("job1")
    assert not reloaded.is_completed("job2")  # backup is one write behind


def test_video_generator_missing_image_returns_false(tmp_path):
    assert image_to_video(str(tmp_path / "missing.jpg"), str(tmp_path / "out.mp4")) is False


def test_video_generator_accepts_ffmpeg_path_kwarg():
    """The runner passes ffmpeg_path/music_path; signature must accept them."""
    import inspect
    params = inspect.signature(image_to_video).parameters
    assert "ffmpeg_path" in params
    assert "music_path" in params


def test_video_generator_reports_missing_ffmpeg(tmp_path, monkeypatch):
    """A missing ffmpeg binary returns False instead of raising."""
    img = tmp_path / "img.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img)
    result = image_to_video(
        str(img), str(tmp_path / "out.mp4"),
        duration=1, ffmpeg_path="definitely-not-a-real-binary-xyz",
    )
    assert result is False
