"""Tests for the MissionRunner orchestration layer.

All external dependencies (AI provider, FFmpeg) are faked - no network,
no Gemini/Ollama, no real video encoding.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.models import ImageAnalysis, JobStatus, MissionContext
from mission_ai.runner import MissionRunner, PipelineError, RunSummary


class FakeProvider:
    """AI provider stub satisfying the AIProvider protocol."""

    def __init__(self, fail_on: set = None):
        self.analyzed: list = []
        self.captions: list = []
        self._fail_on = fail_on or set()

    def analyze_image(self, image_path, mission_context):
        if image_path in self._fail_on:
            raise RuntimeError(f"AI failure for {image_path}")
        self.analyzed.append(image_path)
        return ImageAnalysis(
            summary=f"summary of {Path(image_path).name}",
            visible_subjects=["subject"],
            visual_context="context",
            relevant_details=["detail"],
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        self.captions.append((image_analysis.summary, platform))
        return f"{platform} caption"


def fake_video_gen(image_path, output_path, **kwargs):
    Path(output_path).write_bytes(b"fake video")
    return True


@pytest.fixture
def config(tmp_path):
    return AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        default_duration=15,
        video_width=1080,
        video_height=1920,
        fps=30,
        ai_provider="gemini",
        gemini_model="gemini-1.5-flash",
        ollama_model="llama3",
        output_directory=tmp_path / "out",
    )


@pytest.fixture
def mission_file(tmp_path):
    data = {
        "mission_id": "M-TEST",
        "main_message": "Test message",
        "instructions": "Test instructions",
        "key_points": ["p1"],
        "platforms": ["instagram", "youtube"],
        "max_hashtags": 5,
        "required_hashtags": ["#req"],
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    paths = []
    for name in ("b.jpg", "a.png"):
        p = img_dir / name
        Image.new("RGB", (10, 10), color="white").save(p)
        paths.append(p)
    return img_dir, paths


def _make_runner(config, provider, video_gen=fake_video_gen):
    return MissionRunner(
        config,
        provider=provider,
        video_generator=video_gen,
        progress=lambda msg: None,
    )


# 1. Mission is loaded.
def test_mission_loaded(config, mission_file, images):
    img_dir, _ = images
    runner = _make_runner(config, FakeProvider())
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert summary.mission_id == "M-TEST"


# 2. Images become jobs.
def test_images_become_jobs(config, mission_file, images):
    img_dir, paths = images
    runner = _make_runner(config, FakeProvider())
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert summary.total_jobs == len(paths)
    assert summary.completed == len(paths)


# 3. Duplicate images are not processed twice.
def test_duplicate_images_processed_once(config, mission_file, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (10, 10)).save(img_dir / "one.jpg")
    dup_dir = tmp_path / "dups"
    dup_dir.mkdir()
    (dup_dir / "copy.jpg").write_bytes((img_dir / "one.jpg").read_bytes())
    runner = _make_runner(config, FakeProvider())
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    # build() receives rglob of img_dir only; duplicate in another dir
    # verifies builder dedup logic via direct call:
    from mission_ai.jobs.builder import ImageJobBuilder
    mission = MissionContext("M-TEST", "", "msg", [], ["ig"], 3)
    jobs = ImageJobBuilder().build(mission, [img_dir / "one.jpg", dup_dir / "copy.jpg"])
    assert len(jobs) == 1
    assert summary.total_jobs == 1


# 4. Completed checkpoint jobs are skipped.
def test_completed_jobs_skipped(config, mission_file, images):
    img_dir, _ = images
    from PIL import Image as PILImage
    # Build jobs first to obtain deterministic job ids.
    from mission_ai.jobs.builder import ImageJobBuilder
    from mission_ai.mission.parser import MissionParser
    mission = MissionParser.parse_json_file(str(mission_file))
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    jobs = ImageJobBuilder().build(mission, paths)
    ckpt_path = config.output_directory / mission.mission_id / "checkpoint.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(ckpt_path)
    for job in jobs:
        ckpt.update(job.job_id, JobStatus.COMPLETED)

    provider = FakeProvider()
    runner = _make_runner(config, provider)
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert summary.skipped == len(jobs)
    assert summary.completed == 0
    assert provider.analyzed == []  # nothing re-processed


# 5. AI analysis is called.
def test_ai_analysis_called(config, mission_file, images):
    img_dir, paths = images
    provider = FakeProvider()
    runner = _make_runner(config, provider)
    runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert len(provider.analyzed) == len(paths)


# 6. Captions are generated for configured platforms.
def test_captions_for_platforms(config, mission_file, images):
    img_dir, _ = images
    provider = FakeProvider()
    runner = _make_runner(config, provider)
    runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    platforms = {(c[1]) for c in provider.captions}
    assert platforms == {"instagram", "youtube"}
    # 2 captions per image (one per platform).
    assert len(provider.captions) == 2 * len(provider.analyzed)


# 7. Output package is saved.
def test_output_package_saved(config, mission_file, images):
    img_dir, _ = images
    runner = _make_runner(config, FakeProvider())
    runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    base = config.output_directory / "M-TEST"
    metadata_files = list((base / "metadata").glob("*.json"))
    caption_files = list((base / "captions").glob("*_caption.txt"))
    assert len(metadata_files) >= 1
    assert len(caption_files) >= 1
    meta = json.loads(metadata_files[0].read_text())
    assert meta["status"] == "COMPLETED"
    assert meta["video_path"]
    assert "analysis" in meta and "captions" in meta
    assert len(meta["captions"]) == 2


# 8. Successful jobs become COMPLETED.
def test_successful_jobs_completed(config, mission_file, images):
    img_dir, _ = images
    runner = _make_runner(config, FakeProvider())
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert summary.completed == summary.total_jobs
    assert summary.failed == 0
    ckpt = CheckpointManager(config.output_directory / "M-TEST" / "checkpoint.json")
    for job_id in ckpt.state:
        assert ckpt.is_completed(job_id)


# 9. Failed jobs become FAILED.
def test_failed_jobs_marked_failed(config, mission_file, images):
    img_dir, paths = images
    provider = FakeProvider(fail_on={str(paths[1])})
    runner = _make_runner(config, provider)
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert summary.failed == 1
    assert summary.completed == len(paths) - 1
    ckpt = CheckpointManager(config.output_directory / "M-TEST" / "checkpoint.json")
    failed = ckpt.failed_ids()
    assert len(failed) == 1
    # Failure metadata was saved.
    meta_file = config.output_directory / "M-TEST" / "metadata" / f"{failed[0]}.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["status"] == "FAILED"
    assert meta["error"]


# 10. One failed image does not prevent another image from processing.
def test_failure_isolation(config, mission_file, images):
    img_dir, paths = images
    provider = FakeProvider(fail_on={str(paths[0])})
    runner = _make_runner(config, provider)
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    # The other image was still analyzed.
    assert len(provider.analyzed) == 1
    assert summary.completed == 1
    assert summary.failed == 1


# 11. Resume skips completed jobs (and retries failed ones).
def test_resume_skips_completed(config, mission_file, images):
    img_dir, paths = images
    provider = FakeProvider(fail_on={str(paths[0])})
    runner = _make_runner(config, provider)
    first = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert first.failed == 1 and first.completed == 1

    # Second run: completed job is skipped, failed job is retried.
    provider2 = FakeProvider()
    runner2 = _make_runner(config, provider2)
    second = runner2.run(str(mission_file), str(img_dir), str(config.output_directory))
    assert second.skipped == 1
    assert second.completed == 1  # previously failed job retried successfully
    assert second.failed == 0
    assert len(provider2.analyzed) == 1


# 12. CLI run invokes the pipeline.
def test_cli_run_invokes_pipeline(config, mission_file, images, tmp_path):
    from click.testing import CliRunner as ClickRunner
    from mission_ai.cli.main import main

    img_dir, _ = images
    fake = FakeProvider()

    class _Stub:
        def __init__(self, cfg, progress=None):
            self.calls = {}

        def run(self, mission_path, input_path, output_dir):
            self.calls["args"] = (mission_path, input_path, output_dir)
            return RunSummary(mission_id="M-TEST", total_jobs=1, completed=1)

    created = {}

    def fake_build_runner(progress):
        r = _Stub(config, progress)
        created["runner"] = r
        return r

    click_runner = ClickRunner()
    with patch("mission_ai.cli.main._build_runner", side_effect=fake_build_runner):
        result = click_runner.invoke(
            main,
            ["run", "--mission", str(mission_file), "--input", str(img_dir),
             "--output", str(config.output_directory)],
        )
    assert result.exit_code == 0, result.output
    assert "M-TEST" in result.output
    assert created["runner"].calls["args"] == (
        str(mission_file), str(img_dir), str(config.output_directory),
    )


def test_cli_run_error_exit_code(mission_file, images, tmp_path):
    from click.testing import CliRunner as ClickRunner
    from mission_ai.cli.main import main

    img_dir, _ = images
    click_runner = ClickRunner()
    with patch("mission_ai.cli.main._build_runner", side_effect=ValueError("bad config")):
        result = click_runner.invoke(
            main,
            ["run", "--mission", str(mission_file), "--input", str(img_dir),
             "--output", str(tmp_path / "out")],
        )
    assert result.exit_code == 1


def test_no_supported_images_raises_pipeline_error(config, mission_file, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not an image")
    runner = _make_runner(config, FakeProvider())
    with pytest.raises(PipelineError):
        runner.run(str(mission_file), str(empty), str(config.output_directory))


def test_missing_input_raises(config, mission_file):
    runner = _make_runner(config, FakeProvider())
    with pytest.raises(PipelineError):
        runner.run(str(mission_file), "does-not-exist", str(config.output_directory))
