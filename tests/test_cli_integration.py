"""End-to-end CLI integration test using the local fixture.

No production code is modified: the AI provider and video generator are
injected as fakes through the existing dependency-injection seam
(`mission_ai.cli.main._build_runner`), so the test makes no network/API calls
and does not require FFmpeg.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mission_ai.cli.main import main
from mission_ai.config import AppConfig
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.models import ImageAnalysis
from mission_ai.runner import MissionRunner

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "integration"
MISSION_FILE = FIXTURE_DIR / "mission.json"
INPUT_DIR = FIXTURE_DIR / "input"


class FakeProvider:
    """AI provider stub satisfying the AIProvider protocol."""

    def analyze_image(self, image_path, mission_context):
        return ImageAnalysis(
            summary=f"Integration fixture image {Path(image_path).name}",
            visible_subjects=["pattern"],
            visual_context="synthetic local fixture",
            relevant_details=["red background", "blue squares"],
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        return f"{platform}: {image_analysis.summary}"


def fake_video_gen(image_path, output_path, **kwargs):
    Path(output_path).write_bytes(b"fake video bytes")
    return True


@pytest.fixture
def run_cli(tmp_path):
    """Invoke the real CLI with fakes injected via the existing DI seam."""
    output_dir = tmp_path / "output"
    captured = {}

    def fake_build_runner(progress):
        config = AppConfig(
            ffmpeg_path="ffmpeg",
            music_directory=tmp_path / "music",  # absent -> no music, video still works
            default_duration=15,
            video_width=1080,
            video_height=1920,
            fps=30,
            ai_provider="gemini",  # irrelevant: provider is injected below
            gemini_model="gemini-1.5-flash",
            ollama_model="llama3",
            output_directory=output_dir,
        )
        runner = MissionRunner(
            config,
            provider=FakeProvider(),
            video_generator=fake_video_gen,
            progress=progress,
        )
        captured["runner"] = runner
        return runner

    click_runner = CliRunner()
    with patch("mission_ai.cli.main._build_runner", side_effect=fake_build_runner):
        result = click_runner.invoke(
            main,
            [
                "run",
                "--mission", str(MISSION_FILE),
                "--input", str(INPUT_DIR),
                "--output", str(output_dir),
            ],
        )
    return result, output_dir, captured


def test_cli_pipeline_end_to_end(run_cli):
    result, output_dir, captured = run_cli
    assert result.exit_code == 0, result.output

    mission_dir = output_dir / "INTEG-1"

    # Required output structure exists.
    assert (mission_dir / "videos").is_dir()
    assert (mission_dir / "captions").is_dir()
    assert (mission_dir / "metadata").is_dir()

    # A checkpoint file exists and records a COMPLETED job for the fixture image.
    checkpoint_file = mission_dir / "checkpoint.json"
    assert checkpoint_file.exists()
    checkpoint = CheckpointManager(checkpoint_file)
    completed_ids = [job_id for job_id in checkpoint.state if checkpoint.is_completed(job_id)]
    assert len(completed_ids) == 1
    job_id = completed_ids[0]
    assert job_id.startswith("INTEG-1_")

    # Metadata contains the expected image/job information.
    meta_file = mission_dir / "metadata" / f"{job_id}.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["image_id"] == job_id
    assert meta["source_path"].endswith("mission_photo.png")
    assert meta["status"] == "COMPLETED"
    assert meta["video_path"]
    assert meta["analysis"]["summary"]
    assert {c["platform"] for c in meta["captions"]} == {"youtube", "instagram"}
    # Required hashtags preserved through the pipeline.
    for caption in meta["captions"]:
        assert "#integ" in caption["hashtags"]
        assert "#fixture" in caption["hashtags"]

    # Video and caption artifacts exist on disk.
    assert (Path(meta["video_path"])).exists()
    assert (mission_dir / "captions" / f"{job_id}_caption.txt").exists()

    # CLI reported success.
    assert "completed: 1" in result.output
    assert "failed: 0" in result.output
