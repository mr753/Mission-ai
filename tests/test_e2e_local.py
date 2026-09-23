"""Phase 2E — Full local end-to-end validation.

Validates the whole local pipeline with NO network and NO external services:

  Mission JSON -> MissionParser -> image discovery -> ImageJobBuilder
  -> AI analysis (fake deterministic provider) -> platform caption generation
  -> hashtag validation -> real FFmpeg MP4 -> OutputManager -> checkpoint -> resume.

Design rules honored:
- Deterministic fake provider (satisfies the existing AIProvider protocol);
  Gemini/Ollama are never contacted.
- FFmpeg is REAL when available (skipped otherwise) and the produced file is
  verified as a genuine h264 MP4 via ffprobe (duration, resolution, pix_fmt).
- Video is never faked when FFmpeg is present: these tests fail if the MP4
  is not decodable.

Run: PYTHONPATH=src .venv/bin/pytest -q tests/test_e2e_local.py
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from PIL import Image

from mission_ai.caption_engine import generate_platform_content
from mission_ai.config import AppConfig
from mission_ai.hashtag_engine import generate_hashtags
from mission_ai.image_analyzer import analyze_image
from mission_ai.jobs.builder import ImageJobBuilder
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.mission.parser import MissionParser
from mission_ai.models import ImageAnalysis, JobStatus
from mission_ai.runner import MissionRunner

# Platform matrix required by Phase 2E point D.
ALL_PLATFORMS = ["instagram", "facebook", "threads", "tiktok", "youtube", "x"]


# ---------------------------------------------------------------- helpers

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe_video(path: Path) -> dict:
    """Return ffprobe JSON for a media file (raises on invalid media)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def probe_video_stream(path: Path) -> dict:
    info = probe_video(path)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    assert streams, f"no video stream in {path}"
    return streams[0]


def ffcfg(tmp_path: Path, duration: int = 1, fps: int = 15) -> AppConfig:
    """Config tuned for fast real-FFmpeg runs (1s videos, no music)."""
    return AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",  # absent -> silent video
        default_duration=duration,
        video_width=320,     # small for fast encoding
        video_height=480,
        fps=fps,
        ai_provider="gemini",  # irrelevant: provider is injected
        gemini_model="gemini-1.5-flash",
        ollama_model="llama3",
        output_directory=tmp_path / "output",
    )


class FakeProvider:
    """Deterministic AIProvider stub — no network, identical input => identical output."""

    def __init__(self, fail_on: set = None):
        self.analyzed: list = []
        self.captions: list = []
        self._fail_on = fail_on or set()

    def analyze_image(self, image_path, mission_context):
        if image_path in self._fail_on:
            raise RuntimeError(f"AI failure for {image_path}")
        self.analyzed.append(image_path)
        return ImageAnalysis(
            summary=f"deterministic summary of {Path(image_path).name}",
            visible_subjects=["synthetic-subject"],
            visual_context="synthetic local fixture",
            relevant_details=[f"detail-{Path(image_path).name}"],
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        self.captions.append((image_analysis.summary, platform))
        return f"{platform}: {image_analysis.summary} | {mission_context.main_message}"


def write_mission(path: Path, mission_id: str = "E2E-1", max_hashtags: int = 4,
                  required: list = None, platforms: list = None) -> Path:
    path.write_text(json.dumps({
        "mission_id": mission_id,
        "main_message": "Phase 2E validation mission",
        "instructions": "local deterministic validation",
        "key_points": ["point one", "point two"],
        "platforms": platforms if platforms is not None else ["instagram", "youtube"],
        "max_hashtags": max_hashtags,
        "required_hashtags": required if required is not None else ["#e2e", "#fixture"],
    }))
    return path


def make_images(directory: Path, names=("a.png", "b.jpg")) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        p = directory / name
        Image.new("RGB", (64, 48), color=(40 * i, 90, 200)).save(p)
        paths.append(p)
    return paths


def build_runner(cfg: AppConfig, provider, video_gen=None) -> MissionRunner:
    return MissionRunner(
        cfg, provider=provider,
        video_generator=video_gen,  # None -> real FFmpeg path
        progress=lambda msg: None,
    )


def run_via_cli(mission: Path, images: Path, output: Path, cfg: AppConfig,
                provider, command: str = "run"):
    """Invoke the real CLI entry point, injecting cfg/provider via the existing
    `_build_runner` DI seam (no monkeypatching of application code)."""
    from mission_ai.cli import main as cli

    def fake_build(progress):
        return MissionRunner(cfg, provider=provider, progress=progress)

    runner = CliRunner()
    with patch.object(cli, "_build_runner", side_effect=fake_build):
        result = runner.invoke(cli.main, [
            command,
            "--mission", str(mission),
            "--input", str(images),
            "--output", str(output),
        ])
    return result


# ---------------------------------------------------------------- A. Mission loading

def test_a_mission_loading(tmp_path):
    mission_file = write_mission(tmp_path / "mission.json")
    mission = MissionParser.parse_json_file(str(mission_file))

    assert mission.mission_id == "E2E-1"
    assert mission.main_message == "Phase 2E validation mission"
    assert mission.platforms == ["instagram", "youtube"]
    assert mission.max_hashtags == 4
    assert mission.required_hashtags == ["#e2e", "#fixture"]

    # Hashtag validation: required survive, cap is respected, output sorted.
    tags = generate_hashtags("anything", mission)
    assert set(mission.required_hashtags) <= set(tags)
    assert len(tags) <= mission.max_hashtags
    assert tags == sorted(tags)


# ---------------------------------------------------------------- B. Image discovery

def test_b_image_discovery_and_deterministic_jobs(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    img_dir = tmp_path / "input"
    make_images(img_dir)

    discovered = sorted(
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    assert len(discovered) == 2  # fixture images found

    jobs = ImageJobBuilder().build(mission, discovered)
    assert len(jobs) == 2
    for i, job in enumerate(jobs):
        assert isinstance(job.job_id, str)
        assert job.job_id.startswith("E2E-1_")
        assert job.order == i

    # Deterministic identity: same run -> same job ids (sha256-based).
    jobs_again = ImageJobBuilder().build(mission, discovered)
    assert [j.job_id for j in jobs] == [j.job_id for j in jobs_again]

    # Deterministic mapping id -> source path, independent of discovery order.
    shuffled = list(reversed(discovered))
    jobs_shuffled = ImageJobBuilder().build(mission, shuffled)
    assert {j.job_id: j.source_path for j in jobs_shuffled} == \
           {j.job_id: j.source_path for j in jobs}
    # Order index follows sorted path, not input order.
    assert [j.source_path for j in jobs_shuffled] == [j.source_path for j in jobs]


# ---------------------------------------------------------------- C. AI analysis

def test_c_ai_analysis_fake_provider_no_network(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")

    provider = FakeProvider()
    analysis = analyze_image(str(paths[0]), mission, provider)
    assert isinstance(analysis, ImageAnalysis)
    assert analysis.summary == f"deterministic summary of {paths[0].name}"
    assert provider.analyzed == [str(paths[0])]

    # Determinism: same input => identical analysis, second call, no hidden state.
    again = analyze_image(str(paths[0]), mission, provider)
    assert again == analysis

    # Provider abstraction: an object satisfying AIProvider is enough —
    # no Gemini/Ollama import or network happens anywhere in this flow.
    captions = generate_platform_content(analysis, mission, provider, "instagram")
    assert captions.caption.startswith("instagram:")
    assert mission.main_message in captions.caption


# ---------------------------------------------------------------- D. Platform content

@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_d_platform_content_matrix(tmp_path, platform):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")
    provider = FakeProvider()
    analysis = analyze_image(str(paths[0]), mission, provider)

    content = generate_platform_content(analysis, mission, provider, platform)

    assert content.platform == platform
    # Caption derives from analysis + mission context.
    assert analysis.summary in content.caption
    assert mission.main_message in content.caption
    # Required hashtags always present; never exceed max_hashtags.
    assert set(mission.required_hashtags) <= set(content.hashtags)
    assert len(content.hashtags) <= mission.max_hashtags

    # YouTube title/description support that actually exists in the code.
    if platform == "youtube":
        assert content.title
        assert content.description
        assert mission.main_message[:30] in content.title
    else:
        assert content.title is None and content.description is None


def test_d_hashtag_cap_enforced(tmp_path):
    # max_hashtags smaller than required set -> engine must cap, required wins.
    mission = MissionParser.parse_json_file(
        str(write_mission(tmp_path / "mission.json", max_hashtags=1)))
    assert generate_hashtags("x", mission) == ["#e2e"]  # sorted first


# ---------------------------------------------------------------- E. Video (real FFmpeg)

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not available")
def test_e_real_ffmpeg_video_validated(tmp_path):
    """One image -> exactly one real, ffprobe-verified MP4 (duration + resolution)."""
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path, duration=1, fps=15)

    runner = build_runner(cfg, FakeProvider())
    summary = runner.run(str(tmp_path / "mission.json"), str(tmp_path / "input"),
                         str(cfg.output_directory))
    assert summary.total_jobs == 2 and summary.completed == 2 and summary.failed == 0

    base = cfg.output_directory / "E2E-1"
    videos = sorted((base / "videos").glob("*.mp4"))
    assert len(videos) == 2  # exactly one MP4 per image

    for video in videos:
        stream = probe_video_stream(video)
        assert stream["codec_name"] == "h264"
        # yuvj420p is the deprecated full-range alias of yuv420p that FFmpeg 8
        # reports when the source image (JPEG/PNG) is full-range; both are
        # 8-bit 4:2:0 and universally playable (see limitation note).
        assert stream["pix_fmt"] in {"yuv420p", "yuvj420p"}
        assert stream["width"] == 320 and stream["height"] == 480
        duration = float(probe_video(video)["format"]["duration"])
        assert 0.9 <= duration <= 2.0  # ~1s requested
        assert video.stat().st_size > 0


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
@pytest.mark.skipif(not ffprobe_available(), reason="ffprobe not available")
def test_e_video_generator_direct_duration_and_music_absent(tmp_path):
    """Direct real-FFmpeg generation check without music (silent video)."""
    from mission_ai.video_generator import image_to_video

    png = make_images(tmp_path / "in", names=("solo.png",))[0]
    out = tmp_path / "solo.mp4"
    ok = image_to_video(str(png), str(out), duration=1, width=320, height=480,
                        fps=15, ffmpeg_path="ffmpeg", music_path=None)
    assert ok is True
    stream = probe_video_stream(out)
    assert stream["codec_name"] == "h264"
    duration = float(probe_video(out)["format"]["duration"])
    assert 0.9 <= duration <= 2.0
    # Silent: no audio stream expected when music is absent.
    info = probe_video(out)
    assert all(s.get("codec_type") != "audio" for s in info.get("streams", []))


# ---------------------------------------------------------------- F. Output structure

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_f_output_structure_and_metadata_linkage(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path)
    jobs = ImageJobBuilder().build(
        mission,
        sorted(p for p in paths) ,
    )

    runner = build_runner(cfg, FakeProvider())
    runner.run(str(tmp_path / "mission.json"), str(tmp_path / "input"),
               str(cfg.output_directory))

    base = cfg.output_directory / "E2E-1"
    assert (base / "videos").is_dir()
    assert (base / "captions").is_dir()
    assert (base / "metadata").is_dir()

    for job in jobs:
        meta_file = base / "metadata" / f"{job.job_id}.json"
        assert meta_file.exists(), f"missing metadata for {job.job_id}"
        meta = json.loads(meta_file.read_text())
        # Metadata links output back to source image + mission identity.
        assert meta["image_id"] == job.job_id
        assert Path(meta["source_path"]).name == Path(job.source_path).name
        assert meta["status"] == "COMPLETED"
        assert meta["video_path"] and Path(meta["video_path"]).exists()
        assert meta["analysis"]["summary"]
        assert {c["platform"] for c in meta["captions"]} == {"instagram", "youtube"}
        for c in meta["captions"]:
            assert set(mission.required_hashtags) <= set(c["hashtags"])

        # Caption file exists and mentions the platforms.
        cap_file = base / "captions" / f"{job.job_id}_caption.txt"
        assert cap_file.exists()
        text = cap_file.read_text()
        assert "---instagram---" in text and "---youtube---" in text


# ---------------------------------------------------------------- G. Checkpoint/resume

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_g_checkpoint_first_run_then_resume_skips(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path)

    # 1. First run: processed and COMPLETED.
    first = build_runner(cfg, FakeProvider()).run(
        str(tmp_path / "mission.json"), str(tmp_path / "input"), str(cfg.output_directory))
    assert first.completed == 2 and first.skipped == 0

    ckpt_file = cfg.output_directory / "E2E-1" / "checkpoint.json"
    ckpt = CheckpointManager(ckpt_file)
    assert len(ckpt.state) == 2
    assert all(ckpt.is_completed(jid) for jid in ckpt.state)
    first_ids = set(ckpt.state)

    # 2. Second run: completed jobs are NOT reprocessed (no AI calls, no re-encode).
    provider2 = FakeProvider()
    second = build_runner(cfg, provider2).run(
        str(tmp_path / "mission.json"), str(tmp_path / "input"), str(cfg.output_directory))
    assert second.skipped == 2 and second.completed == 0 and second.failed == 0
    assert provider2.analyzed == []  # nothing re-analyzed

    # 4. Checkpoint stays valid after both runs.
    ckpt_after = CheckpointManager(ckpt_file)
    assert set(ckpt_after.state) == first_ids
    assert all(ckpt_after.is_completed(jid) for jid in ckpt_after.state)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_g_checkpoint_failed_job_retried_on_resume(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path)

    # First run: one provider failure.
    failing = {str(paths[0])}
    first = build_runner(cfg, FakeProvider(fail_on=failing)).run(
        str(tmp_path / "mission.json"), str(tmp_path / "input"), str(cfg.output_directory))
    assert first.completed == 1 and first.failed == 1

    ckpt_file = cfg.output_directory / "E2E-1" / "checkpoint.json"
    ckpt = CheckpointManager(ckpt_file)
    failed_id = [jid for jid in ckpt.state if ckpt.status_of(jid) == JobStatus.FAILED]
    assert len(failed_id) == 1

    # Resume (CLI): completed skipped, failed retried and now completes.
    provider2 = FakeProvider()  # healthy provider
    result = run_via_cli(tmp_path / "mission.json", tmp_path / "input",
                         cfg.output_directory, cfg, provider2, command="resume")
    assert result.exit_code == 0, result.output
    assert "skipped (already done): 1" in result.output
    assert "completed: 1" in result.output
    assert len(provider2.analyzed) == 1  # only the failed job was retried

    ckpt_final = CheckpointManager(ckpt_file)
    assert all(ckpt_final.is_completed(jid) for jid in ckpt_final.state)


# ---------------------------------------------------------------- H. Failure isolation

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_h_failure_isolation_real_ffmpeg(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    paths = make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path)

    provider = FakeProvider(fail_on={str(paths[1])})
    summary = build_runner(cfg, provider).run(
        str(tmp_path / "mission.json"), str(tmp_path / "input"), str(cfg.output_directory))

    # The healthy job finished with a real MP4 despite the other failing.
    assert summary.completed == 1 and summary.failed == 1
    base = cfg.output_directory / "E2E-1"
    videos = list((base / "videos").glob("*.mp4"))
    assert len(videos) == 1
    stream = probe_video_stream(videos[0])
    assert stream["codec_name"] == "h264"

    # Error recorded in metadata + checkpoint; runner did not abort the mission.
    ckpt = CheckpointManager(base / "checkpoint.json")
    failed_ids = ckpt.failed_ids()
    assert len(failed_ids) == 1
    meta = json.loads((base / "metadata" / f"{failed_ids[0]}.json").read_text())
    assert meta["status"] == "FAILED"
    assert "RuntimeError" in meta["error"] or "AI failure" in meta["error"]
    assert Path(meta["source_path"]).name == paths[1].name


# ---------------------------------------------------------------- I. Determinism

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_i_pipeline_determinism_across_runs(tmp_path):
    """Two identical runs produce identical identity/mapping/metadata/structure
    (MP4 bytes intentionally NOT compared — container metadata is not deterministic)."""
    runs = []
    for i in range(2):
        out = tmp_path / f"run{i}" / "output"
        cfg = ffcfg(out.parent, duration=1, fps=15)
        write_mission(tmp_path / f"mission{i}.json")
        make_images(tmp_path / f"run{i}" / "input")

        provider = FakeProvider()
        summary = build_runner(cfg, provider).run(
            str(tmp_path / f"mission{i}.json"), str(tmp_path / f"run{i}" / "input"),
            str(cfg.output_directory))
        assert summary.completed == 2

        base = cfg.output_directory / "E2E-1"
        metas = {}
        for mf in sorted((base / "metadata").glob("*.json")):
            meta = json.loads(mf.read_text())
            meta["video_path"] = Path(meta["video_path"]).name  # path prefix differs
            meta["source_path"] = Path(meta["source_path"]).name
            metas[mf.name] = meta
        runs.append({
            "job_ids": sorted(metas.keys()),
            "metas": metas,
            "videos": sorted(v.name for v in (base / "videos").glob("*.mp4")),
            "captions": sorted(c.name for c in (base / "captions").glob("*_caption.txt")),
            "checkpoint": {k: v for k, v in sorted(
                CheckpointManager(base / "checkpoint.json").state.items())},
        })

    assert runs[0]["job_ids"] == runs[1]["job_ids"]
    assert runs[0]["videos"] == runs[1]["videos"]
    assert runs[0]["captions"] == runs[1]["captions"]
    assert runs[0]["checkpoint"] == runs[1]["checkpoint"]
    assert runs[0]["metas"] == runs[1]["metas"]


# ---------------------------------------------------------------- J. CLI

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_j_cli_run_status_resume_full_cycle(tmp_path):
    """Full run -> status -> resume cycle through the real CLI entry point."""
    mission = write_mission(tmp_path / "mission.json")
    make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path, duration=1, fps=15)
    provider = FakeProvider()

    # run
    result = run_via_cli(mission, tmp_path / "input", cfg.output_directory, cfg, provider, "run")
    assert result.exit_code == 0, result.output
    assert "completed: 2" in result.output
    assert "failed: 0" in result.output
    base = cfg.output_directory / "E2E-1"
    assert len(list((base / "videos").glob("*.mp4"))) == 2

    # status
    runner = CliRunner()
    from mission_ai.cli import main as cli
    with patch.object(cli, "_build_runner", side_effect=lambda p: None):
        status_result = runner.invoke(cli.main, [
            "status", "--mission", str(mission), "--output", str(cfg.output_directory)])
    assert status_result.exit_code == 0
    assert "all jobs completed" in status_result.output
    assert "videos: 2" in status_result.output

    # resume: everything skipped
    resume_result = run_via_cli(mission, tmp_path / "input", cfg.output_directory,
                                cfg, FakeProvider(), "resume")
    assert resume_result.exit_code == 0, resume_result.output
    assert "skipped (already done): 2" in resume_result.output


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_j_cli_run_reports_failure_but_stays_alive(tmp_path):
    """CLI stays exit-code-0 on per-job failure (mission continues); artifacts
    and failure metadata still written; rerun with healthy provider completes all."""
    mission = write_mission(tmp_path / "mission.json")
    paths = make_images(tmp_path / "input")
    cfg = ffcfg(tmp_path)

    result = run_via_cli(mission, tmp_path / "input", cfg.output_directory, cfg,
                         FakeProvider(fail_on={str(paths[0])}), "run")
    assert result.exit_code == 0, result.output
    assert "failed: 1" in result.output
    assert "completed: 1" in result.output
    base = cfg.output_directory / "E2E-1"
    assert len(list((base / "videos").glob("*.mp4"))) == 1  # healthy job produced MP4

    # Recovery via CLI resume with healthy provider.
    provider2 = FakeProvider()
    result2 = run_via_cli(mission, tmp_path / "input", cfg.output_directory, cfg, provider2, "resume")
    assert result2.exit_code == 0, result2.output
    assert "completed: 1" in result2.output and "skipped (already done): 1" in result2.output
    assert len(list((base / "videos").glob("*.mp4"))) == 2
