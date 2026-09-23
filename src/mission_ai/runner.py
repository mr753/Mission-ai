"""End-to-end mission pipeline orchestration.

Pipeline per job:
  mission.json -> MissionParser -> input discovery -> ImageJobBuilder
  -> CheckpointManager (skip COMPLETED) -> ImageAnalyzer -> CaptionEngine
  -> MusicSelector -> VideoGenerator -> OutputManager -> checkpoint COMPLETED.

Individual job failures are isolated: the job is marked FAILED (retryable on
resume) and processing continues with the remaining jobs.

Dependency injection: tests can pass fake providers / video generators so no
network, AI API, or real FFmpeg is required.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from mission_ai.config import AppConfig
from mission_ai.models import ContentPackage, ImageAnalysis, ImageJob, JobStatus, MissionContext, PlatformContent
from mission_ai.caption_engine import generate_platform_content
from mission_ai.image_analyzer import analyze_image
from mission_ai.jobs.builder import ImageJobBuilder
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.music_selector import select_music
from mission_ai.output_manager import OutputManager
from mission_ai.video_generator import image_to_video

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PLATFORM = "instagram"
DEFAULT_MUSIC_MOOD = "default"

CHECKPOINT_FILENAME = "checkpoint.json"


class PipelineError(Exception):
    """Unrecoverable configuration/input error (CLI maps this to exit code 1)."""


@dataclass
class RunSummary:
    mission_id: str
    total_jobs: int = 0
    skipped: int = 0          # already COMPLETED in checkpoint
    completed: int = 0        # finished successfully this run
    failed: int = 0           # failed this run (retryable on resume)
    failed_job_ids: List[str] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return self.completed + self.failed

    def describe(self) -> str:
        lines = [
            f"Mission {self.mission_id}: {self.total_jobs} job(s)",
            f"  completed: {self.completed}",
            f"  skipped (already done): {self.skipped}",
            f"  failed: {self.failed}",
        ]
        if self.failed_job_ids:
            lines.append(f"  failed jobs (retryable): {', '.join(self.failed_job_ids)}")
        return "\n".join(lines)


class MissionRunner:
    def __init__(
        self,
        config: AppConfig,
        provider=None,
        video_generator: Optional[Callable] = None,
        progress: Optional[Callable[[str], None]] = None,
        state_sink=None,
    ):
        """`provider` and `video_generator` are injectable for testing;
        by default they are built from `config` (Gemini/Ollama) and the real
        FFmpeg wrapper."""
        self.config = config
        self._provider = provider
        self._video_generator = video_generator or image_to_video
        self._progress = progress or (lambda msg: print(msg))
        from mission_ai.jobs.sink import create_state_sink
        self.state_sink = state_sink or create_state_sink(self.config, self._progress)

    @property
    def provider(self):
        if self._provider is None:
            from mission_ai.providers import create_provider
            self._provider = create_provider(self.config)
        return self._provider

    def run(self, mission_path: str, input_path: str, output_dir: str) -> RunSummary:
        # 1. Load and validate the mission.
        from mission_ai.mission.parser import MissionParser
        mission: MissionContext = MissionParser.parse_json_file(mission_path)
        self._progress(f"Mission: {mission.main_message}")

        # --- Sink Mission ---
        try:
            self.state_sink.sink_mission(mission)
            self.state_sink.sink_event(mission.mission_id, "MISSION_STARTED", status="STARTED")
        except Exception as e:
            self._progress(f"StateSink error: {e}")

        # 2. Discover input images and build deterministic jobs.
        image_paths = self._discover_images(input_path)
        jobs = ImageJobBuilder().build(mission, image_paths)
        if not jobs:
            try:
                self.state_sink.sink_event(
                    mission.mission_id,
                    "MISSION_FAILED",
                    status="FAILED",
                    payload={"reason": f"No supported images found for input: {input_path}"}
                )
            except Exception as e:
                self._progress(f"StateSink error: {e}")
            raise PipelineError(
                f"No supported images found for input: {input_path} "
                f"(supported extensions: {sorted(SUPPORTED_IMAGE_EXTS)})"
            )

        output_root = Path(output_dir)
        checkpoint = CheckpointManager(output_root / mission.mission_id / CHECKPOINT_FILENAME)
        output_manager = OutputManager(output_root, mission.mission_id)

        summary = RunSummary(mission_id=mission.mission_id, total_jobs=len(jobs))

        # 3. Process each job; failures are isolated.
        for job in jobs:
            if checkpoint.is_completed(job.job_id):
                summary.skipped += 1
                self._progress(f"[{job.order + 1}/{len(jobs)}] SKIP {job.job_id} (already completed)")
                try:
                    self.state_sink.sink_job_status(
                        job.job_id,
                        mission.mission_id,
                        str(job.source_path),
                        job.order,
                        JobStatus.COMPLETED
                    )
                    self.state_sink.sink_event(
                        mission.mission_id,
                        "JOB_SKIPPED",
                        status="SKIPPED",
                        job_id=job.job_id
                    )
                except Exception as e:
                    self._progress(f"StateSink error: {e}")
                continue

            checkpoint.update(job.job_id, JobStatus.PROCESSING)
            self._progress(f"[{job.order + 1}/{len(jobs)}] Processing {job.job_id}")
            try:
                self.state_sink.sink_job_status(
                    job.job_id,
                    mission.mission_id,
                    str(job.source_path),
                    job.order,
                    JobStatus.PROCESSING
                )
                self.state_sink.sink_event(
                    mission.mission_id,
                    "JOB_PROCESSING",
                    status="PROCESSING",
                    job_id=job.job_id
                )
            except Exception as e:
                self._progress(f"StateSink error: {e}")

            try:
                self._process_job(mission, job, output_manager)
                checkpoint.update(job.job_id, JobStatus.COMPLETED)
                summary.completed += 1
                self._progress(f"[{job.order + 1}/{len(jobs)}] COMPLETED {job.job_id}")
                try:
                    self.state_sink.sink_job_status(
                        job.job_id,
                        mission.mission_id,
                        str(job.source_path),
                        job.order,
                        JobStatus.COMPLETED
                    )
                    self.state_sink.sink_event(
                        mission.mission_id,
                        "JOB_COMPLETED",
                        status="COMPLETED",
                        job_id=job.job_id
                    )
                except Exception as e:
                    self._progress(f"StateSink error: {e}")
            except Exception as e:  # noqa: BLE001 - isolate any per-job failure
                error = f"{type(e).__name__}: {e}"
                checkpoint.update(job.job_id, JobStatus.FAILED)
                try:
                    output_manager.save_error(job.job_id, job.source_path, error)
                except OSError as save_err:
                    error += f" (metadata save failed: {save_err})"
                summary.failed += 1
                summary.failed_job_ids.append(job.job_id)
                self._progress(f"[{job.order + 1}/{len(jobs)}] FAILED {job.job_id}: {error}")
                try:
                    self.state_sink.sink_job_status(
                        job.job_id,
                        mission.mission_id,
                        str(job.source_path),
                        job.order,
                        JobStatus.FAILED,
                        last_error=error
                    )
                    self.state_sink.sink_event(
                        mission.mission_id,
                        "JOB_FAILED",
                        status="FAILED",
                        job_id=job.job_id,
                        payload={"error": error}
                    )
                    # Sink fail output
                    from mission_ai.models import ContentPackage
                    pkg = ContentPackage(
                        image_id=job.job_id,
                        source_path=str(job.source_path),
                        analysis=None,
                        captions=[],
                        status=JobStatus.FAILED,
                        error=error
                    )
                    self.state_sink.sink_output(pkg)
                except Exception as sink_err:
                    self._progress(f"StateSink error: {sink_err}")

        # --- Sink Mission End ---
        try:
            m_status = "COMPLETED" if summary.failed == 0 else "FAILED"
            m_event = "MISSION_COMPLETED" if summary.failed == 0 else "MISSION_FAILED"
            self.state_sink.sink_event(
                mission.mission_id,
                m_event,
                status=m_status,
                payload={
                    "total_jobs": summary.total_jobs,
                    "completed": summary.completed,
                    "failed": summary.failed,
                    "skipped": summary.skipped
                }
            )
        except Exception as e:
            self._progress(f"StateSink error: {e}")

        return summary

    def _process_job(self, mission: MissionContext, job: ImageJob, output_manager: OutputManager) -> None:
        # Analysis via the configured AI provider.
        analysis: ImageAnalysis = analyze_image(job.source_path, mission, self.provider)

        # Platform-specific captions (+ hashtags via the caption engine).
        platforms = mission.platforms or [DEFAULT_PLATFORM]
        captions: List[PlatformContent] = [
            generate_platform_content(analysis, mission, self.provider, platform)
            for platform in platforms
        ]

        # Music (optional): generation still works without it.
        music = select_music(DEFAULT_MUSIC_MOOD, self.config.music_directory)

        # Video generation with configured ffmpeg/dimension settings.
        video_path = output_manager.videos_dir / f"{job.job_id}.mp4"
        ok = self._video_generator(
            job.source_path,
            str(video_path),
            duration=self.config.default_duration,
            width=self.config.video_width,
            height=self.config.video_height,
            fps=self.config.fps,
            ffmpeg_path=self.config.ffmpeg_path,
            music_path=str(music) if music else None,
        )
        if not ok:
            raise RuntimeError("Video generation failed (see FFmpeg output)")

        package = ContentPackage(
            image_id=job.job_id,
            source_path=job.source_path,
            analysis=analysis,
            captions=captions,
            status=JobStatus.COMPLETED,
            video_path=str(video_path),
        )
        output_manager.save_package(package)
        try:
            self.state_sink.sink_output(package)
        except Exception as e:
            self._progress(f"StateSink error: {e}")

    def _discover_images(self, input_path: str) -> List[Path]:
        path = Path(input_path)
        if not path.exists():
            raise PipelineError(f"Input path does not exist: {input_path}")

        if path.is_file():
            if path.suffix.lower() == ".json":
                # Image manifest: download/fetch referenced images.
                from mission_ai.sources.manifest import ManifestResolver
                resolver = ManifestResolver(download_dir=path.parent)
                resolved = resolver.resolve(str(path))
                if not resolved:
                    raise PipelineError(f"Manifest produced no usable images: {input_path}")
                return resolved
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                raise PipelineError(f"Unsupported image file: {input_path}")
            return [path]

        # Directory: deterministic ordering (sorted rglob).
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
