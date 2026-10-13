import pytest
from pathlib import Path
from PIL import Image

from mission_ai.upload import (
    UploadJob,
    UploadWorker,
    get_platform_adapter,
    list_supported_platforms,
    BasePlatformAdapter,
    register_adapter,
    UploadResult,
)
from mission_ai.models import ContentPackage, ImageAnalysis, JobStatus, PlatformContent
from mission_ai.output_manager import OutputManager

def test_registry_resolves_all_five_platforms():
    platforms = list_supported_platforms()
    expected = ["facebook", "instagram", "threads", "x", "youtube"]
    for p in expected:
        assert p in platforms
        adapter = get_platform_adapter(p)
        assert adapter is not None
        assert adapter.platform_name == p or (p == "x" and adapter.platform_name == "x")

def test_unknown_platform_handled_cleanly():
    with pytest.raises((ValueError, KeyError)):
        get_platform_adapter("non_existent_platform_999")

def test_upload_worker_one_platform(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    job = UploadJob(
        mission_id="M-1",
        video_path=str(video),
        caption="Test caption",
        hashtags=["#test"],
        selected_platforms=["instagram"]
    )

    worker = UploadWorker()
    batch = worker.process_job(job)

    assert batch.mission_id == "M-1"
    assert batch.success_count == 0
    assert batch.simulated_count == 1
    assert batch.failure_count == 0
    assert batch.not_configured_count == 0
    assert len(batch.results) == 1
    res = batch.results[0]
    assert res.platform == "instagram"
    assert res.status == "stub_success"
    assert "stub" in res.message.lower() or "not configured" in res.message.lower()

def test_upload_worker_multiple_platforms(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    platforms = ["youtube", "instagram", "facebook", "x", "threads"]
    job = UploadJob(
        mission_id="M-MULTI",
        video_path=str(video),
        caption="Multi platform test",
        hashtags=["#ai"],
        selected_platforms=platforms
    )

    worker = UploadWorker()
    batch = worker.process_job(job)

    assert batch.mission_id == "M-MULTI"
    assert batch.success_count == 0
    assert batch.simulated_count == len(platforms)
    assert batch.failure_count == 0
    assert len(batch.results) == len(platforms)
    for res in batch.results:
        assert res.status == "stub_success"

def test_failure_isolation_in_worker(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    # Define a failing adapter mock class
    class FailingAdapter(BasePlatformAdapter):
        @property
        def platform_name(self) -> str:
            return "failing_platform"
        def validate_package(self, package):
            return True
        def prepare_payload(self, package):
            return {}
        def upload(self, package, credentials=None):
            raise RuntimeError("Simulated network/API failure")

    register_adapter("failing_platform", FailingAdapter)

    job = UploadJob(
        mission_id="M-ISOLATION",
        video_path=str(video),
        caption="Isolation test",
        hashtags=["#iso"],
        selected_platforms=["instagram", "failing_platform", "youtube"]
    )

    worker = UploadWorker()
    batch = worker.process_job(job)

    assert batch.mission_id == "M-ISOLATION"
    assert batch.success_count == 0
    assert batch.simulated_count == 2
    assert batch.failure_count == 1
    assert len(batch.results) == 3

    statuses = {r.platform: r.status for r in batch.results}
    assert statuses["instagram"] == "stub_success"
    assert statuses["youtube"] == "stub_success"
    assert statuses["failing_platform"] == "failed"

def test_upload_worker_real_success_and_not_configured(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")

    class RealSuccessAdapter(BasePlatformAdapter):
        @property
        def platform_name(self) -> str:
            return "real_success"
        def validate_package(self, package):
            return True
        def prepare_payload(self, package):
            return {}
        def upload(self, package, credentials=None):
            return UploadResult(platform=self.platform_name, status="success", message="Real upload success")

    class NotConfiguredAdapter(BasePlatformAdapter):
        @property
        def platform_name(self) -> str:
            return "not_conf"
        def validate_package(self, package):
            return True
        def prepare_payload(self, package):
            return {}
        def upload(self, package, credentials=None):
            return UploadResult(platform=self.platform_name, status="not_configured", message="Not configured")

    register_adapter("real_success", RealSuccessAdapter)
    register_adapter("not_conf", NotConfiguredAdapter)

    job = UploadJob(
        mission_id="M-COUNTS",
        video_path=str(video),
        caption="Test counts",
        hashtags=["#test"],
        selected_platforms=["real_success", "not_conf", "instagram"]
    )

    worker = UploadWorker()
    batch = worker.process_job(job)

    assert batch.success_count == 1
    assert batch.simulated_count == 1
    assert batch.not_configured_count == 1
    assert batch.failure_count == 0

def test_empty_or_invalid_job_rejected(tmp_path):
    worker = UploadWorker()
    # Missing fields / invalid video path
    with pytest.raises(ValueError):
        worker.process_job(UploadJob(mission_id="", video_path="", caption="", hashtags=[], selected_platforms=[]))

    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy")
    with pytest.raises(ValueError):
        worker.process_job(UploadJob(mission_id="M-ERR", video_path=str(video), caption="", hashtags=[], selected_platforms=[]))

    with pytest.raises(ValueError):
        worker.process_job(UploadJob(mission_id="M-ERR", video_path="/non/existent/path.mp4", caption="", hashtags=[], selected_platforms=["instagram"]))

def test_manual_output_artifacts_remain_untouched(tmp_path):
    output_mgr = OutputManager(tmp_path, "MISSION-MANUAL")
    img_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32), color="red").save(img_path)

    video_path = output_mgr.videos_dir / "vid.mp4"
    video_path.write_bytes(b"mp4 data")

    package = ContentPackage(
        image_id="img_1",
        source_path=str(img_path),
        analysis=ImageAnalysis(summary="Manual", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="instagram", caption="Manual caption", hashtags=["#manual"])],
        status=JobStatus.COMPLETED,
        video_path=str(video_path)
    )

    output_mgr.save_package(package)

    # Verify manual output artifacts exist and are untouched by upload worker
    assert (output_mgr.metadata_dir / "img_1.json").exists()
    assert (output_mgr.captions_dir / "img_1_caption.txt").exists()
    assert video_path.exists()

    # Run upload worker on package
    worker = UploadWorker()
    batch = worker.process_package("MISSION-MANUAL", package, ["instagram", "youtube"])
    assert batch.simulated_count == 2
    assert batch.success_count == 0

    # Verify manual artifacts are still intact and unmodified
    assert (output_mgr.metadata_dir / "img_1.json").exists()
    assert (output_mgr.captions_dir / "img_1_caption.txt").exists()
    assert video_path.exists()
