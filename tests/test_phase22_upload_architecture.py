import json
from pathlib import Path
import pytest
from PIL import Image

from mission_ai.models import ContentPackage, PlatformContent, ImageAnalysis, JobStatus
from mission_ai.upload import get_platform_adapter
from mission_ai.output_manager import OutputManager


def test_upload_adapters_registry_and_payload(tmp_path):
    platforms = ["youtube", "instagram", "facebook", "x", "threads"]

    # Create sample package with video and captions using tmp_path
    img_path = tmp_path / "dummy_img.png"
    Image.new("RGB", (32, 32), color="green").save(img_path)

    package = ContentPackage(
        image_id="img_001",
        source_path=str(img_path),
        analysis=ImageAnalysis(summary="Test", visible_subjects=["subject"], visual_context="context", relevant_details=["detail"]),
        captions=[
            PlatformContent(platform="youtube", caption="YT Caption", hashtags=["#shorts", "#test"]),
            PlatformContent(platform="instagram", caption="IG Caption", hashtags=["#ig", "#test"]),
            PlatformContent(platform="facebook", caption="FB Caption", hashtags=["#fb", "#test"]),
            PlatformContent(platform="x", caption="X Caption", hashtags=["#x", "#test"]),
            PlatformContent(platform="threads", caption="Threads Caption", hashtags=["#threads", "#test"]),
        ],
        status=JobStatus.COMPLETED,
        video_path=str(img_path)
    )

    for p in platforms:
        adapter = get_platform_adapter(p)
        assert adapter.platform_name == p or (p == "x" and adapter.platform_name == "x")
        assert adapter.validate_package(package) is True

        payload = adapter.prepare_payload(package)
        assert isinstance(payload, dict)
        assert payload["platform"] == adapter.platform_name

        # Verify stub upload result indicates real API not configured (Phase 22C requirement)
        result = adapter.upload(package)
        assert result.status == "stub_success"
        assert "not configured" in result.message.lower() or "stub" in result.message.lower()


def test_manual_upload_readiness_artifacts(tmp_path):
    output_mgr = OutputManager(tmp_path, "MISSION-22B")

    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32), color="blue").save(img_path)

    video_path = output_mgr.videos_dir / "test_video.mp4"
    video_path.write_bytes(b"fake mp4 content")

    package = ContentPackage(
        image_id="test_img_1",
        source_path=str(img_path),
        analysis=ImageAnalysis(summary="Manual readiness test", visible_subjects=["art"], visual_context="studio", relevant_details=["blue"]),
        captions=[
            PlatformContent(platform="instagram", caption="Amazing manual post caption", hashtags=["#mission", "#ai", "#art"])
        ],
        status=JobStatus.COMPLETED,
        video_path=str(video_path)
    )

    output_mgr.save_package(package)

    # Verify artifacts exist and are ready for manual copy & download
    meta_file = output_mgr.metadata_dir / "test_img_1.json"
    cap_file = output_mgr.captions_dir / "test_img_1_caption.txt"

    assert meta_file.exists()
    assert cap_file.exists()
    assert video_path.exists()

    meta_data = json.loads(meta_file.read_text())
    assert meta_data["image_id"] == "test_img_1"
    assert meta_data["video_path"] == str(video_path)
    assert len(meta_data["captions"]) == 1
    assert meta_data["captions"][0]["caption"] == "Amazing manual post caption"
    assert "#mission" in meta_data["captions"][0]["hashtags"]

    cap_text = cap_file.read_text()
    assert "instagram" in cap_text
    assert "Amazing manual post caption" in cap_text
    assert "#mission" in cap_text
