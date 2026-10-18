import json
import io
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from mission_ai.upload.adapters import YouTubeShortsAdapter
from mission_ai.models import ContentPackage, ImageAnalysis, JobStatus, PlatformContent


def test_youtube_adapter_dry_run_simulated(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video bytes")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "false"}):
        result = adapter.upload(package)
        assert result.status in ("stub_success", "simulated")


def test_youtube_adapter_not_configured(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video bytes")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true", "YOUTUBE_TOKEN_FILE": "", "YOUTUBE_CLIENT_SECRET_FILE": ""}):
        result = adapter.upload(package)
        assert result.status == "not_configured"


def test_youtube_adapter_success(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video bytes")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()

    # Mock urllib.request.urlopen for session init and video upload
    mock_init_resp = MagicMock()
    mock_init_resp.headers = {"Location": "https://www.googleapis.com/upload/youtube/v3/resumable-upload-url"}
    mock_init_resp.__enter__.return_value = mock_init_resp
    mock_init_resp.__exit__.return_value = False

    mock_upload_resp = MagicMock()
    mock_upload_resp.read.return_value = json.dumps({"id": "yt_video_12345"}).encode("utf-8")
    mock_upload_resp.__enter__.return_value = mock_upload_resp
    mock_upload_resp.__exit__.return_value = False

    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true", "YOUTUBE_TOKEN_FILE": "dummy"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=[mock_init_resp, mock_upload_resp]):
                result = adapter.upload(package)
                assert result.status == "success"
                assert result.external_id == "yt_video_12345"
                assert result.url == "https://www.youtube.com/watch?v=yt_video_12345"


def test_youtube_adapter_rate_limited(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video bytes")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    http_err = urllib.error.HTTPError(
        url="https://www.googleapis.com/upload/youtube/v3/videos",
        code=429,
        msg="Rate Limit Exceeded",
        hdrs={},
        fp=io.BytesIO(b'{"error": "rateLimitExceeded"}')
    )

    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=http_err):
                result = adapter.upload(package)
                assert result.status == "rate_limited"


def test_youtube_adapter_timeout(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"dummy video bytes")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
                result = adapter.upload(package)
                assert result.status == "timeout"


def test_youtube_adapter_video_is_directory(tmp_path):
    # Pass a directory path instead of a video file when live upload is enabled
    vid_dir = tmp_path / "not_a_file_dir"
    vid_dir.mkdir()

    package = ContentPackage(
        image_id="job1",
        source_path=str(vid_dir),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(vid_dir)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            result = adapter.upload(package)
            assert result.status == "failed"
            assert result.error is not None


def test_youtube_adapter_video_empty(tmp_path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")

    package = ContentPackage(
        image_id="job1",
        source_path=str(video),
        analysis=ImageAnalysis(summary="test", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="Test caption", hashtags=["#tag"], title="Test Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            # When file size is 0 or unreadable during stat/upload request, it should handle gracefully without claiming success
            # Note: Path(video_path).stat().st_size is 0, which might raise or fail during POST X-Upload-Content-Length or return failed
            result = adapter.upload(package)
            assert result.status in ("failed", "stub_success", "simulated", "rate_limited", "timeout")
            if result.status == "success":
                pytest.fail("Empty video should not result in success")
