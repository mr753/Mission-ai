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


def test_youtube_token_still_valid(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    future_exp = time.time() + 3600
    token_data = {
        "access_token": "valid_access_token_123",
        "refresh_token": "refresh_token_abc",
        "expires_at": future_exp
    }
    token_file.write_text(json.dumps(token_data))

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        token = adapter._get_access_token()
        assert token == "valid_access_token_123"


def test_youtube_token_refresh_success(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    past_exp = time.time() - 100
    token_data = {
        "access_token": "expired_token",
        "refresh_token": "refresh_token_abc",
        "client_id": "client_123",
        "client_secret": "secret_456",
        "expires_at": past_exp
    }
    token_file.write_text(json.dumps(token_data))

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "access_token": "new_refreshed_access_token",
        "expires_in": 3600
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            token = adapter._get_access_token()
            assert token == "new_refreshed_access_token"

            # Verify updated token file content
            updated_data = json.loads(token_file.read_text())
            assert updated_data["access_token"] == "new_refreshed_access_token"
            assert "client_secret" not in updated_data  # client_secret must not be stored in token file


def test_youtube_token_expired_no_refresh_token(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    past_exp = time.time() - 100
    token_data = {
        "access_token": "expired_token",
        "expires_at": past_exp
        # No refresh_token
    }
    token_file.write_text(json.dumps(token_data))

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        token = adapter._get_access_token()
        # When expired and no refresh token, it should return the old token or None
        assert token == "expired_token" or token is None


def test_youtube_refresh_http_400_401(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    past_exp = time.time() - 100
    token_data = {
        "access_token": "expired_token",
        "refresh_token": "invalid_refresh",
        "client_id": "client_123",
        "client_secret": "secret_456",
        "expires_at": past_exp
    }
    token_file.write_text(json.dumps(token_data))

    http_err = urllib.error.HTTPError(
        url="https://oauth2.googleapis.com/token",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=io.BytesIO(b'{"error": "invalid_grant"}')
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        with patch("urllib.request.urlopen", side_effect=http_err):
            token = adapter._get_access_token()
            assert token is None


def test_youtube_refresh_timeout_and_invalid_json(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    past_exp = time.time() - 100
    token_data = {
        "access_token": "expired_token",
        "refresh_token": "refresh_123",
        "client_id": "client_123",
        "client_secret": "secret_456",
        "expires_at": past_exp
    }
    token_file.write_text(json.dumps(token_data))

    adapter = YouTubeShortsAdapter()

    # 1. Timeout test
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")):
            token = adapter._get_access_token()
            assert token is None

    # 2. Invalid JSON response test
    token_file.write_text(json.dumps(token_data))
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not valid json"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            token = adapter._get_access_token()
            assert token is None


def test_youtube_expired_token_returns_none(tmp_path):
    import time
    token_file = tmp_path / "token.json"
    past_exp = time.time() - 200
    token_data = {
        "access_token": "expired_access_token",
        "expires_at": past_exp
        # No refresh_token
    }
    token_file.write_text(json.dumps(token_data))

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"YOUTUBE_TOKEN_FILE": str(token_file)}):
        token = adapter._get_access_token()
        assert token is None


def test_youtube_token_uri_priority(tmp_path):
    adapter = YouTubeShortsAdapter()

    # 1. Credentials priority over token file and default
    creds = {"token_uri": "https://custom.cred/token", "client_id": "c1", "client_secret": "s1"}
    token_data = {"token_uri": "https://custom.file/token"}
    cid, csec, turi = adapter._get_client_secrets(creds, token_data)
    assert turi == "https://custom.cred/token"

    # 2. Token file priority over default when no creds token_uri
    cid, csec, turi = adapter._get_client_secrets(None, token_data)
    assert turi == "https://custom.file/token"


def test_youtube_token_atomic_write_location(tmp_path):
    token_file = tmp_path / "sub" / "token.json"
    adapter = YouTubeShortsAdapter()
    adapter._write_token_atomically(token_file, {"access_token": "new_val"})
    assert token_file.exists()
    assert json.loads(token_file.read_text())["access_token"] == "new_val"
    # Ensure no lingering tmp file in parent directory
    tmp_files = list(token_file.parent.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_youtube_token_atomic_write_failure(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"access_token": "old"}))

    adapter = YouTubeShortsAdapter()
    with patch.object(Path, "replace", side_effect=OSError("Disk write error")):
        with pytest.raises(RuntimeError, match="Failed to write token file atomically"):
            adapter._write_token_atomically(token_file, {"access_token": "new"})

        # Verify original token file is intact
        orig_data = json.loads(token_file.read_text())
        assert orig_data["access_token"] == "old"


def test_youtube_simulation_mode_no_real_upload(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_sim",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sim", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "false"}):
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = adapter.upload(package)
            assert result.status == "stub_success"
            mock_urlopen.assert_not_called()


def test_youtube_adapter_http_error_sanitizes_secret_response_body(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_sec",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sec", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    sensitive_token = "ya29.a0AfH6SMD_super_secret_token_12345"
    sensitive_secret = "GOCSPX-secret_client_key_999"
    body_content = f'{{"error": {{"code": 401, "message": "Invalid token: {sensitive_token}", "client_secret": "{sensitive_secret}"}}}}'
    http_err = urllib.error.HTTPError(
        url="https://www.googleapis.com/upload/youtube/v3/videos",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(body_content.encode("utf-8"))
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=http_err):
                result = adapter.upload(package)
                assert result.status == "failed"
                assert result.error is not None
                assert sensitive_token not in result.error
                assert sensitive_secret not in result.error
                assert "Invalid token" not in result.error
                assert "401" in result.error


def test_youtube_adapter_rate_limit_sanitizes_secret_response_body(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_rl_sec",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sec", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    secret_quota_id = "secret_project_id_private_999"
    body_content = f'{{"error": "rateLimitExceeded", "details": "{secret_quota_id}"}}'
    http_err = urllib.error.HTTPError(
        url="https://www.googleapis.com/upload/youtube/v3/videos",
        code=429,
        msg="Rate Limit Exceeded",
        hdrs={},
        fp=io.BytesIO(body_content.encode("utf-8"))
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=http_err):
                result = adapter.upload(package)
                assert result.status == "rate_limited"
                assert result.error is not None
                assert secret_quota_id not in result.error
                assert "rateLimitExceeded" not in result.error
                assert "429" in result.error


def test_youtube_adapter_server_error_sanitizes_secret_response_body(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_srv_sec",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sec", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    secret_db_url = "postgres://root:super_secret_db_pass@internal.db:5432/main"
    http_err = urllib.error.HTTPError(
        url="https://www.googleapis.com/upload/youtube/v3/videos",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=io.BytesIO(secret_db_url.encode("utf-8"))
    )

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=http_err):
                result = adapter.upload(package)
                assert result.status == "failed"
                assert result.error is not None
                assert secret_db_url not in result.error
                assert "super_secret_db_pass" not in result.error
                assert "500" in result.error


def test_youtube_adapter_url_error_sanitizes_secret_network_exception(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_net_sec",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sec", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    sensitive_endpoint = "https://user:ULTRA_SECRET_TOKEN@oauth2.internal.google.com/upload"
    url_err = urllib.error.URLError(f"Connection refused to {sensitive_endpoint}")

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=url_err):
                result = adapter.upload(package)
                assert result.status == "timeout"
                assert result.error is not None
                assert sensitive_endpoint not in result.error
                assert "ULTRA_SECRET_TOKEN" not in result.error
                assert "Connection refused" not in result.error


def test_youtube_adapter_unexpected_exception_sanitizes_secret(tmp_path):
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"dummy mp4")

    package = ContentPackage(
        image_id="job_exc_sec",
        source_path=str(video),
        analysis=ImageAnalysis(summary="sec", visible_subjects=[], visual_context="", relevant_details=[]),
        captions=[PlatformContent(platform="youtube", caption="cap", hashtags=["#shorts"], title="Title")],
        status=JobStatus.COMPLETED,
        video_path=str(video)
    )

    secret_key = "AIzaSySuperSecretInternalKey12345"
    exc = RuntimeError(f"Unexpected crash with sensitive token {secret_key}")

    adapter = YouTubeShortsAdapter()
    with patch.dict("os.environ", {"ENABLE_LIVE_UPLOAD": "true"}):
        with patch("mission_ai.upload.adapters.YouTubeShortsAdapter._get_access_token", return_value="fake_token"):
            with patch("urllib.request.urlopen", side_effect=exc):
                result = adapter.upload(package)
                assert result.status == "failed"
                assert result.error is not None
                assert secret_key not in result.error
                assert "Unexpected crash" not in result.error
