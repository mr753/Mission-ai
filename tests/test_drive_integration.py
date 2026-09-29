"""Phase 15C — Google Drive resolver fixes (query encoding, u/0 URLs, refresh transport).

Targeted tests for:

  * _api_request builds query strings with urllib.parse.urlencode, so values
    containing spaces, quotes or unicode no longer break the request.
  * Google Drive URL formats are recognised: plain, /u/0 indexed view,
    trailing parameters.
  * Service-account token refresh uses google.auth.transport.requests.Request.
  * Access-token (primary) path still works.
  * Local directory / manifest / single-image chains are untouched.
"""
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mission_ai.config import AppConfig
from mission_ai.mission.parser import MissionParser
from mission_ai.runner import MissionRunner, PipelineError, is_google_drive_url
from mission_ai.sources.google_drive import GoogleDriveFolderResolver


DRIVE_URL = "https://drive.google.com/drive/folders/test123"
DRIVE_U0_URL = "https://drive.google.com/drive/u/0/folders/test456"


# ---------------------------------------------------------------- parser (regression) ---


def test_parser_propagates_image_source_url(tmp_path):
    mission = MissionParser.parse_json_file(
        str(write_mission(tmp_path / "mission.json",
                          image_source_url=DRIVE_URL)))
    assert mission.image_source_url == DRIVE_URL


def test_parser_image_source_url_defaults_to_none(tmp_path):
    mission = MissionParser.parse_json_file(str(write_mission(tmp_path / "mission.json")))
    assert mission.image_source_url is None


def test_parser_existing_fields_unchanged(tmp_path):
    mission = MissionParser.parse_json_file(
        str(write_mission(tmp_path / "mission.json",
                          key_points=["p1"], required_hashtags=["#req"],
                          platforms=["ig", "youtube"], max_hashtags=4)))
    assert mission.key_points == ["p1"]
    assert mission.required_hashtags == ["#req"]
    assert mission.platforms == ["ig", "youtube"]
    assert mission.max_hashtags == 4


# ---------------------------------------------------------------- URL detection (2A) ---


def test_is_google_drive_url():
    # Plain Drive folder URL
    assert is_google_drive_url("https://drive.google.com/drive/folders/test123") is True
    # /u/0/folders/<ID>
    assert is_google_drive_url("https://drive.google.com/drive/u/0/folders/test456") is True
    # Trailing slash
    assert is_google_drive_url("https://drive.google.com/drive/folders/test123/") is True
    # Query parameter
    assert is_google_drive_url("https://drive.google.com/drive/folders/test123?usp=sharing") is True
    # Subpath
    assert is_google_drive_url("https://drive.google.com/drive/folders/test123/sub") is True
    
    # Negative/non-Google-Drive URL
    assert is_google_drive_url("C:/Users/someone/images") is False
    assert is_google_drive_url("./input") is False
    assert is_google_drive_url("input/manifest.json") is False
    assert is_google_drive_url("https://docs.google.com/document/d/123") is False
    assert is_google_drive_url("https://folders.google.com/f/123") is False


# ---------------------------------------------------------------- query encoding (2B) ---

ENCODABLE_VALUES = [
    ("plain", "abc123 in parents and trashed = false"),      # spaces + '='
    ("single_quote", "'abc123' in parents and trashed = false"),  # apostrophe
    ("unicode", "folder/with spaces & symbols = > < \u2728"),   # spaces, symbols, unicode
]


@pytest.mark.parametrize("value_label,value", ENCODABLE_VALUES, ids=[v for v, _ in ENCODABLE_VALUES])
def test_api_request_query_encodes_spaces_quotes_and_unicode(value_label, value):
    """The query string must be RFC 3986 encoded (urllib.parse.urlencode)."""
    resolver = GoogleDriveFolderResolver(download_dir=Path("/tmp/drive"), access_token="tok")

    captured = {}
    with patch("urllib.request.Request") as mock_req_cls, \
            patch("urllib.request.urlopen") as mock_urlopen:

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self): return b'{"files": []}'

        mock_urlopen.return_value = _FakeResp()
        mock_req_cls.side_effect = lambda url, **kwargs: captured.update({"url": url}) or MagicMock()

        resolver._api_request("files", {"q": value, "fields": "files(id,name)", "pageSize": 100})

    url = captured.get("url", "")
    assert " " not in url, f"space leaked into query ({value_label})"
    assert "'" not in url, f"single quote leaked into query ({value_label})"
    assert "=" in url
    assert "q=" in url
    assert "fields=" in url
    assert "pageSize=100" in url


# ---------------------------------------------------------------- folder ID extraction (2C) ---


def test_resolver_extracts_folder_id_variants():
    resolver = GoogleDriveFolderResolver(download_dir=Path("/tmp/drive"), access_token="tok")
    
    # plain URL
    assert resolver._extract_folder_id("https://drive.google.com/drive/folders/ID123") == "ID123"
    # trailing slash
    assert resolver._extract_folder_id("https://drive.google.com/drive/folders/ID123/") == "ID123"
    # query parameter
    assert resolver._extract_folder_id("https://drive.google.com/drive/folders/ID123?x=y") == "ID123"
    # subfolder/subpath
    assert resolver._extract_folder_id("https://drive.google.com/drive/folders/ID123/sub") == "ID123"
    # /u/0/folders/<ID>
    assert resolver._extract_folder_id("https://drive.google.com/drive/u/0/folders/ID456") == "ID456"
    # /U/0/folders/<ID>
    assert resolver._extract_folder_id("https://drive.google.com/drive/U/0/folders/ID789") == "ID789"
    
    # also support the drive/googleusercontent.com variant requested in prompt
    assert resolver._extract_folder_id("https://drive/googleusercontent.com/drive/folders/ID000") == "ID000"


# ---------------------------------------------------------------- resolver invocation (2D) ---


def test_resolver_resolve_calls_extractor_on_url(tmp_path):
    cfg = make_config(tmp_path)
    resolver = GoogleDriveFolderResolver(download_dir=tmp_path / "downloads", access_token="tok")
    
    with patch.object(GoogleDriveFolderResolver, "_extract_folder_id", return_value="f123") as mock_extract, \
         patch.object(GoogleDriveFolderResolver, "_list_files_in_folder", return_value=[
             {"id": "i1", "name": "img1.png", "mimeType": "image/png"},
             {"id": "i2", "name": "img2.jpg", "mimeType": "image/jpeg"}
         ]), \
         patch.object(GoogleDriveFolderResolver, "_download_file", side_effect=lambda f: tmp_path / f["name"]):
        
        # Create dummy files with different content
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.jpg"
        Image.new("RGB", (16, 16), color="red").save(img1)
        Image.new("RGB", (16, 16), color="blue").save(img2)
        
        paths = resolver.resolve(DRIVE_U0_URL)
        mock_extract.assert_called_once_with(DRIVE_U0_URL)
        assert len(paths) == 2

    # Verify MissionRunner integration
    mission_file = write_mission(tmp_path / "mission.json")
    runner = MissionRunner(cfg, provider=FakeProvider(),
                           video_generator=fake_video_gen, progress=lambda m: None)
    
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (16, 16), color="red").save(img1)
    Image.new("RGB", (16, 16), color="blue").save(img2)

    with patch("mission_ai.sources.google_drive.GoogleDriveFolderResolver.resolve", return_value=[img1, img2]):
        summary = runner.run(str(mission_file), DRIVE_U0_URL, str(cfg.output_directory))
    
    assert summary.total_jobs == 2


# ---------------------------------------------------------------- refresh transport (2E) ---


def test_refresh_access_token_uses_google_auth_transport():
    resolver = GoogleDriveFolderResolver.__new__(GoogleDriveFolderResolver)
    resolver._credentials = MagicMock()
    
    # Mock google.auth.transport.requests.Request
    with patch("google.auth.transport.requests.Request") as mock_req_cls:
        mock_req_inst = mock_req_cls.return_value
        resolver._refresh_access_token()
        
        mock_req_cls.assert_called_once_with()
        resolver._credentials.refresh.assert_called_once_with(mock_req_inst)


# ---------------------------------------------------------------- access token path (2F) ---


def test_access_token_path_works_without_third_party():
    # Use constructor with access_token
    resolver = GoogleDriveFolderResolver(download_dir=Path("/tmp/drive"), access_token="tok123")
    assert resolver._access_token == "tok123"
    
    # Verify it doesn't try to refresh if token is present
    resolver._credentials = MagicMock()
    with patch.object(resolver, "_refresh_access_token") as mock_refresh:
        headers = resolver._get_auth_headers()
        assert headers["Authorization"] == "Bearer tok123"
        mock_refresh.assert_not_called()


# ---------------------------------------------------------------- missing auth (2G) ---


def test_missing_auth_raises_value_error():
    with patch.dict(os.environ, {}, clear=True):
        resolver = GoogleDriveFolderResolver(download_dir=Path("/tmp/drive"))
        with pytest.raises(ValueError, match="Google Drive authentication required"):
            resolver.resolve(DRIVE_URL)


# ---------------------------------------------------------------- regression (2H) ---


def test_local_directory_still_works(tmp_path):
    cfg = make_config(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (16, 16)).save(img_dir / "a.png")
    mission_file = write_mission(tmp_path / "mission.json", mission_id="LOCAL-1")

    runner = MissionRunner(cfg, provider=FakeProvider(),
                           video_generator=fake_video_gen, progress=lambda m: None)
    summary = runner.run(str(mission_file), str(img_dir), str(cfg.output_directory))
    assert summary.completed == 1 and summary.failed == 0


def test_manifest_input_still_works(tmp_path):
    cfg = make_config(tmp_path)
    img = tmp_path / "img.png"
    Image.new("RGB", (16, 16)).save(img)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"url": f"file://{img}", "filename": "img.png"}]}))
    mission_file = write_mission(tmp_path / "mission.json", mission_id="MANI-1")

    runner = MissionRunner(cfg, provider=FakeProvider(),
                           video_generator=fake_video_gen, progress=lambda m: None)
    with patch("mission_ai.sources.manifest.ManifestResolver.resolve",
               side_effect=lambda _: [img]):
        summary = runner.run(str(mission_file), str(manifest), str(cfg.output_directory))
    assert summary.completed == 1 and summary.failed == 0


def test_single_image_input_still_works(tmp_path):
    cfg = make_config(tmp_path)
    img = tmp_path / "solo.png"
    Image.new("RGB", (16, 16)).save(img)
    mission_file = write_mission(tmp_path / "mission.json", mission_id="SOLO-1")

    runner = MissionRunner(cfg, provider=FakeProvider(),
                           video_generator=fake_video_gen, progress=lambda m: None)
    summary = runner.run(str(mission_file), str(img), str(cfg.output_directory))
    assert summary.completed == 1 and summary.failed == 0


# ---------------------------------------------------------------- helpers -------


def write_mission(path: Path, **extra) -> Path:
    data = {
        "mission_id": "DRV-1",
        "main_message": "drive test",
        "max_hashtags": 3,
    }
    data.update(extra)
    path.write_text(json.dumps(data))
    return path


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        output_directory=tmp_path / "out",
    )


class FakeProvider:
    def analyze_image(self, image_path, mission_context):
        from mission_ai.models import ImageAnalysis
        return ImageAnalysis(
            summary=f"summary of {Path(image_path).name}",
            visible_subjects=["subject"],
            visual_context="context",
            relevant_details=["detail"],
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        from mission_ai.models import PlatformContent
        return PlatformContent(platform=platform, caption=f"{platform}: {image_analysis.summary}", hashtags=[])

    def generate_voiceover_script(self, image_analysis, mission_context):
        return "voiceover script"


def fake_video_gen(image_path, output_path, **kwargs):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"fake video")
    return True
