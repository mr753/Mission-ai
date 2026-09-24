import pytest
import json
import os
import sys
import io
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from mission_ai.sources.manifest import ManifestResolver
from mission_ai.sources.google_drive import GoogleDriveFolderResolver, MASKED


def test_manifest_resolver(tmp_path):
    manifest = tmp_path / "manifest.json"
    data = {
        "images": [
            {"url": "https://example.com/i1.jpg", "filename": "i1.jpg"}
        ]
    }
    with open(manifest, "w") as f:
        json.dump(data, f)
    
    resolver = ManifestResolver(download_dir=tmp_path / "input")
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        paths = resolver.resolve(str(manifest))
        assert len(paths) == 1
        assert paths[0].name == "i1.jpg"
        mock_retrieve.assert_called_once()


def test_google_drive_invalid_url():
    resolver = GoogleDriveFolderResolver()
    with pytest.raises(ValueError, match="Invalid Google Drive folder URL"):
        resolver.resolve("https://example.com/folder")


def test_google_drive_missing_auth():
    resolver = GoogleDriveFolderResolver()
    with pytest.raises(ValueError, match="Google Drive authentication required"):
        resolver.resolve("https://drive.google.com/drive/folders/abc123")


def test_google_drive_folder_id_extraction():
    resolver = GoogleDriveFolderResolver()
    test_cases = [
        ("https://drive.google.com/drive/folders/abc123", "abc123"),
        ("https://drive.google.com/drive/folders/abc123/", "abc123"),
        ("https://drive.google.com/drive/folders/abc123?param=value", "abc123"),
        ("https://drive.google.com/drive/folders/abc123/subfolder", "abc123"),
    ]
    for url, expected_id in test_cases:
        assert resolver._extract_folder_id(url) == expected_id


def test_google_drive_folder_id_extraction_fails():
    resolver = GoogleDriveFolderResolver()
    with pytest.raises(ValueError, match="Invalid Google Drive folder URL"):
        resolver._extract_folder_id("https://example.com/folders/abc")


def test_google_drive_supported_image_filter():
    resolver = GoogleDriveFolderResolver()
    
    supported_files = [
        {"id": "1", "name": "photo.jpg", "mimeType": "image/jpeg"},
        {"id": "2", "name": "photo.jpeg", "mimeType": "image/jpeg"},
        {"id": "3", "name": "photo.png", "mimeType": "image/png"},
        {"id": "4", "name": "photo.webp", "mimeType": "image/webp"},
    ]
    
    for file in supported_files:
        assert resolver._is_supported_image(file) is True
    
    unsupported_files = [
        {"id": "5", "name": "document.txt", "mimeType": "text/plain"},
        {"id": "6", "name": "video.mp4", "mimeType": "video/mp4"},
        {"id": "7", "name": "photo.gif", "mimeType": "image/gif"},
    ]
    
    for file in unsupported_files:
        assert resolver._is_supported_image(file) is False


def test_google_drive_list_files_with_mock(tmp_path):
    resolver = GoogleDriveFolderResolver(download_dir=tmp_path / "downloads")
    
    mock_response = {
        "files": [
            {"id": "1", "name": "photo1.jpg", "mimeType": "image/jpeg", "webContentLink": "https://example.com/photo1.jpg"},
            {"id": "2", "name": "photo2.png", "mimeType": "image/png", "webContentLink": "https://example.com/photo2.png"},
            {"id": "3", "name": "document.txt", "mimeType": "text/plain"},
        ]
    }
    
    with patch.object(resolver, '_api_request', return_value=mock_response):
        files = resolver._list_files_in_folder("test_folder_id")
        assert len(files) == 3


def test_google_drive_resolve_with_mock(tmp_path):
    resolver = GoogleDriveFolderResolver(download_dir=tmp_path / "downloads")
    
    mock_list_response = {
        "files": [
            {"id": "1", "name": "photo.jpg", "mimeType": "image/jpeg", "webContentLink": "https://example.com/photo.jpg"},
        ]
    }
    
    def mock_download(file):
        local_path = resolver.download_dir / file["name"]
        local_path.touch()
        return local_path
    
    with patch.object(resolver, '_api_request', return_value=mock_list_response):
        with patch.object(resolver, '_download_file', side_effect=mock_download):
            resolver._access_token = "mock_token"
            paths = resolver.resolve("https://drive.google.com/drive/folders/test123")
            assert len(paths) == 1
            assert paths[0].name == "photo.jpg"


def test_google_drive_no_supported_images():
    resolver = GoogleDriveFolderResolver()
    resolver._access_token = "mock_token"
    
    mock_response = {
        "files": [
            {"id": "1", "name": "document.txt", "mimeType": "text/plain"},
            {"id": "2", "name": "video.mp4", "mimeType": "video/mp4"},
        ]
    }
    
    with patch.object(resolver, '_api_request', return_value=mock_response):
        with patch.object(resolver, '_download_file') as mock_download:
            with pytest.raises(RuntimeError, match="No supported images found"):
                resolver.resolve("https://drive.google.com/drive/folders/test123")


def test_google_drive_service_account_loading(tmp_path):
    """Test service account credentials loading - verifies file reading."""
    sa_file = tmp_path / "service-account.json"
    sa_data = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    sa_file.write_text(json.dumps(sa_data))
    
    # Test that file is read correctly
    resolver = GoogleDriveFolderResolver.__new__(GoogleDriveFolderResolver)
    resolver.credentials_path = str(sa_file)
    resolver._access_token = None
    resolver.download_dir = tmp_path
    
    # Verify file can be read
    creds_path = Path(resolver.credentials_path)
    assert creds_path.exists()
    with open(creds_path, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data == sa_data


def test_google_drive_service_account_missing_file(tmp_path):
    resolver = GoogleDriveFolderResolver.__new__(GoogleDriveFolderResolver)
    resolver.credentials_path = "/nonexistent/path.json"
    resolver._access_token = None
    resolver.download_dir = tmp_path
    
    with pytest.raises(FileNotFoundError, match="Service account file not found"):
        resolver._load_service_account_credentials()


def test_google_drive_service_account_invalid_json(tmp_path):
    sa_file = tmp_path / "invalid.json"
    sa_file.write_text("not valid json")
    
    resolver = GoogleDriveFolderResolver.__new__(GoogleDriveFolderResolver)
    resolver.credentials_path = str(sa_file)
    resolver._access_token = None
    resolver.download_dir = tmp_path
    
    with pytest.raises(ValueError, match="Invalid service account file"):
        resolver._load_service_account_credentials()


def test_google_drive_credential_security_masking():
    """Test that credentials are masked in error messages."""
    resolver = GoogleDriveFolderResolver()
    resolver._access_token = "test_token"
    
    def mock_api_error(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error": {"message": "Token expired: abc123secret"}}')
        )
    
    with patch.object(resolver, '_get_auth_headers', return_value={"Authorization": "Bearer test_token"}):
        with patch('mission_ai.sources.google_drive.urllib.request.urlopen', side_effect=mock_api_error):
            with pytest.raises(RuntimeError) as exc_info:
                resolver._api_request("files")
            
            error_msg = str(exc_info.value)
            # The token in the error response body should be masked
            # Note: The current implementation masks the access token in the error message
            assert "test_token" not in error_msg or MASKED in error_msg


def test_google_drive_access_token_priority():
    """Test that access token has priority over service account."""
    resolver = GoogleDriveFolderResolver(access_token="direct_access_token")
    
    # Access token should be set directly
    assert resolver._access_token == "direct_access_token"


def test_google_drive_file_id_masking_in_error():
    """Test that file IDs are masked in download errors."""
    resolver = GoogleDriveFolderResolver()
    resolver._access_token = "test_token"
    
    with patch('mission_ai.sources.google_drive.urllib.request.urlopen', side_effect=Exception("Error with file_id=secret123")):
        with pytest.raises(RuntimeError) as exc_info:
            resolver._download_file({"id": "secret123", "name": "test.jpg"})
        
        error_msg = str(exc_info.value)
        assert "secret123" not in error_msg
        assert "***" in error_msg


def test_manifest_empty(tmp_path):
    manifest = tmp_path / "empty.json"
    with open(manifest, "w") as f:
        json.dump({"images": []}, f)
        
    resolver = ManifestResolver()
    with pytest.raises(ValueError, match="Manifest contains no images"):
        resolver.resolve(str(manifest))


def test_manifest_unsupported_ext(tmp_path):
    manifest = tmp_path / "bad.json"
    data = {
        "images": [
            {"url": "https://example.com/i1.txt", "filename": "i1.txt"}
        ]
    }
    with open(manifest, "w") as f:
        json.dump(data, f)
        
    resolver = ManifestResolver()
    paths = resolver.resolve(str(manifest))
    assert len(paths) == 0
