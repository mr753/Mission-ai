import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from mission_ai.sources.manifest import ManifestResolver
from mission_ai.sources.google_drive import GoogleDriveFolderResolver

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
    
    # Mock urllib.request.urlretrieve
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        paths = resolver.resolve(str(manifest))
        assert len(paths) == 1
        assert paths[0].name == "i1.jpg"
        mock_retrieve.assert_called_once()

def test_google_drive_invalid_url():
    resolver = GoogleDriveFolderResolver()
    with pytest.raises(ValueError, match="Not a valid Google Drive folder URL"):
        resolver.resolve("https://example.com/folder")

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
