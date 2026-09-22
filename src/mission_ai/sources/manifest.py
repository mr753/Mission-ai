import json
import urllib.request
import os
from pathlib import Path
from typing import List

class ManifestResolver:
    def __init__(self, download_dir: Path = Path("./input")):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.supported_exts = {'.jpg', '.jpeg', '.png', '.webp'}

    def resolve(self, manifest_path: str) -> List[Path]:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(path, "r") as f:
            data = json.load(f)

        images = data.get("images", [])
        if not images:
            raise ValueError("Manifest contains no images")

        resolved_paths = []
        for img in images:
            url = img.get("url")
            filename = img.get("filename")
            
            if not url or not filename:
                continue

            ext = Path(filename).suffix.lower()
            if ext not in self.supported_exts:
                continue

            local_path = self.download_dir / filename
            # Simple deduplication
            if local_path.exists():
                resolved_paths.append(local_path)
                continue

            try:
                urllib.request.urlretrieve(url, local_path)
                resolved_paths.append(local_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download {url}: {e}")

        return resolved_paths
