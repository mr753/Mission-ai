import json
import dataclasses
from pathlib import Path
from typing import Any, Optional
from mission_ai.models import ContentPackage, JobStatus

class OutputManager:
    def __init__(self, base_output_dir: Path, mission_id: str):
        self.base_dir = base_output_dir / mission_id
        self.videos_dir = self.base_dir / "videos"
        self.captions_dir = self.base_dir / "captions"
        self.metadata_dir = self.base_dir / "metadata"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.captions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_package(self, package: ContentPackage):
        image_id = package.image_id

        # Save metadata
        meta_path = self.metadata_dir / f"{image_id}.json"
        with open(meta_path, "w") as f:
            json.dump(self._to_jsonable(package), f, indent=2)

        # Save caption
        cap_path = self.captions_dir / f"{image_id}_caption.txt"
        with open(cap_path, "w") as f:
            for platform_content in package.captions:
                f.write(f"---{platform_content.platform}---\n")
                f.write(f"{platform_content.caption}\n")
                f.write(f"{', '.join(platform_content.hashtags)}\n\n")

    def save_error(self, image_id: str, source_path: str, error: str):
        """Persist a minimal metadata record for a failed job so resume and
        reporting have full visibility. Keeps the job retryable."""
        package = ContentPackage(
            image_id=image_id,
            source_path=str(source_path),
            analysis=None,
            captions=[],
            status=JobStatus.FAILED,
            error=error,
        )
        meta_path = self.metadata_dir / f"{image_id}.json"
        with open(meta_path, "w") as f:
            json.dump(self._to_jsonable(package), f, indent=2)

    def _to_jsonable(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: self._to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, JobStatus):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, list):
            return [self._to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        return obj
