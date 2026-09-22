import json
from pathlib import Path
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
            # Simplistic serialization for dataclass
            json.dump(package.__dict__, f, default=lambda x: x.value if hasattr(x, 'value') else str(x))
            
        # Save caption
        cap_path = self.captions_dir / f"{image_id}_caption.txt"
        with open(cap_path, "w") as f:
            for platform_content in package.captions:
                f.write(f"---{platform_content.platform}---\n")
                f.write(f"{platform_content.caption}\n")
                f.write(f"{', '.join(platform_content.hashtags)}\n\n")
