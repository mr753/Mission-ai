import hashlib
import os
from pathlib import Path
from typing import List, Dict
from mission_ai.models import MissionContext, ImageJob, JobStatus

class ImageJobBuilder:
    def __init__(self, supported_exts: set = None):
        self.supported_exts = supported_exts or {'.jpg', '.jpeg', '.png', '.webp'}

    def _get_file_hash(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def build(self, mission: MissionContext, image_paths: List[Path]) -> List[ImageJob]:
        jobs: List[ImageJob] = []
        seen_hashes: Dict[str, ImageJob] = {}
        
        # Sort deterministically
        sorted_paths = sorted(image_paths)
        
        order = 0
        for path in sorted_paths:
            if not path.exists():
                raise ValueError(f"File does not exist: {path}")
            if path.suffix.lower() not in self.supported_exts:
                continue
                
            file_hash = self._get_file_hash(path)
            
            if file_hash in seen_hashes:
                continue
            
            job_id = f"{mission.mission_id}_{file_hash}"
            job = ImageJob(
                job_id=job_id,
                source_path=str(path),
                mission_context=mission,
                status=JobStatus.PENDING,
                order=order
            )
            
            seen_hashes[file_hash] = job
            jobs.append(job)
            order += 1
            
        return jobs

    def get_output_dir(self, mission_root: Path, job: ImageJob) -> Path:
        # Expected format: 001_{short_hash}
        short_hash = job.job_id.split('_')[1][:8]
        folder_name = f"{job.order + 1:03d}_{short_hash}"
        return mission_root / folder_name
