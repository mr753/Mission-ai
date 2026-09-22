import json
from pathlib import Path
from typing import Dict
from mission_ai.models import JobStatus

class CheckpointManager:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict[str, str] = {}
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                self.state = json.load(f)

    def update(self, image_id: str, status: JobStatus):
        self.state[image_id] = status.value
        with open(self.state_file, "w") as f:
            json.dump(self.state, f)

    def is_completed(self, image_id: str) -> bool:
        return self.state.get(image_id) == JobStatus.COMPLETED.value
