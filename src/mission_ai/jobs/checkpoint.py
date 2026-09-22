import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from mission_ai.models import JobStatus

class CheckpointManager:
    """File-based checkpoint state keyed by image/job id.

    Writes are atomic (temp file + os.replace) so an interrupted run cannot
    corrupt the state file. A corrupt existing state file is recovered from
    backup (.bak) or reset instead of crashing the whole run.
    """

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.state: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.state = {str(k): str(v) for k, v in data.items()}
            return
        except (json.JSONDecodeError, OSError):
            # Try the backup before giving up.
            backup = self.state_file.with_suffix(self.state_file.suffix + ".bak")
            if backup.exists():
                try:
                    with open(backup, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self.state = {str(k): str(v) for k, v in data.items()}
                        return
                except (json.JSONDecodeError, OSError):
                    pass
            # Reset to empty state so processing can continue.
            self.state = {}

    def update(self, image_id: str, status: JobStatus) -> None:
        self.state[image_id] = status.value
        self._write()

    def _write(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        backup = self.state_file.with_suffix(self.state_file.suffix + ".bak")
        # Keep the previous good state as backup.
        if self.state_file.exists():
            try:
                os.replace(self.state_file, backup)
            except OSError:
                pass
        # Atomic write: temp file in the same directory, then replace.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.state_file.parent), prefix=self.state_file.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp_path, self.state_file)
        except OSError:
            # Best effort cleanup; previous state survives in the backup.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def is_completed(self, image_id: str) -> bool:
        return self.state.get(image_id) == JobStatus.COMPLETED.value

    def status_of(self, image_id: str) -> JobStatus:
        """Return the recorded status (PENDING if unknown)."""
        value = self.state.get(image_id)
        try:
            return JobStatus(value)
        except ValueError:
            return JobStatus.PENDING

    def failed_ids(self) -> List[str]:
        """Ids whose last recorded status is FAILED (retryable)."""
        return [k for k, v in self.state.items() if v == JobStatus.FAILED.value]
