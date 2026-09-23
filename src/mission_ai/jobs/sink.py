import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List, Protocol
from datetime import datetime, timezone
from mission_ai.models import MissionContext, JobStatus, ContentPackage
from mission_ai.config import AppConfig

logger = logging.getLogger("mission_ai.sink")

class StateSink(Protocol):
    """Abstraction for mirroring pipeline states to a persistence backend."""

    def sink_mission(self, mission: MissionContext) -> None:
        """Mirror mission configuration."""
        ...

    def sink_job_status(
        self,
        job_id: str,
        mission_id: str,
        source_path: str,
        order: int,
        status: JobStatus,
        last_error: Optional[str] = None
    ) -> None:
        """Mirror job execution state."""
        ...

    def sink_output(self, package: ContentPackage) -> None:
        """Mirror job outputs (analysis, captions, video path)."""
        ...

    def sink_event(
        self,
        mission_id: str,
        event_type: str,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log structured pipeline audit event."""
        ...


class NullStateSink:
    """Default no-op implementation of StateSink."""

    def sink_mission(self, mission: MissionContext) -> None:
        pass

    def sink_job_status(
        self,
        job_id: str,
        mission_id: str,
        source_path: str,
        order: int,
        status: JobStatus,
        last_error: Optional[str] = None
    ) -> None:
        pass

    def sink_output(self, package: ContentPackage) -> None:
        pass

    def sink_event(
        self,
        mission_id: str,
        event_type: str,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        pass


class SupabaseHttpSink:
    """Supabase REST-based implementation of StateSink using stdlib urllib.

    Avoids heavy external dependencies like the Supabase Python SDK.
    Integrates with Row Level Security (RLS) policies by passing the api key as Bearer token.
    """

    def __init__(self, supabase_url: str, supabase_key: str, progress: Optional[Any] = None):
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self._progress = progress or (lambda msg: None)
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }

    def _send(self, table: str, payload: Dict[str, Any]) -> None:
        url = f"{self.supabase_url}/rest/v1/{table}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                pass
        except Exception as e:
            # Failure Isolation: fail gracefully to ensure local pipeline is uninterrupted
            msg = f"StateSink fail for {table}: {e}"
            self._progress(msg)
            logger.error(msg)


    def sink_mission(self, mission: MissionContext) -> None:
        payload = {
            "mission_id": mission.mission_id,
            "instructions": mission.instructions,
            "main_message": mission.main_message,
            "key_points": mission.key_points,
            "platforms": mission.platforms,
            "max_hashtags": mission.max_hashtags,
            "required_hashtags": mission.required_hashtags,
            "image_source_url": mission.image_source_url
        }
        self._send("missions", payload)

    def sink_job_status(
        self,
        job_id: str,
        mission_id: str,
        source_path: str,
        order: int,
        status: JobStatus,
        last_error: Optional[str] = None
    ) -> None:
        payload = {
            "job_id": job_id,
            "mission_id": mission_id,
            "source_path": source_path,
            "status": status.value,
            "job_order": order,
            "last_error": last_error,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if status == JobStatus.PROCESSING:
            payload["processing_started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            payload["completed_at"] = datetime.now(timezone.utc).isoformat()
            
        self._send("mission_jobs", payload)

    def sink_output(self, package: ContentPackage) -> None:
        payload = {
            "job_id": package.image_id,
            "analysis": {
                "summary": package.analysis.summary,
                "visible_subjects": package.analysis.visible_subjects,
                "visual_context": package.analysis.visual_context,
                "relevant_details": package.analysis.relevant_details,
                "confidence": package.analysis.confidence
            } if package.analysis else None,
            "captions": [
                {
                    "platform": c.platform,
                    "caption": c.caption,
                    "hashtags": c.hashtags,
                    "title": c.title,
                    "description": c.description
                } for c in package.captions
            ],
            "video_path": package.video_path,
            "status": package.status.value,
            "error": package.error
        }
        self._send("job_outputs", payload)

    def sink_event(
        self,
        mission_id: str,
        event_type: str,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        payload_data = {
            "mission_id": mission_id,
            "job_id": job_id,
            "event_type": event_type,
            "status": status if status else "",
            "payload": payload if payload else {}
        }
        self._send("pipeline_events", payload_data)


def create_state_sink(config: AppConfig, progress: Optional[Any] = None) -> StateSink:
    """Factory to build StateSink based on AppConfig."""
    if config.enable_supabase_sink and config.supabase_url and config.supabase_key:
        return SupabaseHttpSink(config.supabase_url, config.supabase_key, progress)
    return NullStateSink()
