import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from mission_ai.models import ContentPackage, PlatformContent, ImageAnalysis, JobStatus
from mission_ai.upload.models import UploadJob, UploadResult, BatchUploadResult
from mission_ai.upload.registry import get_platform_adapter, list_supported_platforms

class UploadWorker:
    def __init__(self, credentials_map: Optional[Dict[str, Dict[str, Any]]] = None, output_dir: Optional[Path] = None):
        self.credentials_map = credentials_map or {}
        self.output_dir = Path(output_dir) if output_dir else None

    def process_job(self, job: UploadJob) -> BatchUploadResult:
        """Process an upload job with idempotency: skip already successful uploads, allow retries for failed/not_configured ones, and re-try stub successes if live upload is enabled."""
        if not job or not job.mission_id or not job.video_path or not job.selected_platforms:
            raise ValueError("Invalid or empty upload job: mission_id, video_path, and selected_platforms are required.")

        # Validate video path exists
        video_p = Path(job.video_path)
        if not video_p.exists():
            raise ValueError(f"Video path does not exist: {job.video_path}")

        # Construct ContentPackage from UploadJob
        captions = [
            PlatformContent(
                platform=p,
                caption=job.caption,
                hashtags=job.hashtags,
                title=job.title,
                description=job.description
            )
            for p in job.selected_platforms
        ]

        package = ContentPackage(
            image_id=job.mission_id,
            source_path=job.video_path,
            analysis=ImageAnalysis(summary="Upload Job Content", visible_subjects=[], visual_context="", relevant_details=[]),
            captions=captions,
            status=JobStatus.COMPLETED,
            video_path=job.video_path
        )

        existing_results: Dict[str, Dict[str, Any]] = {}
        out_base = self.output_dir or Path("./output")
        mission_out_dir = out_base / job.mission_id
        report_path = mission_out_dir / "upload_report.json"

        if mission_out_dir.exists() and report_path.exists():
            try:
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
                for r in report_data.get("results", []):
                    if "platform" in r:
                        existing_results[r["platform"]] = r
            except Exception:
                pass

        live_enabled = os.getenv("ENABLE_LIVE_UPLOAD", "false").lower() == "true"

        results: List[UploadResult] = []
        success_count = 0
        simulated_count = 0
        failure_count = 0
        not_configured_count = 0

        for platform in job.selected_platforms:
            prev = existing_results.get(platform)
            should_skip = False
            if prev:
                prev_status = prev.get("status")
                if prev_status == "success":
                    should_skip = True
                elif prev_status in ("stub_success", "simulated"):
                    if not live_enabled:
                        should_skip = True
                    else:
                        # stub_success / simulated is not a real live upload when live upload is requested, so retry
                        should_skip = False
                elif prev_status in ("failed", "not_configured"):
                    should_skip = False

            if should_skip:
                res_ts = prev.get("timestamp") if prev else None
                if not res_ts:
                    res_ts = datetime.now(timezone.utc).isoformat()
                res = UploadResult(
                    platform=platform,
                    status=prev.get("status", "success"),
                    url=prev.get("url"),
                    external_id=prev.get("external_id"),
                    message=prev.get("message", "Skipped: already successfully uploaded (idempotent resume)"),
                    error=prev.get("error"),
                    timestamp=res_ts
                )
                results.append(res)
                if res.status == "success":
                    success_count += 1
                elif res.status in ("stub_success", "simulated"):
                    simulated_count += 1
                elif res.status == "not_configured":
                    not_configured_count += 1
                else:
                    failure_count += 1
            else:
                try:
                    adapter = get_platform_adapter(platform)
                    creds = self.credentials_map.get(platform) or job.platform_specific_data.get(platform)
                    res = adapter.upload(package, credentials=creds)
                    results.append(res)
                    if res.status == "success":
                        success_count += 1
                    elif res.status in ("stub_success", "simulated"):
                        simulated_count += 1
                    elif res.status == "not_configured":
                        not_configured_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    # Isolate failure so other platforms are still attempted
                    err_res = UploadResult(
                        platform=platform,
                        status="failed",
                        error=str(e),
                        message=f"Upload failed for platform {platform} due to exception: {e}"
                    )
                    results.append(err_res)
                    failure_count += 1

        batch_result = BatchUploadResult(
            mission_id=job.mission_id,
            results=results,
            success_count=success_count,
            simulated_count=simulated_count,
            failure_count=failure_count,
            not_configured_count=not_configured_count
        )

        # Save upload report JSON if mission output directory exists (without overwriting checkpoint or metadata)
        try:
            out_base = self.output_dir or Path("./output")
            mission_out_dir = out_base / job.mission_id
            if mission_out_dir.exists():
                report_path = mission_out_dir / "upload_report.json"
                import json
                report_data = {
                    "mission_id": job.mission_id,
                    "results": [
                        {
                            "platform": r.platform,
                            "status": r.status,
                            "url": r.url,
                            "external_id": r.external_id,
                            "message": r.message,
                            "error": r.error,
                            "timestamp": r.timestamp
                        }
                        for r in results
                    ],
                    "success_count": success_count,
                    "simulated_count": simulated_count,
                    "failure_count": failure_count,
                    "not_configured_count": not_configured_count
                }
                report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        except Exception:
            pass

        return batch_result

    def process_package(self, mission_id: str, package: ContentPackage, selected_platforms: List[str]) -> BatchUploadResult:
        """Helper to process a ContentPackage directly as an UploadJob."""
        caption_text = ""
        hashtags = []
        title = None
        description = None

        if package.captions:
            cap_obj = package.captions[0]
            caption_text = cap_obj.caption
            hashtags = cap_obj.hashtags
            title = cap_obj.title
            description = cap_obj.description

        job = UploadJob(
            mission_id=mission_id,
            video_path=package.video_path or package.source_path,
            caption=caption_text,
            hashtags=hashtags,
            selected_platforms=selected_platforms,
            title=title,
            description=description
        )
        return self.process_job(job)
