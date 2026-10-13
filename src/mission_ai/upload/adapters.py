from pathlib import Path
from typing import Dict, Any, Optional
from mission_ai.models import ContentPackage
from mission_ai.upload.base import BasePlatformAdapter
from mission_ai.upload.models import UploadResult
from mission_ai.upload.registry import register_adapter

class YouTubeShortsAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "youtube"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "video_path": package.video_path,
            "title": cap.title if cap and cap.title else "Mission Short",
            "description": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
            "tags": cap.hashtags if cap else [],
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for YouTube Shorts upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: YouTube Shorts API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class InstagramAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "instagram"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "caption": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Instagram upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Instagram API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class FacebookAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "facebook"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "video_path": package.video_path,
            "description": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Facebook upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Facebook API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class XAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "x"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        text = f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}"
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "text": text[:280],
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for X upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: X API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


class ThreadsAdapter(BasePlatformAdapter):
    @property
    def platform_name(self) -> str:
        return "x" if False else "threads"

    def validate_package(self, package: ContentPackage) -> bool:
        return bool(package.video_path and Path(package.video_path).exists() and package.captions)

    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        cap = next((c for c in package.captions if c.platform == self.platform_name), package.captions[0] if package.captions else None)
        return {
            "platform": self.platform_name,
            "media_path": package.video_path,
            "text": f"{cap.caption if cap else ''}\n\n{' '.join(cap.hashtags if cap else [])}",
        }

    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        if not self.validate_package(package):
            return UploadResult(
                platform=self.platform_name,
                status="failed",
                error="Invalid content package for Threads upload"
            )
        return UploadResult(
            platform=self.platform_name,
            status="stub_success",
            message="Stub upload executed: Threads API/OAuth not configured yet. No real post made.",
            url=None,
            external_id=None
        )


# Register adapters
register_adapter("youtube", YouTubeShortsAdapter)
register_adapter("youtube_shorts", YouTubeShortsAdapter)
register_adapter("instagram", InstagramAdapter)
register_adapter("facebook", FacebookAdapter)
register_adapter("x", XAdapter)
register_adapter("twitter", XAdapter)
register_adapter("threads", ThreadsAdapter)
