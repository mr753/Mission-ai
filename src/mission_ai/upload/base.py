from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from mission_ai.models import ContentPackage
from mission_ai.upload.models import UploadResult

class BasePlatformAdapter(ABC):
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the target platform identifier."""
        pass

    @abstractmethod
    def validate_package(self, package: ContentPackage) -> bool:
        """Validate if the content package has all required assets for this platform."""
        pass

    @abstractmethod
    def prepare_payload(self, package: ContentPackage) -> Dict[str, Any]:
        """Prepare platform-specific payload (metadata, caption, hashtags, media path)."""
        pass

    @abstractmethod
    def upload(self, package: ContentPackage, credentials: Optional[Dict[str, Any]] = None) -> UploadResult:
        """Perform stub upload execution for Phase 22C. Never claims a real live post was made."""
        pass
