from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class UploadJob:
    mission_id: str
    video_path: str
    caption: str
    hashtags: List[str]
    selected_platforms: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    platform_specific_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class UploadResult:
    platform: str
    status: str  # "stub_success", "failed", "not_configured"
    url: Optional[str] = None
    external_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

@dataclass
class BatchUploadResult:
    mission_id: str
    results: List[UploadResult] = field(default_factory=list)
    success_count: int = 0
    simulated_count: int = 0
    failure_count: int = 0
    not_configured_count: int = 0
