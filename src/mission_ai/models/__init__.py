from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class JobStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AUTO_PUBLISHED = "AUTO_PUBLISHED"
    MANUAL_READY = "MANUAL_READY"

@dataclass
class MissionContext:
    mission_id: str
    instructions: str
    main_message: str
    key_points: List[str]
    platforms: List[str]
    max_hashtags: int
    required_hashtags: List[str] = field(default_factory=list)
    image_source_url: Optional[str] = None

@dataclass
class ImageAnalysis:
    summary: str
    visible_subjects: List[str]
    visual_context: str
    relevant_details: List[str]
    confidence: str = "high"

@dataclass
class PlatformContent:
    platform: str
    caption: str
    hashtags: List[str]
    title: Optional[str] = None
    description: Optional[str] = None

@dataclass
class ContentPackage:
    image_id: str
    source_path: str
    analysis: ImageAnalysis
    captions: List[PlatformContent]
    status: JobStatus = JobStatus.PENDING
    video_path: Optional[str] = None
    error: Optional[str] = None
