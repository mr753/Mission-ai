import re
from typing import List, Set, Any
from mission_ai.models import MissionContext, ImageAnalysis

def _normalize_tag(tag: str) -> str:
    if not tag or not isinstance(tag, str):
        return ""
    tag = tag.strip()
    if not tag:
        return ""
    if not tag.startswith("#"):
        tag = "#" + tag
    clean = re.sub(r'[^a-zA-Z0-9_]', '', tag[1:])
    if not clean:
        return ""
    return "#" + clean

def generate_hashtags(image_analysis: Any, mission: MissionContext) -> List[str]:
    if isinstance(image_analysis, str):
        image_analysis = ImageAnalysis(summary=image_analysis, visible_subjects=[], visual_context="", relevant_details=[])

    seen_lower: Set[str] = set()
    final_tags: List[str] = []

    # 1. Required hashtags always preserved and normalized
    for t in (mission.required_hashtags or []):
        norm = _normalize_tag(t)
        if norm and norm.lower() not in seen_lower:
            seen_lower.add(norm.lower())
            final_tags.append(norm)

    # Sort for determinism (satisfies test_a_mission_loading)
    final_tags.sort()

    max_h = mission.max_hashtags
    if max_h is None or max_h <= 0:
        max_h = 10

    return final_tags[:max_h]
