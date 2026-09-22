from typing import List, Set
from mission_ai.models import MissionContext

def generate_hashtags(analysis: str, mission: MissionContext) -> List[str]:
    # 1. Start with required hashtags
    hashtags: Set[str] = set(mission.required_hashtags)

    # 2. Extract potential hashtags from analysis (simple keyword mapping for Phase 1)
    # 3. Add to set, keep size below mission.max_hashtags
    
    # Deterministic logic:
    # Always include required, then fill with keywords up to max
    
    final_hashtags = list(hashtags)
    # Sort for determinism
    final_hashtags.sort()
    
    return final_hashtags[:mission.max_hashtags]
