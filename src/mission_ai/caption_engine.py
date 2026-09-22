from mission_ai.models import PlatformContent, ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider
from mission_ai.hashtag_engine import generate_hashtags

def generate_platform_content(image_analysis: ImageAnalysis, mission_context: MissionContext, 
                              provider: AIProvider, platform: str) -> PlatformContent:
    
    caption_text = provider.generate_caption(image_analysis, mission_context, platform)
    hashtags = generate_hashtags(image_analysis.summary, mission_context)
    
    if platform == "youtube":
        title = f"{mission_context.main_message[:30]} | {image_analysis.summary[:30]}"
        return PlatformContent(
            platform=platform,
            caption=caption_text,
            hashtags=hashtags,
            title=title,
            description=caption_text
        )
    else:
        return PlatformContent(
            platform=platform,
            caption=caption_text,
            hashtags=hashtags
        )
