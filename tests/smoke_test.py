import pytest
import os
import json
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from mission_ai.models import MissionContext, JobStatus, ImageAnalysis, ContentPackage, PlatformContent
from mission_ai.hashtag_engine import generate_hashtags
from mission_ai.video_generator import image_to_video
from mission_ai.output_manager import OutputManager
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.caption_engine import generate_platform_content

# Fake Provider for Caption Engine testing
class FakeProvider:
    def analyze_image(self, image_path, mission_context): return ImageAnalysis("sum", ["s"], "ctx", ["d"])
    def generate_caption(self, analysis, mission, platform): return "cap"
    def generate_voiceover_script(self, analysis, mission): return "voiceover script"

def test_smoke_workflow(tmp_path):
    mission = MissionContext(
        mission_id="test-m1", instructions="do X", main_message="msg",
        key_points=["p1"], platforms=["ig"], max_hashtags=2, required_hashtags=["#req"]
    )
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(img_path)
    out_manager = OutputManager(tmp_path / "OUTPUT", mission.mission_id)
    checkpoint = CheckpointManager(tmp_path / "state.json")
    video_path = out_manager.videos_dir / "01.mp4"
    success = image_to_video(str(img_path), str(video_path), duration=2)
    assert success
    assert video_path.exists()
    package = ContentPackage(
        image_id="01", source_path=str(img_path), video_path=str(video_path),
        analysis=ImageAnalysis("summary", ["sub"], "context", ["det"]),
        captions=[PlatformContent("ig", "cap", ["#req"])]
    )
    out_manager.save_package(package)
    checkpoint.update("01", JobStatus.COMPLETED)
    assert checkpoint.is_completed("01")
    assert (out_manager.metadata_dir / "01.json").exists()
    assert (out_manager.captions_dir / "01_caption.txt").exists()

def test_hashtag_deduplication():
    m = MissionContext(mission_id="m1", instructions="do X", main_message="msg",
                       key_points=["p1"], platforms=["ig"], max_hashtags=5,
                       required_hashtags=["#req", "#req"])
    hashtags = generate_hashtags("analysis", m)
    assert len(hashtags) == 1
    assert hashtags == ["#req"]

def test_gemini_sdk_import():
    import google.genai
    assert google.genai is not None

def test_gemini_provider_initialization():
    from mission_ai.providers.gemini import GeminiProvider
    provider = GeminiProvider()
    assert provider.model_name == "gemini-3.8-flash"
    custom_provider = GeminiProvider(model_name="gemini-2.5-flash")
    assert custom_provider.model_name == "gemini-2.5-flash"
    with patch("google.genai.Client") as mock_client_cls:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-test-key"}):
            client = provider._get_client()
            mock_client_cls.assert_called_once_with(api_key="fake-test-key")
            assert client is not None

def test_gemini_model_configuration_in_config_and_factory():
    from mission_ai.config import AppConfig
    from mission_ai.providers import create_provider
    cfg = AppConfig()
    assert cfg.gemini_model == "gemini-3.8-flash"
    provider = create_provider(cfg)
    assert provider.model_name == "gemini-3.8-flash"

    cfg_custom = AppConfig(gemini_model="gemini-3.8-flash")
    provider_custom = create_provider(cfg_custom)
    assert provider_custom.model_name == "gemini-3.8-flash"

def test_gemini_provider_api_key_error():
    from mission_ai.providers.gemini import GeminiProvider
    provider = GeminiProvider()
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        # ensure GEMINI_API_KEY is empty/unset
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
                provider._get_client()

def test_youtube_title_generation():
    m = MissionContext(mission_id="m1", instructions="do X", main_message="My Main Message",
                       key_points=["p1"], platforms=["youtube"], max_hashtags=2)
    analysis = ImageAnalysis("Analysis Summary", ["s"], "ctx", ["d"])
    provider = FakeProvider()
    content = generate_platform_content(analysis, m, provider, "youtube")
    assert content.title is not None
    assert "Generated Title" not in content.title
    assert "My Main Message" in content.title
