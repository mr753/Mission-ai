"""Tests for GeminiProvider retry, backoff, and JSON response normalization."""
import json
import pytest
from unittest.mock import MagicMock
from PIL import Image
from mission_ai.providers.gemini import GeminiProvider
from mission_ai.models import MissionContext, ImageAnalysis


@pytest.fixture
def mission_ctx():
    return MissionContext(
        mission_id="M-TEST",
        instructions="Test instructions",
        main_message="Test message",
        key_points=[],
        platforms=["instagram"],
        max_hashtags=3,
        required_hashtags=[],
        image_source_url="dummy"
    )


def test_gemini_retry_on_transient_503_then_success(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()

    mock_client = MagicMock()
    success_response = MagicMock()
    success_response.text = '{"summary": "ok", "visible_subjects": [], "visual_context": "", "relevant_details": []}'

    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("503 Service Unavailable: High demand")
        return success_response

    mock_client.models.generate_content.side_effect = side_effect
    provider._client = mock_client

    monkeypatch.setattr("time.sleep", lambda s: None)

    analysis = provider.analyze_image(str(img), mission_ctx)
    assert analysis.summary == "ok"
    assert call_count == 3


def test_gemini_retry_exhaustion(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")
    provider._client = mock_client

    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="503"):
        provider.analyze_image(str(img), mission_ctx)


def test_gemini_non_transient_error_not_retried(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()
    mock_client = MagicMock()
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid API key 401")

    mock_client.models.generate_content.side_effect = side_effect
    provider._client = mock_client

    with pytest.raises(ValueError, match="401"):
        provider.analyze_image(str(img), mission_ctx)
    assert call_count == 1


def test_gemini_json_markdown_code_fences(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = '```json\n{"summary": "fenced", "visible_subjects": ["a"], "visual_context": "b", "relevant_details": ["c"]}\n```'
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client

    analysis = provider.analyze_image(str(img), mission_ctx)
    assert analysis.summary == "fenced"


def test_gemini_empty_response_raises(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = '   '
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client

    with pytest.raises(ValueError, match="empty"):
        provider.analyze_image(str(img), mission_ctx)


def test_gemini_malformed_json_raises_json_decode_error(monkeypatch, mission_ctx, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    img = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(img)

    provider = GeminiProvider()
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = 'not json at all'
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client

    with pytest.raises(json.JSONDecodeError):
        provider.analyze_image(str(img), mission_ctx)


def test_gemini_caption_and_voiceover_none_response_text(monkeypatch, mission_ctx):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = GeminiProvider()
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = None
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client

    analysis = ImageAnalysis(
        summary="summary",
        visible_subjects=["subject"],
        visual_context="context",
        relevant_details=["detail"]
    )

    caption = provider.generate_caption(analysis, mission_ctx, "instagram")
    assert caption == ""

    script = provider.generate_voiceover_script(analysis, mission_ctx)
    assert script == ""
