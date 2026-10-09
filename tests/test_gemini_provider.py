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
    monkeypatch.setattr(provider, "_search_web_context", lambda topic: "")
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
    assert script == "Topik yang dibahas adalah summary."



def test_voiceover_rejects_english_image_description_and_falls_back(monkeypatch, mission_ctx):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = GeminiProvider()
    monkeypatch.setattr(provider, "_search_web_context", lambda topic: "- Sumber berita: Koperasi desa didorong terhubung dengan produsen lokal. (Sumber: https://example.test)")
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = "An informational poster displaying the headline 'Rantai Pasok MBG Membuka Ruang Produk Lokal Masuk ke Ekosistem Penyediaan Pangan Nasional' with three numbered points."
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client
    analysis = ImageAnalysis(
        summary="Poster berjudul \"Rantai Pasok MBG Membuka Ruang Produk Lokal Masuk ke Ekosistem Penyediaan Pangan Nasional\"",
        visible_subjects=[],
        visual_context="",
        relevant_details=[]
    )
    script = provider.generate_voiceover_script(analysis, mission_ctx)
    assert script.startswith("Topik yang dibahas adalah Rantai Pasok MBG")
    assert "informational poster" not in script.lower()
    assert mock_client.models.generate_content.call_count == 2


def test_voiceover_accepts_natural_indonesian_narration(monkeypatch, mission_ctx):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = GeminiProvider()
    monkeypatch.setattr(provider, "_search_web_context", lambda topic: "- Rantai pasok MBG membuka peluang produk lokal. (Sumber: https://example.test)")
    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = "Rantai pasok MBG menyoroti peluang produk lokal masuk ke ekosistem penyediaan pangan nasional."
    mock_client.models.generate_content.return_value = resp
    provider._client = mock_client
    analysis = ImageAnalysis(
        summary="Poster berjudul \"Rantai Pasok MBG Membuka Ruang Produk Lokal Masuk ke Ekosistem Penyediaan Pangan Nasional\"",
        visible_subjects=[],
        visual_context="",
        relevant_details=[]
    )
    script = provider.generate_voiceover_script(analysis, mission_ctx)
    assert script == resp.text
    assert mock_client.models.generate_content.call_count == 1


def test_voiceover_retries_unsupported_certainty_claim(monkeypatch, mission_ctx):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = GeminiProvider()
    monkeypatch.setattr(provider, "_search_web_context", lambda topic: "- Koperasi desa dan Dapur MBG didorong terhubung dengan UMKM lokal. (Sumber: https://example.test)")
    mock_client = MagicMock()
    bad = MagicMock()
    bad.text = "Langkah ini memastikan pasokan pangan berasal langsung dari produsen lokal."
    good = MagicMock()
    good.text = "Koperasi desa dan Dapur MBG didorong terhubung dengan UMKM lokal."
    mock_client.models.generate_content.side_effect = [bad, good]
    provider._client = mock_client
    analysis = ImageAnalysis(
        summary='Judul "Koperasi desa dan Dapur MBG didorong terhubung dengan UMKM lokal"',
        visible_subjects=[], visual_context="", relevant_details=[]
    )
    script = provider.generate_voiceover_script(analysis, mission_ctx)
    assert script == good.text
    assert mock_client.models.generate_content.call_count == 2


def test_web_research_failure_does_not_crash_pipeline(monkeypatch, mission_ctx):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    provider = GeminiProvider()
    monkeypatch.setattr(provider, "_search_web_context", lambda topic: "")
    mock_client = MagicMock()
    response = MagicMock()
    response.text = "Rantai pasok MBG mengangkat peluang produk lokal masuk ke ekosistem pangan nasional."
    mock_client.models.generate_content.return_value = response
    provider._client = mock_client
    analysis = ImageAnalysis(
        summary='Judul "Rantai pasok MBG membuka ruang produk lokal masuk ke ekosistem pangan nasional"',
        visible_subjects=[], visual_context="", relevant_details=[]
    )
    assert provider.generate_voiceover_script(analysis, mission_ctx) == response.text


def test_google_news_rss_parser_extracts_research_context(monkeypatch):
    import urllib.request

    xml_data = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item>
        <title>Koperasi desa dan UMKM lokal - Contoh Berita</title>
        <link>https://news.google.com/rss/articles/example?oc=5</link>
        <description>&lt;p&gt;Koperasi desa didorong terhubung dengan petani lokal.&lt;/p&gt;</description>
        <source url="https://example.test">Contoh Berita</source>
        <pubDate>Fri, 09 Oct 2026 07:00:00 GMT</pubDate>
      </item>
      <item>
        <title>Program pangan daerah</title>
        <link>https://news.google.com/rss/articles/example2?oc=5</link>
        <description>Informasi kedua.</description>
        <source>Media Kedua</source>
      </item>
    </channel></rss>"""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return xml_data

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    provider = GeminiProvider()
    result = provider._search_web_context("koperasi desa UMKM")

    assert "Koperasi desa dan UMKM lokal - Contoh Berita" in result
    assert "Sumber: Contoh Berita" in result
    assert "Koperasi desa didorong dengan petani lokal" not in result  # parser must preserve exact source text
    assert "Koperasi desa didorong terhubung dengan petani lokal." in result
    assert "Tanggal: Fri, 09 Oct 2026" in result
    assert "https://news.google.com/rss/articles/example?oc=5" in result
    assert "Program pangan daerah" in result


def test_google_news_rss_parser_respects_max_results(monkeypatch):
    import urllib.request

    xml_data = b"""<rss><channel>
      <item><title>Berita satu</title><link>https://example.test/1</link></item>
      <item><title>Berita dua</title><link>https://example.test/2</link></item>
    </channel></rss>"""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return xml_data

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    result = GeminiProvider()._search_web_context("topik", max_results=1)
    assert "Berita satu" in result
    assert "Berita dua" not in result
