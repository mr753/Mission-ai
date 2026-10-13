"""Comprehensive tests for Edge TTS provider, voice configuration, MP3-to-WAV conversion, fallback, FFmpeg availability, runner integration, and video pipeline compatibility.
"""
import os
import asyncio
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from mission_ai.config import AppConfig
from mission_ai.models import VoiceOverScript, MissionContext, ImageAnalysis
from mission_ai.providers.edge_tts_provider import EdgeTTSProvider
from mission_ai.providers.pyttsx3 import PyTTSX3Provider
from mission_ai.tts_engine import generate_audio_for_script
from mission_ai.runner import MissionRunner
from mission_ai.video_generator import image_to_video


def test_edge_tts_voice_selection_and_defaults():
    provider = EdgeTTSProvider()
    assert provider.voice_id == "id-ID-GadisNeural"
    assert provider.rate == "-5%"
    assert provider.volume == "+0%"
    assert "id-ID-ArdiNeural" in EdgeTTSProvider.AVAILABLE_VOICES
    assert "id-ID-GadisNeural" in EdgeTTSProvider.AVAILABLE_VOICES

    # Custom valid voice
    provider_custom = EdgeTTSProvider(voice_id="id-ID-ArdiNeural", rate="-10%", volume="+5%")
    assert provider_custom.voice_id == "id-ID-ArdiNeural"
    assert provider_custom.rate == "-10%"
    assert provider_custom.volume == "+5%"

    # Invalid voice falls back deterministically to default
    provider_invalid = EdgeTTSProvider(voice_id="unknown-voice")
    assert provider_invalid.voice_id == "id-ID-GadisNeural"


def test_empty_script_rejected(tmp_path):
    provider = EdgeTTSProvider()
    out_file = tmp_path / "empty.wav"
    with pytest.raises(ValueError, match="Cannot synthesize empty text script"):
        provider.synthesize("   ", str(out_file))


def test_edge_tts_success_mocked(tmp_path):
    """Test successful Edge TTS synthesis when mocked (writing mock audio and converting/copying)."""
    provider = EdgeTTSProvider()
    out_file = tmp_path / "success.wav"

    async def mock_save(target_path):
        Path(target_path).write_bytes(b"MOCK-MP3-AUDIO-DATA")

    with patch("edge_tts.Communicate") as mock_comm_cls:
        mock_comm = mock_comm_cls.return_value
        mock_comm.save = AsyncMock(side_effect=mock_save)

        # Mock ffmpeg convert or test when ffmpeg is mocked/available
        with patch.object(provider, "_convert_mp3_to_wav", return_value=True) as mock_conv:
            # Also create the output file so size check passes
            def side_effect_conv(src, dst):
                Path(dst).write_bytes(b"RIFF-WAVE-MOCK")
                return True
            mock_conv.side_effect = side_effect_conv

            res = provider.synthesize("Halo dunia", str(out_file))
            assert res == str(out_file)
            assert Path(res).exists()
            assert Path(res).stat().st_size > 0


def test_edge_tts_connection_failure_and_fallback(tmp_path):
    provider = EdgeTTSProvider()
    out_file = tmp_path / "fallback.wav"

    with patch("edge_tts.Communicate", side_effect=Exception("ClientConnectorError: Cannot connect")):
        res = provider.synthesize("Test fallback on connection error", str(out_file))

    assert res == str(out_file)
    assert Path(res).exists()
    assert Path(res).stat().st_size > 0


def test_edge_tts_empty_or_corrupt_output_fallback(tmp_path):
    provider = EdgeTTSProvider()
    out_file = tmp_path / "corrupt.wav"

    # Mock communicate to write 0 bytes (empty output)
    async def mock_save_empty(target_path):
        Path(target_path).write_bytes(b"")

    with patch("edge_tts.Communicate") as mock_comm_cls:
        mock_comm = mock_comm_cls.return_value
        mock_comm.save = AsyncMock(side_effect=mock_save_empty)

        # Should trigger exception -> fallback to pyttsx3
        res = provider.synthesize("Test empty output fallback", str(out_file))
        assert res == str(out_file)
        assert Path(res).exists()


def test_ffmpeg_availability_and_conversion_handling(tmp_path):
    provider = EdgeTTSProvider()
    mp3_file = tmp_path / "test.mp3"
    mp3_file.write_bytes(b"MP3DATA")
    wav_file = tmp_path / "test.wav"

    # Test when ffmpeg is not found
    with patch("shutil.which", return_value=None):
        assert provider._convert_mp3_to_wav(str(mp3_file), str(wav_file)) is False


def test_app_config_and_runner_integration(tmp_path):
    config = AppConfig(tts_provider="edge", tts_voice="id-ID-GadisNeural")
    assert config.tts_provider == "edge"
    assert config.tts_voice == "id-ID-GadisNeural"

    class MockAIProvider:
        def analyze_image(self, image_path, mission_context):
            return ImageAnalysis(summary="summary", visible_subjects=["obj"], visual_context="ctx", relevant_details=["detail"], confidence="high")
        def generate_caption(self, image_analysis, mission_context, platform):
            return "caption"
        def generate_voiceover_script(self, image_analysis, mission_context):
            return "Voice-over script for test"

    img = tmp_path / "img.png"
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (32, 32), color="green").save(img)

    runner = MissionRunner(
        config,
        provider=MockAIProvider(),
        video_generator=lambda img_p, out_p, **kw: Path(out_p).write_bytes(b"MP4") or True,
        progress=lambda m: None
    )

    mission_json = tmp_path / "mission.json"
    mission_json.write_text('{"mission_id": "M-1", "main_message": "Test", "instructions": "Inst", "platforms": ["instagram"], "max_hashtags": 2}')

    # Run mission runner with Edge TTS provider (mocking edge_tts to fail and fallback cleanly)
    with patch("edge_tts.Communicate", side_effect=Exception("Offline")):
        summary = runner.run(str(mission_json), str(img), str(tmp_path / "out"))

    assert summary.completed == 1
    voiceovers_dir = tmp_path / "out" / "M-1" / "voiceovers"
    wav_files = list(voiceovers_dir.glob("*.wav"))
    assert len(wav_files) == 1
    assert wav_files[0].exists()


def test_audio_usable_in_video_pipeline(tmp_path):
    """Verify generated audio can be passed to image_to_video pipeline."""
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"RIFF-WAVE-DUMMY")

    img_path = tmp_path / "image.png"
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (32, 32), color="green").save(img_path)

    out_mp4 = tmp_path / "video.mp4"

    def fake_ffmpeg_run(cmd, **kw):
        Path(out_mp4).write_bytes(b"MP4-DATA")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_ffmpeg_run):
        success = image_to_video(
            image_path=str(img_path),
            output_path=str(out_mp4),
            duration=5,
            audio_path=str(wav_path)
        )
        assert success is True
        assert out_mp4.exists()
        assert out_mp4.stat().st_size > 0


def test_edge_tts_timeout_triggers_fallback(tmp_path):
    provider = EdgeTTSProvider()
    out_file = tmp_path / "timeout.wav"

    with patch("edge_tts.Communicate") as mock_comm_cls:
        mock_comm = mock_comm_cls.return_value
        mock_comm.save = AsyncMock(side_effect=asyncio.TimeoutError("Timed out"))

        res = provider.synthesize("Timeout test", str(out_file))

    assert res == str(out_file)
    assert Path(res).exists()
    assert Path(res).stat().st_size > 0


def test_edge_tts_conversion_failure_triggers_fallback(tmp_path):
    provider = EdgeTTSProvider()
    out_file = tmp_path / "conv_fail.wav"

    with patch("edge_tts.Communicate") as mock_comm_cls:
        mock_comm = mock_comm_cls.return_value
        mock_comm.save = AsyncMock(return_value=None)

        with patch.object(provider, "_convert_mp3_to_wav", return_value=False):
            res = provider.synthesize("Conversion failure test", str(out_file))

    assert res == str(out_file)
    assert Path(res).exists()
    assert Path(res).stat().st_size > 0


def test_edge_tts_telemetry_logging_success_and_fallback(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    provider = EdgeTTSProvider()
    out_file = tmp_path / "log_test.wav"

    # Test success path logging
    async def mock_save(target_path):
        Path(target_path).write_bytes(b"MP3DATA")

    with patch("edge_tts.Communicate") as mock_comm_cls:
        mock_comm = mock_comm_cls.return_value
        mock_comm.save = AsyncMock(side_effect=mock_save)

        def fake_subprocess_run(*args, **kwargs):
            Path(out_file).write_bytes(b"WAVDATA")
            res = MagicMock()
            res.returncode = 0
            res.stdout = ""
            res.stderr = ""
            return res

        with patch("subprocess.run", side_effect=fake_subprocess_run), patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            provider.synthesize("Log success test", str(out_file))

    assert "Selected TTS provider: EdgeTTSProvider" in caplog.text
    assert "Starting Edge TTS synthesis request" in caplog.text
    assert "Edge TTS successfully generated MP3 audio output" in caplog.text
    assert "Successfully converted MP3 to WAV" in caplog.text
    assert "EdgeTTSProvider successfully generated final WAV audio" in caplog.text


def test_edge_tts_fallback_telemetry_logging(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    provider = EdgeTTSProvider()
    out_file = tmp_path / "log_fallback.wav"

    with patch("edge_tts.Communicate", side_effect=Exception("Network error")):
        provider.synthesize("Log fallback test", str(out_file))

    assert "Edge TTS failed" in caplog.text
    assert "Edge TTS did NOT succeed" in caplog.text
    assert "Switching to fallback provider: PyTTSX3Provider" in caplog.text
    assert "Final audio successfully generated using fallback provider: PyTTSX3Provider" in caplog.text
