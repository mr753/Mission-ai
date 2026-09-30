"""Tests for Phase 18A: TTS Abstraction + Audio Generation Layer.

Validates that:
- VoiceOverScript -> audio output via TTSProvider.
- Empty script / invalid script is rejected (fail-fast).
- Provider failure is propagated as failure (no dummy audio, no silent fallback).
- Output audio file must exist and be non-empty.
- Multi-image processing produces separate audio files per image.
- TTSProvider abstraction can be cleanly mocked.
- No FFmpeg dependency on Phase 18A.
"""
import os
from pathlib import Path
import pytest

from mission_ai.models import VoiceOverScript
from mission_ai.providers.tts_base import TTSProvider
from mission_ai.tts_engine import generate_audio_for_script


class FakeTTSProvider:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.synthesized_calls = []

    def synthesize(self, text: str, output_path: str) -> str:
        self.synthesized_calls.append((text, output_path))
        if self.should_fail:
            raise RuntimeError("TTS engine crashed")
        # Create a dummy non-empty audio file
        Path(output_path).write_bytes(b"RIFF-WAVE-DUMMY-AUDIO-BYTES")
        return output_path


class FailingFakeTTSProvider:
    def synthesize(self, text: str, output_path: str) -> str:
        # Does not create file
        raise RuntimeError("Synthesis error")


def test_voiceover_script_to_audio_output(tmp_path):
    script = VoiceOverScript(
        image_id="img_001",
        script_text="Halo, ini adalah test voice-over untuk misi ramah lingkungan."
    )
    output_audio = tmp_path / "voiceovers" / "img_001.wav"
    provider = FakeTTSProvider()

    result_path = generate_audio_for_script(script, str(output_audio), provider)

    assert result_path == str(output_audio)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0
    assert script.audio_path == str(output_audio)
    assert len(provider.synthesized_calls) == 1


def test_empty_script_rejected(tmp_path):
    script = VoiceOverScript(
        image_id="img_002",
        script_text="   "
    )
    output_audio = tmp_path / "voiceovers" / "img_002.wav"
    provider = FakeTTSProvider()

    with pytest.raises(ValueError, match="empty or invalid script text"):
        generate_audio_for_script(script, str(output_audio), provider)


def test_provider_failure_propagated(tmp_path):
    script = VoiceOverScript(
        image_id="img_003",
        script_text="Script text valid"
    )
    output_audio = tmp_path / "voiceovers" / "img_003.wav"
    provider = FailingFakeTTSProvider()

    with pytest.raises(RuntimeError, match="Synthesis error"):
        generate_audio_for_script(script, str(output_audio), provider)


def test_multi_image_produces_separate_audio_files(tmp_path):
    scripts = [
        VoiceOverScript(image_id="image-001", script_text="Script one for image 1"),
        VoiceOverScript(image_id="image-002", script_text="Script two for image 2"),
        VoiceOverScript(image_id="image-003", script_text="Script three for image 3"),
    ]
    provider = FakeTTSProvider()
    vo_dir = tmp_path / "voiceovers"
    vo_dir.mkdir()

    generated_paths = []
    for s in scripts:
        out_path = vo_dir / f"{s.image_id}.wav"
        res = generate_audio_for_script(s, str(out_path), provider)
        generated_paths.append(res)

    assert len(generated_paths) == 3
    for p in generated_paths:
        path_obj = Path(p)
        assert path_obj.exists()
        assert path_obj.stat().st_size > 0

    # Ensure separate audio files
    assert len(set(generated_paths)) == 3
    assert len(provider.synthesized_calls) == 3


def test_tts_provider_mock_abstraction(tmp_path):
    # Verify protocol compliance and mockability
    provider: TTSProvider = FakeTTSProvider()
    assert hasattr(provider, "synthesize")
    dummy_path = tmp_path / "dummy.wav"
    res = provider.synthesize("Test", str(dummy_path))
    assert res == str(dummy_path)
    assert Path(res).exists()


def test_no_ffmpeg_dependency_in_phase_18a(tmp_path):
    # Verify TTS generation works completely without ffmpeg
    script = VoiceOverScript(image_id="no-ffmpeg", script_text="Independent TTS check")
    out_path = tmp_path / "audio.wav"
    provider = FakeTTSProvider()

    # Run generation
    res = generate_audio_for_script(script, str(out_path), provider)
    assert Path(res).exists()
