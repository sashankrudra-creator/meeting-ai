"""
Tests for app.utils.audio and app.pipeline.preprocessing, exercised against
real (synthetic, ffmpeg-generated) media — no model downloads required.
"""

from __future__ import annotations

import pytest

from app.pipeline.preprocessing import preprocess_audio, validate_media_file
from app.utils.audio import extract_and_normalize_audio, probe_media
from app.utils.errors import CorruptMediaError, NoAudioStreamError


def test_probe_valid_wav(settings, valid_wav_file):
    probe = probe_media(valid_wav_file, settings)
    assert probe.has_audio is True
    assert probe.duration_seconds == pytest.approx(2.0, abs=0.2)


def test_probe_video_with_audio(settings, valid_video_with_audio):
    probe = probe_media(valid_video_with_audio, settings)
    assert probe.has_audio is True


def test_probe_video_without_audio_stream(settings, video_without_audio):
    probe = probe_media(video_without_audio, settings)
    assert probe.has_audio is False


def test_probe_corrupt_media_raises(settings, corrupt_media_file):
    with pytest.raises(CorruptMediaError):
        probe_media(corrupt_media_file, settings)


def test_probe_empty_file_raises(settings, empty_file):
    with pytest.raises(CorruptMediaError):
        probe_media(empty_file, settings)


def test_extract_and_normalize_from_wav(settings, valid_wav_file, tmp_path):
    out = tmp_path / "normalized.wav"
    result = extract_and_normalize_audio(valid_wav_file, out, settings)
    assert result.exists()
    probe = probe_media(result, settings)
    assert probe.sample_rate == settings.target_sample_rate
    assert probe.channels == settings.target_channels


def test_extract_and_normalize_from_video(settings, valid_video_with_audio, tmp_path):
    out = tmp_path / "normalized_from_video.wav"
    result = extract_and_normalize_audio(valid_video_with_audio, out, settings)
    assert result.exists()
    probe = probe_media(result, settings)
    assert probe.has_audio is True
    assert probe.sample_rate == settings.target_sample_rate


def test_extract_raises_no_audio_stream(settings, video_without_audio, tmp_path):
    out = tmp_path / "should_not_exist.wav"
    with pytest.raises(NoAudioStreamError):
        extract_and_normalize_audio(video_without_audio, out, settings)


def test_pipeline_preprocess_audio_end_to_end(settings, valid_video_with_audio):
    validate_media_file(valid_video_with_audio, settings)  # should not raise
    result_path = preprocess_audio(valid_video_with_audio, "meeting_test_1", settings)
    assert result_path.exists()
    assert result_path.parent == settings.processed_dir


def test_pipeline_validate_media_file_raises_on_corrupt(settings, corrupt_media_file):
    with pytest.raises(CorruptMediaError):
        validate_media_file(corrupt_media_file, settings)
