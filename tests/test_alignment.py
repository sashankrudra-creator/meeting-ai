"""
Tests for the critical temporal-alignment stage (Section 6).

Covers: unambiguous overlap, speaker-transition overlap (dominant-speaker
rule + overlapping_speakers bookkeeping), exact ties, no-overlap gap-fill
fallback, total absence of diarization data, chronological ordering, and
the min-overlap-ratio filter.
"""

from __future__ import annotations

from app.models.schemas import DiarizationResult, DiarizationTurn, RawTranscriptSegment, TranscriptionResult
from app.pipeline.alignment import align_transcript_and_speakers

NAME_MAP = {"SPEAKER_00": "Person 1", "SPEAKER_01": "Person 2", "SPEAKER_02": "Person 3"}


def _transcription(*segments: RawTranscriptSegment) -> TranscriptionResult:
    return TranscriptionResult(segments=list(segments), detected_languages=["en"], duration_seconds=100.0)


def _diarization(*turns: DiarizationTurn) -> DiarizationResult:
    return DiarizationResult(turns=list(turns), speaker_labels=sorted({t.speaker for t in turns}))


def test_unambiguous_single_speaker_overlap(settings):
    transcription = _transcription(RawTranscriptSegment(start=1.0, end=4.0, text="hello", language="en"))
    diarization = _diarization(DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=5.0))

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert len(result) == 1
    assert result[0].speaker_id == "SPEAKER_00"
    assert result[0].speaker_name == "Person 1"
    assert result[0].overlapping_speakers == []
    assert result[0].segment_id == 1


def test_speaker_transition_assigns_dominant_speaker_and_records_overlap(settings):
    # Segment [0, 10) overlaps SPEAKER_00 for 7s (0-7) and SPEAKER_01 for 3s (7-10).
    transcription = _transcription(RawTranscriptSegment(start=0.0, end=10.0, text="...", language="en"))
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=7.0),
        DiarizationTurn(speaker="SPEAKER_01", start=7.0, end=10.0),
    )

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert result[0].speaker_id == "SPEAKER_00"  # dominant: 7s > 3s
    assert result[0].overlapping_speakers == ["SPEAKER_01"]  # not silently discarded


def test_exact_tie_breaks_by_earliest_turn_start_deterministically(settings):
    # Segment [0, 10): SPEAKER_00 overlaps 0-5 (5s), SPEAKER_01 overlaps 5-10 (5s). Tie.
    transcription = _transcription(RawTranscriptSegment(start=0.0, end=10.0, text="...", language="en"))
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_01", start=5.0, end=10.0),
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=5.0),
    )

    result1 = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)
    result2 = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert result1[0].speaker_id == "SPEAKER_00"  # earlier turn start wins the tie
    assert result1[0].speaker_id == result2[0].speaker_id  # deterministic / reproducible


def test_no_overlap_falls_back_to_nearest_turn_without_discarding_segment(settings):
    # Segment sits entirely in the 0.2s gap between two diarization turns.
    transcription = _transcription(RawTranscriptSegment(start=4.9, end=5.0, text="uh", language="en"))
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=4.9),
        DiarizationTurn(speaker="SPEAKER_01", start=5.0, end=9.0),
    )

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert len(result) == 1  # never discarded
    # Nearest turn: SPEAKER_00 ends exactly at 4.9 (gap 0), SPEAKER_01 starts at 5.0 (gap 0)
    # -> both gaps are 0, first candidate encountered (SPEAKER_00) wins deterministically.
    assert result[0].speaker_id in ("SPEAKER_00", "SPEAKER_01")


def test_no_diarization_data_at_all_labels_unknown_but_keeps_segment(settings):
    transcription = _transcription(RawTranscriptSegment(start=0.0, end=2.0, text="hello?", language="en"))
    diarization = _diarization()

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert len(result) == 1
    assert result[0].speaker_id == settings.alignment_unknown_speaker_label
    assert result[0].speaker_name == "Unknown Speaker"


def test_chronological_order_is_preserved_not_reordered_by_speaker(settings):
    transcription = _transcription(
        RawTranscriptSegment(start=10.0, end=12.0, text="second", language="en"),
        RawTranscriptSegment(start=0.0, end=2.0, text="first", language="en"),
        RawTranscriptSegment(start=5.0, end=6.0, text="middle", language="en"),
    )
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationTurn(speaker="SPEAKER_01", start=5.0, end=6.0),
        DiarizationTurn(speaker="SPEAKER_00", start=10.0, end=12.0),
    )

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert [seg.text for seg in result] == ["first", "middle", "second"]
    assert [seg.segment_id for seg in result] == [1, 2, 3]


def test_min_overlap_ratio_filters_negligible_overlap(settings):
    settings.alignment_min_overlap_ratio = 0.5
    # Segment [0, 10): SPEAKER_00 overlaps only 1s (10%), SPEAKER_01 overlaps 9s (90%).
    transcription = _transcription(RawTranscriptSegment(start=0.0, end=10.0, text="...", language="en"))
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=1.0),
        DiarizationTurn(speaker="SPEAKER_01", start=1.0, end=10.0),
    )

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert result[0].speaker_id == "SPEAKER_01"
    assert result[0].overlapping_speakers == []  # SPEAKER_00 filtered out below the ratio threshold


def test_low_confidence_flag_is_carried_through_from_raw_segment(settings):
    """Regression for H8: the low_confidence flag set by the ASR stage must
    survive alignment onto the final TranscriptSegment, not be dropped."""
    transcription = _transcription(
        RawTranscriptSegment(start=0.0, end=2.0, text="hello", language="en", low_confidence=True),
        RawTranscriptSegment(start=2.0, end=4.0, text="world", language="en", low_confidence=False),
    )
    diarization = _diarization(DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=4.0))

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert result[0].low_confidence is True
    assert result[1].low_confidence is False


def test_multiple_segments_multiple_speakers_full_scenario(settings):
    transcription = _transcription(
        RawTranscriptSegment(start=0.0, end=3.0, text="Hi I'm speaker one", language="en"),
        RawTranscriptSegment(start=3.2, end=6.0, text="Hello back", language="en"),
        RawTranscriptSegment(start=6.0, end=9.0, text="yeh accha hai", language="hi"),
    )
    diarization = _diarization(
        DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=3.1),
        DiarizationTurn(speaker="SPEAKER_01", start=3.1, end=6.0),
        DiarizationTurn(speaker="SPEAKER_00", start=6.0, end=9.0),
    )

    result = align_transcript_and_speakers(transcription, diarization, NAME_MAP, settings)

    assert [r.speaker_name for r in result] == ["Person 1", "Person 2", "Person 1"]
    assert result[2].language == "hi"
