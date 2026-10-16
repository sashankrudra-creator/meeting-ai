from __future__ import annotations

from app.models.schemas import TranscriptSegment
from app.pipeline.statistics import compute_speaker_statistics


def _seg(seg_id, speaker_id, name, start, end, text="x", overlapping=None):
    return TranscriptSegment(
        segment_id=seg_id, speaker_id=speaker_id, speaker_name=name,
        start=start, end=end, text=text, language="en",
        overlapping_speakers=overlapping or [],
    )


def test_basic_two_speaker_statistics():
    transcript = [
        _seg(1, "SPEAKER_00", "Person 1", 0.0, 10.0),
        _seg(2, "SPEAKER_01", "Person 2", 10.0, 15.0),
        _seg(3, "SPEAKER_00", "Person 1", 15.0, 20.0),
    ]
    stats = compute_speaker_statistics(transcript, duration_seconds=20.0)

    by_id = {s.speaker_id: s for s in stats}
    assert by_id["SPEAKER_00"].speaking_time_seconds == 15.0
    assert by_id["SPEAKER_00"].segment_count == 2
    assert by_id["SPEAKER_00"].average_segment_duration_seconds == 7.5
    assert by_id["SPEAKER_00"].longest_segment_duration_seconds == 10.0
    assert by_id["SPEAKER_00"].speaking_percentage == 75.0

    assert by_id["SPEAKER_01"].speaking_time_seconds == 5.0
    assert by_id["SPEAKER_01"].speaking_percentage == 25.0


def test_statistics_ordered_by_first_appearance():
    transcript = [
        _seg(1, "SPEAKER_01", "Person 2", 0.0, 2.0),
        _seg(2, "SPEAKER_00", "Person 1", 2.0, 4.0),
    ]
    stats = compute_speaker_statistics(transcript, duration_seconds=4.0)
    assert [s.speaker_id for s in stats] == ["SPEAKER_01", "SPEAKER_00"]


def test_overlapping_speech_not_double_counted():
    # Segment covers 0-10s and is attributed to SPEAKER_00 only (per alignment
    # policy), even though SPEAKER_01 also spoke during part of it — recorded
    # via overlapping_speakers, not via a second statistics contribution.
    transcript = [
        _seg(1, "SPEAKER_00", "Person 1", 0.0, 10.0, overlapping=["SPEAKER_01"]),
    ]
    stats = compute_speaker_statistics(transcript, duration_seconds=10.0)
    assert len(stats) == 1  # SPEAKER_01 gets no separate tally for this segment
    assert stats[0].speaking_time_seconds == 10.0
    # The sum of all speaking time never exceeds the meeting duration.
    assert sum(s.speaking_time_seconds for s in stats) <= 10.0


def test_empty_transcript_returns_empty_statistics():
    assert compute_speaker_statistics([], duration_seconds=100.0) == []


def test_zero_duration_meeting_does_not_divide_by_zero():
    transcript = [_seg(1, "SPEAKER_00", "Person 1", 0.0, 0.0)]
    stats = compute_speaker_statistics(transcript, duration_seconds=0.0)
    assert stats[0].speaking_percentage == 0.0


def test_reproducible_from_same_transcript():
    transcript = [
        _seg(1, "SPEAKER_00", "Person 1", 0.0, 5.0),
        _seg(2, "SPEAKER_01", "Person 2", 5.0, 8.0),
    ]
    stats1 = compute_speaker_statistics(transcript, duration_seconds=8.0)
    stats2 = compute_speaker_statistics(transcript, duration_seconds=8.0)
    assert stats1 == stats2
