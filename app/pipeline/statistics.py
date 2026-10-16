"""
Speaker statistics stage (Section 9).

Computed entirely from the already-aligned transcript (plus the meeting's
total duration), so these numbers are reproducible from stored data alone
without re-running diarization or ASR — feed the same transcript in twice
and you get identical statistics out.

Overlapping speech and double-counting (documented per spec requirement):
The temporal-alignment stage (Section 6 / app/pipeline/alignment.py)
already resolves every ASR segment to exactly ONE dominant speaker_id,
even when multiple speakers' diarization turns overlapped that segment
(the other speaker(s) are recorded on `overlapping_speakers` but are not
the segment's attributed speaker). Because of that, when this module sums
segment durations per speaker, each second of audio is attributed to at
most one speaker's speaking-time tally — overlapping/cross-talk speech is
never counted twice. The trade-off, stated plainly: during genuine
simultaneous speech, the non-dominant speaker's contribution to that
interval is not separately added to their own speaking time; it remains
visible only via `overlapping_speakers` on the transcript segment. This is
a deliberate simplification (attributing a truly overlapping interval to
two speakers at once would double-count the meeting's total duration),
not an oversight.

`speaking_percentage` is each speaker's speaking time as a fraction of the
whole meeting's `duration_seconds` (not of the sum of all speakers'
speaking time), matching the spec's worked example in Section 9.
"""

from __future__ import annotations

from app.models.schemas import SpeakerStats, TranscriptSegment
from app.utils.errors import StatisticsError
from app.utils.logging import get_logger

logger = get_logger("pipeline.statistics")


def compute_speaker_statistics(
    transcript: list[TranscriptSegment],
    duration_seconds: float,
) -> list[SpeakerStats]:
    try:
        per_speaker: dict[str, dict] = {}
        first_segment_index: dict[str, int] = {}

        for order, segment in enumerate(transcript):
            seg_duration = max(0.0, segment.end - segment.start)
            bucket = per_speaker.setdefault(
                segment.speaker_id,
                {"name": segment.speaker_name, "durations": []},
            )
            bucket["durations"].append(seg_duration)
            first_segment_index.setdefault(segment.speaker_id, order)

        stats: list[SpeakerStats] = []
        for speaker_id, bucket in per_speaker.items():
            durations: list[float] = bucket["durations"]
            total_time = sum(durations)
            segment_count = len(durations)
            average = total_time / segment_count if segment_count else 0.0
            longest = max(durations) if durations else 0.0
            percentage = (total_time / duration_seconds * 100.0) if duration_seconds > 0 else 0.0

            stats.append(
                SpeakerStats(
                    speaker_id=speaker_id,
                    name=bucket["name"],
                    speaking_time_seconds=round(total_time, 3),
                    speaking_percentage=round(percentage, 2),
                    segment_count=segment_count,
                    average_segment_duration_seconds=round(average, 3),
                    longest_segment_duration_seconds=round(longest, 3),
                )
            )

        # Order by first appearance in the transcript — reproducible, and
        # matches the natural "Person 1 spoke first" expectation.
        stats.sort(key=lambda s: first_segment_index[s.speaker_id])

        total_pct = sum(s.speaking_percentage for s in stats)
        logger.info(
            "Computed statistics for %d speaker(s); sum of speaking_percentage=%.1f%% "
            "(may be <100%% due to silence/gaps, or reflect only dominant-speaker time "
            "during overlapping speech)",
            len(stats), total_pct,
        )
        return stats
    except Exception as exc:
        raise StatisticsError(f"Failed to compute speaker statistics: {exc}") from exc
