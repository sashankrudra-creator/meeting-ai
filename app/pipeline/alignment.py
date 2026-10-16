"""
Temporal alignment stage (Section 6) — CRITICAL COMPONENT.

Whisper (ASR) and pyannote (diarization) segment the same audio
independently, using different boundary logic (VAD windows vs. speaker
change detection), so their segment boundaries never line up exactly.
This module is the single place responsible for turning:

    ASR segments:          [start, end, text, language]
    Diarization turns:     [speaker, start, end]

into the final, speaker-attributed transcript. It deliberately does NOT
use "nearest speaker by timestamp" as the primary rule — that is exactly
the naive approach the spec calls out as insufficient. The primary rule is
temporal overlap/intersection: for every ASR segment, we sum how many
seconds of *each* candidate speaker's diarization turns fall inside that
segment's [start, end) window, and assign the segment to whichever speaker
overlaps it the most.

Documented policy (Section 6 requires this to be documented, not just
implemented):

1.  Exactly one speaker overlaps the segment -> unambiguous assignment.
2.  Multiple speakers overlap the segment (this happens right around a
    speaker transition, or during genuine overlapping/cross-talk speech):
    the segment is assigned to the speaker with the GREATEST total
    overlap duration ("dominant speaker"). The other overlapping
    speaker(s) are NOT discarded — they are recorded on the segment's
    `overlapping_speakers` field, so a consumer of the transcript can see
    that this segment was ambiguous and which other speaker(s) it
    involved, rather than that ambiguity being silently erased.
    Ties (equal overlap duration) are broken deterministically by
    preferring the speaker whose diarization turn started earliest within
    the segment, so re-running alignment on the same inputs always
    produces the same result.
3.  No diarization turn overlaps the segment at all (this happens when
    ASR's VAD produces a segment inside a small gap between diarization
    turns, e.g. a very short utterance right at a turn boundary): rather
    than dropping the segment or labeling it permanently "unknown", it is
    assigned to the temporally NEAREST diarization turn (smallest gap to
    either side). If there is no diarization data at all, the segment is
    labeled with `settings.alignment_unknown_speaker_label` ("unknown")
    rather than being discarded — every ASR segment always appears
    exactly once in the final transcript, in chronological order.

`settings.alignment_min_overlap_ratio` optionally requires a candidate
speaker's overlap to cover at least that fraction of the segment's
duration before it is even considered a candidate (default 0.0 = any
overlap counts, however small).
"""

from __future__ import annotations

from app.config import Settings
from app.models.schemas import (
    DiarizationResult,
    DiarizationTurn,
    RawTranscriptSegment,
    TranscriptionResult,
    TranscriptSegment,
)
from app.utils.errors import AlignmentError
from app.utils.logging import get_logger

logger = get_logger("pipeline.alignment")


def _overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _nearest_speaker(
    segment: RawTranscriptSegment,
    turns: list[DiarizationTurn],
    unknown_label: str,
) -> str:
    if not turns:
        return unknown_label
    best_speaker = unknown_label
    best_gap = float("inf")
    for turn in turns:
        if segment.end <= turn.start:
            gap = turn.start - segment.end
        elif segment.start >= turn.end:
            gap = segment.start - turn.end
        else:
            gap = 0.0  # defensive: an actual overlap should have been caught earlier
        if gap < best_gap:
            best_gap = gap
            best_speaker = turn.speaker
    return best_speaker


def align_transcript_and_speakers(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
    speaker_name_mapping: dict[str, str],
    settings: Settings,
) -> list[TranscriptSegment]:
    """Combine ASR segments and diarization turns into the final,
    chronologically-ordered, speaker-attributed transcript."""
    try:
        turns = sorted(diarization.turns, key=lambda t: t.start)
        segments = sorted(transcription.segments, key=lambda s: s.start)

        aligned: list[TranscriptSegment] = []
        for idx, segment in enumerate(segments, start=1):
            overlaps: dict[str, float] = {}
            for turn in turns:
                ov = _overlap_duration(segment.start, segment.end, turn.start, turn.end)
                if ov > 0:
                    overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + ov

            if settings.alignment_min_overlap_ratio > 0 and overlaps:
                seg_duration = max(segment.end - segment.start, 1e-9)
                filtered = {
                    spk: dur
                    for spk, dur in overlaps.items()
                    if dur / seg_duration >= settings.alignment_min_overlap_ratio
                }
                # Never let the ratio filter eliminate every candidate outright —
                # that would silently discard a real (if partial) overlap.
                overlaps = filtered or overlaps

            overlapping_ids: list[str] = []
            if overlaps:
                max_overlap = max(overlaps.values())
                tied_candidates = [spk for spk, dur in overlaps.items() if dur == max_overlap]
                if len(tied_candidates) > 1:
                    def _first_turn_start(spk: str) -> float:
                        return min(t.start for t in turns if t.speaker == spk)

                    dominant_speaker = min(tied_candidates, key=_first_turn_start)
                else:
                    dominant_speaker = tied_candidates[0]
                overlapping_ids = sorted(spk for spk in overlaps if spk != dominant_speaker)
            else:
                dominant_speaker = _nearest_speaker(
                    segment, turns, settings.alignment_unknown_speaker_label
                )

            is_unknown = dominant_speaker == settings.alignment_unknown_speaker_label
            speaker_name = (
                "Unknown Speaker" if is_unknown
                else speaker_name_mapping.get(dominant_speaker, dominant_speaker)
            )

            aligned.append(
                TranscriptSegment(
                    segment_id=idx,
                    speaker_id=dominant_speaker,
                    speaker_name=speaker_name,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    language=segment.language,
                    overlapping_speakers=overlapping_ids,
                    low_confidence=segment.low_confidence,
                )
            )

        logger.info(
            "Aligned %d ASR segments against %d diarization turns (%d ambiguous overlaps)",
            len(aligned), len(turns), sum(1 for a in aligned if a.overlapping_speakers),
        )
        return aligned
    except AlignmentError:
        raise
    except Exception as exc:
        raise AlignmentError(f"Temporal alignment failed: {exc}") from exc
