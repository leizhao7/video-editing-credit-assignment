"""Frame-paired temporal scoring on an explicitly sampled time grid.

Scores are mean dot products of corresponding descriptor rows. For unit-normalized
descriptors this is mean cosine similarity, not an acceptance probability. Sampling
offsets and uncertainty diagnostics are in samples at the caller's sampling FPS;
they do not identify exact source-video presentation timestamps.
"""
import math

import numpy as np


SCORE_KIND = "paired-frame-mean-cosine-v1"


def peak_interval(scores, band=0.02):
    """Describe the contiguous score band around argmax, without confidence claims.

    The total count of near-peak offsets can include disconnected alternatives;
    it is recorded separately and must not be mistaken for a plateau width.
    """
    scores = np.asarray(scores)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not math.isfinite(band) or band <= 0):
        raise ValueError("peak interval needs finite scores and a positive score band")
    best = int(np.argmax(scores))
    near = scores > scores[best] - band
    left = right = best
    while left and near[left - 1]:
        left -= 1
    while right + 1 < len(scores) and near[right + 1]:
        right += 1
    width = right - left + 1
    count = int(near.sum())
    return {"selected_offset": best, "peak_left_offset": left,
            "peak_right_offset": right, "plateau_frames": width,
            "near_peak_count": count, "disjoint_near_peaks": int(count > width)}


def paired_frame_scores(source, query, sample_offsets=None):
    """Score each valid source start against temporally corresponding query rows.

    ``sample_offsets`` optionally supplies strictly increasing integer offsets
    from the first query sample (for a sparse coarse query). With no offsets,
    query row i pairs with source row start+i, never every row in that window.
    Empty queries or source windows too short to contain a query return no scores.
    """
    source, query = np.asarray(source), np.asarray(query)
    if source.ndim != 2 or query.ndim != 2 or source.shape[1] != query.shape[1]:
        raise ValueError("source and query must be 2-D descriptors of equal width")
    k = len(query)
    offsets = np.arange(k) if sample_offsets is None else np.asarray(sample_offsets)
    if (offsets.ndim != 1 or len(offsets) != k
            or not np.issubdtype(offsets.dtype, np.integer)
            or (k and (offsets[0] != 0 or np.any(np.diff(offsets) <= 0)))):
        raise ValueError("sample offsets must start at 0 and be increasing integers")
    if not k or len(source) <= offsets[-1]:
        return np.empty(0, dtype=np.float64)
    similarity = source @ query.T
    if not np.isfinite(similarity).all():
        raise ValueError("alignment descriptors must produce finite scores")
    n = len(source) - int(offsets[-1])
    scores = np.zeros(n, dtype=np.float64)
    for i, offset in enumerate(offsets):
        scores += similarity[offset:offset + n, i]
    return scores / k


def aligned_source_start(window_start_s, offset_frames, sample_fps,
                         query_start_s, reference_start_s):
    """Estimate the reference shot's source in-point in seconds.

    Pass the actual (nonnegative, rounded) decoder seek origins. If the query
    falls back from its middle to its full shot, query_start_s must change too.
    """
    values = (window_start_s, sample_fps, query_start_s, reference_start_s)
    if (not all(math.isfinite(x) for x in values) or sample_fps <= 0
            or min(window_start_s, query_start_s, reference_start_s) < 0
            or not isinstance(offset_frames, (int, np.integer)) or offset_frames < 0):
        raise ValueError("alignment needs finite nonnegative times, a positive FPS and an integer offset")
    return max(0.0, window_start_s + offset_frames / sample_fps
               - (query_start_s - reference_start_s))


def colour_verdict(score, threshold=None):
    """Require review unless an operator explicitly sets a paired-score threshold.

    The previous 0.482 threshold belonged to an all-pairs window mean and is not
    calibrated for this score. A user-supplied threshold is an explicit policy,
    not a claim that the value has been validated on other films.
    """
    if not math.isfinite(score):
        raise ValueError("colour score must be finite")
    if threshold is None:
        return "REVIEW(colour)"
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("paired colour threshold must be finite and between -1 and 1")
    return "MATCH(colour)" if score >= threshold else "NOT-IN-SOURCE"
