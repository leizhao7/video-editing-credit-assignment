"""Playable-duration metadata, preserving the container and every stream fact.

Not a decoder or grader. Complete AV stream durations exclude data/subtitle
tracks. Missing facts are labelled fallback, never silently invented. A stream
duration and a decoded window length remain different measurements.
"""
from copy import deepcopy
import math

VERSION = 'playable-media-duration-v1'


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def normalize_media_metadata(media):
    if not isinstance(media, dict) or not isinstance(media.get('streams'), list):
        raise ValueError('Expected probed media with stream inventory')
    if 'duration_metadata_version' in media:
        if media['duration_metadata_version'] != VERSION:
            raise ValueError('Unrecognized duration metadata version')
        return deepcopy(media)
    added = ('container_duration_s', 'duration_basis', 'playable_stream_count',
             'playable_duration_complete', 'timeline_origin_verified')
    if any(k in media for k in added):
        raise ValueError('Do not overwrite existing duration provenance')
    container = finite(media.get('duration_s'))
    if container is None or container <= 0:
        raise ValueError('Invalid original container duration')
    out = deepcopy(media)
    av = [s for s in media['streams'] if isinstance(s, dict)
          and s.get('codec_type') in ('video', 'audio')]
    durations = [finite(s.get('duration')) for s in av]
    starts = [finite(s.get('start_time')) for s in av]
    complete = bool(av) and all(v is not None and v > 0 for v in durations)
    known_origin = complete and all(v is not None for v in starts)
    if known_origin:
        effective = max(start+duration for start,duration in zip(starts,durations)) - min(starts)
        basis = 'complete_av_stream_span'
    elif complete:
        effective = max(durations)
        basis = 'longest_av_stream_duration_without_complete_start_times'
    else:
        effective = container
        basis = 'container_fallback_missing_playable_duration'
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError('Invalid computed playable duration')
    out.update(duration_s=effective, container_duration_s=media['duration_s'],
        duration_basis=basis, playable_stream_count=len(av),
        playable_duration_complete=complete, timeline_origin_verified=bool(known_origin),
        duration_metadata_version=VERSION)
    return out


def restore_container_metadata(media):
    if media.get('duration_metadata_version') != VERSION:
        raise ValueError('Expected versioned duration metadata')
    out = deepcopy(media)
    out['duration_s'] = out.pop('container_duration_s')
    for key in ('duration_basis', 'playable_stream_count', 'playable_duration_complete',
                'timeline_origin_verified', 'duration_metadata_version'):
        out.pop(key)
    return out
