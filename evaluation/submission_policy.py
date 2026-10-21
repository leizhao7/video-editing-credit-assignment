"""Public delivery policy shared by the native rollout and trusted grader."""
import math

CONTRACT = 'delivery-duration-10pct-1to3s-v1'


def duration_tolerance(target):
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise RuntimeError('Trusted target duration must be finite and positive')
    return max(1., min(3., float(target) * .10))


def submission_prompt(original):
    # Applied exactly once at the shared native launch boundary (train and val).
    # Frozen bundle bytes/hashes remain intact; this code is integration-pinned.
    return original + '''

DELIVERY FORMAT AND DURATION SCORING RULES
Submit the JSON deliverable specified by the task. S4E also requires a decodable output/preview.mp4 containing video frames.
Missing files, malformed deliverables, references to nonexistent assets, or edits outside an asset's usable time range receive a score of 0.
The total-duration tolerance is 10% of the target duration, with a minimum of 1 second and a maximum of 3 seconds; the tolerance boundaries are inclusive.
For S3, use target_duration_s in the manifest as the target and sum the durations of all retained clips. If no target is provided, no total-duration constraint is imposed.
For S4E, use stretch_dur_s in the manifest as the target, falling back to the sum of the shots' dur_s values if absent. Check the total duration of the final audiovisual deliverable.
This total-duration tolerance supersedes any older total-duration tolerance stated in the task prompt. It does not permit edits outside the usable time range of the source assets.
'''
