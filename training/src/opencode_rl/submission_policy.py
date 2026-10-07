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

【交付格式与时长评分规则】
提交任务指定的 JSON 交付物，S4E 还必须交付可解码且包含视频画面的 output/preview.mp4。
缺失文件、格式错误、引用不存在的素材或越过素材可用区间，直接记 0 分。
总时长允许误差：目标时长的 10%，最少 1 秒、最多 3 秒（边界包含在内）。
S3 以 manifest 中 target_duration_s 为目标，计算所有保留片段的总时长；没有该目标时不加总时长限制。
S4E 以 manifest 中 stretch_dur_s（未提供则为 shots 的 dur_s 之和）为目标，检查最终音视频交付总时长。
此总时长容忍度替代旧题面中的总时长误差数值，不代表可以超出原素材的可用时间范围。
'''
