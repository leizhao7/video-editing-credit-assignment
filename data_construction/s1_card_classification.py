"""Public construction draft: s1_card_classification.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""
try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import copy
import math
import re


POLICY_VERSION = "s1-card-correction-v1"

# Bilingual evidence patterns preserve compatibility with historical annotations.
_EN_CARD = (r"(?:(?:pure|full[- ]screen|standalone)\s+)?"
            r"(?:title|text|release[- ]date|end|closing|studio|logo|credit|credits)\s+card")
_EN_ACTUAL = r"\b(?:is\s+actually|actually\s+is|is\s+in\s+fact|this\s+is)\s+(?:a\s+)?"
_LIVE = re.compile(
    r"(?:实际是|其实是|这是一个|而是)[^。；，：]{0,14}实拍"
    r"|本镜画面本身是实拍"
    r"|不是独立字幕卡，是镜头内"
    r"|是镜头内显示器上的字而非字幕卡"
    r"|" + _EN_ACTUAL + r"live[- ]action\b"
    r"|\b(?:text|lettering)\s+on\s+an?\s+in[- ]shot\s+display\b", re.I)
_TRUE_CARD = re.compile(
    r"(?:实际是|其实是|实际是一张|其实是一张)[^。；，：]{0,8}"
    r"(?:纯字幕卡|纯文字卡|纯字卡|全屏字幕卡|整屏文字卡|整屏片名卡"
    r"|主片名字幕卡|主片名卡|片名卡|字幕卡|尾卡字幕|上映日期卡|出品方标志卡)"
    r"|(?:实际是|其实是)纯黑底白字的标语卡"
    r"|(?:实际是|其实是)[A-Z ]{1,30}厂标卡"
    r"|" + _EN_ACTUAL + _EN_CARD + r"\b", re.I)
_BLACK = re.compile(
    r"(?:实际是|其实是|全程是)(?:纯黑|纯白|全黑|全白)(?:过场|画面|帧|场)"
    r"|" + _EN_ACTUAL + r"(?:pure|solid|all)[- ](?:black|white)\s+(?:frames?|shot|transition)\b"
    r"|\b(?:the\s+)?(?:entire|whole)\s+shot\s+is\s+(?:pure|solid|all)[- ](?:black|white)\b", re.I)
_MIXED = re.compile(
    r"(?:白卡|字幕卡|字卡|卡)[^。]{0,55}(?:中间夹|交替|再切到)[^。]{0,30}实拍"
    r"|(?:白卡|字幕卡|字卡|卡)[^。]{0,30}实拍[^。]{0,30}(?:交替|混合)"
    r"|(?:中间夹|夹着)\s*\d*\s*帧实拍"
    r"|先闪\s*\d*\s*帧[^。]{0,20}白卡再切"
    r"|\b" + _EN_CARD + r"s?\b[^.;\n]{0,70}\b(?:alternat\w*|intercut\w*|mixed|followed\s+by)"
    r"[^.;\n]{0,40}\blive[- ]action\b"
    r"|\blive[- ]action\b[^.;\n]{0,70}\b(?:alternat\w*|intercut\w*|mixed)"
    r"[^.;\n]{0,40}\b" + _EN_CARD + r"s?\b", re.I)
_CARD_DESCRIPTION = re.compile(
    r"^(?:(?:纯黑|纯白|纯色|黑|白)底的?)?(?:全屏|整屏)?"
    r"(?:字幕卡|字卡|文字卡|标语卡|片名卡|主片名卡|上映日期卡|档期卡|厂标卡"
    r"|演职员表|演职员卡|信息卡|尾卡)(?:[：:，、。]|$)"
    r"|^(?:(?:a|the)\s+)?" + _EN_CARD + r"(?:[\s:,.]|$)", re.I)


def _number(value):
    """Finite real scalar, excluding bool (JSON true is not a confidence)."""
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def note_classification(note, description=""):
    """Return a narrow evidence label; mention of a card alone is not evidence."""
    note = note if isinstance(note, str) else ""
    if _MIXED.search(note):
        return "mixed_card_footage"
    if _BLACK.search(note):
        return "featureless"
    # A statement of actual live action takes precedence over explaining why the
    # detector called it a card, or describing subtitles drawn over the live action.
    if _LIVE.search(note):
        return "footage"
    if _TRUE_CARD.search(note):
        return "card"
    # Shot logs also contain direct type declarations with no card_misflag field.
    # Match only an opening declaration, never a card used as a temporal landmark.
    if _CARD_DESCRIPTION.search(description.strip()):
        return "card_description_requires_review"
    return "unresolved" if note.strip() else "no_correction_note"


def verified_source_match(match, confidence, minimum=0.75):
    """Restoring kind=card must not make an unverified p2 guess into an answer.

    MATCH(colour) is deliberately insufficient here, even above its usual threshold.
    A legacy card may never have had a meaningful film match measured at all.
    """
    t0, score = _number(match.get("refined_src_s")), _number(confidence)
    return (match.get("verdict") == "MATCH" and t0 is not None and t0 >= 0
            and score is not None and minimum <= score <= 1.0)


def source_boundary_evidence(match, source_manifest):
    """Validate the declared source shot using half-open [start_s, end_s) intervals.
    Do not snap an out-of-interval timestamp to a neighbouring source shot.
    Use the known reference moment until a rejected match has been re-verified."""
    source_id = match.get("source_clip")
    result = {"source_clip": source_id, "refined_src_s": match.get("refined_src_s"),
              "valid": False, "reason": None}
    if not isinstance(source_id, str) or re.fullmatch(r"s\d{4,}", source_id) is None:
        result["reason"] = "unparseable_source_clip"
        return result
    rows = [row for row in source_manifest if row.get("id") == source_id]
    if len(rows) != 1:
        result["reason"] = "missing_or_duplicate_source_clip"
        return result
    row = rows[0]
    start, end = _number(row.get("start_s")), _number(row.get("end_s"))
    timestamp = _number(match.get("refined_src_s"))
    result.update({"source_start_s": start, "source_end_s": end})
    if start is None or end is None or start < 0 or end <= start:
        result["reason"] = "invalid_source_interval"
    elif timestamp is None:
        result["reason"] = "invalid_refined_timestamp"
    elif timestamp < start:
        result["reason"] = "before_declared_source_start"
        result["boundary_delta_s"] = timestamp - start
    elif timestamp >= end:
        result["reason"] = "at_or_after_declared_source_end"
        result["boundary_delta_s"] = timestamp - end
    else:
        result.update({"valid": True, "reason": "inside_declared_source_shot"})
    return result


def correct_s1_classification(script, raw_inputs, legacy_inputs, p2, slug,
                              luma=None, match_confidence=None, overrides=None,
                              flat_std=8.0, match_minimum=0.75, source_manifest=None,
                              verified_spans=None):
    """Return ``(corrected_inputs, corrected_p2, audit_rows)`` without side effects.

    ``legacy_inputs`` is a copy after the existing shared ``mark_cards`` call.
    Optional manual overrides are keyed by shot ID and require ``classification``
    (footage/keep/golden_only/source/card/featureless/mixed_card_footage) and a nonempty
    ``evidence`` string. ``golden_only`` forces trailer sourcing; ``source`` requires
    a verified source match or raises instead of silently accepting a guessed match.
    ``keep`` means reviewed live-action footage (not keep the erroneous old kind).
    An explicit override wins over textual heuristics and luma, but cannot manufacture
    source matches. A mixed-card override is a review label, not an exclusion command.
    Every note, override and changed decision is retained for independent review.
    Provide raw source_manifest.csv rows to verify each source route against the
    shot ID p2 actually declares. The S1 builder always supplies this input. Pure
    metadata callers may omit it, but receive source_boundary.status=not_checked.
    """
    luma, match_confidence, overrides = luma or {}, match_confidence or {}, overrides or {}
    verified_spans = verified_spans or {}
    raw_by_id = {s["id"]: s for s in raw_inputs["shots"]}
    legacy_by_id = {s["id"]: s for s in legacy_inputs["shots"]}
    script_by_id = {s["id"]: s for s in script.get("shots", [])}
    if len(raw_by_id) != len(raw_inputs["shots"]):
        raise ValueError("Duplicate input shot IDs")
    if len(legacy_by_id) != len(legacy_inputs["shots"]) or set(raw_by_id) != set(legacy_by_id):
        raise ValueError("Legacy classification must contain exactly the raw input shots")
    if set(overrides) - set(raw_by_id):
        raise ValueError("Unknown override shot IDs: " + repr(sorted(set(overrides) - set(raw_by_id))))
    fixed_inputs, fixed_p2 = copy.deepcopy(legacy_inputs), copy.deepcopy(p2)
    audit = []
    allowed = {"footage", "keep", "golden_only", "source", "card", "featureless", "mixed_card_footage"}
    for shot in fixed_inputs["shots"]:
        sid = shot["id"]
        raw_kind, legacy_kind = raw_by_id[sid].get("kind"), shot.get("kind")
        source = script_by_id.get(sid, {})
        note, desc = source.get("card_misflag") or "", source.get("desc") or ""
        label = note_classification(note, desc)
        override = overrides.get(sid)
        if override and override.get("verified_candidate") and sid not in verified_spans:
            raise ValueError("Candidate override must be hash/span validated before classification: " + sid)
        if override is not None:
            if (not isinstance(override, dict) or override.get("classification") not in allowed
                    or not isinstance(override.get("evidence"), str)
                    or not override["evidence"].strip()):
                raise ValueError("Override requires classification and evidence: " + sid)
            label = override["classification"]
            if label == "keep":
                label = "footage"
        measured = luma.get(f"{slug}/{sid}")
        contrast = (_number(measured[1]) if isinstance(measured, (tuple, list))
                    and len(measured) >= 2 else None)
        flat = contrast is not None and contrast < flat_std
        review, reason = False, "legacy_classification_retained"
        if sid in verified_spans:
            if not override or label not in {"footage", "source", "golden_only"}:
                raise ValueError("Verified candidate requires an explicit footage override: " + sid)
            span = verified_spans[sid]
            shot["kind"] = "source" if span["origin"] == "film" else "golden_only"
            reason = "explicit_hash_bound_candidate_span"
        elif flat and override is None:
            # Low grayscale contrast is not evidence that a whole video is blank:
            # color, dark faces and fade endpoints can remain recognizable. Never
            # publish this inherited exclusion without an explicit visual decision.
            reason = "low_contrast_requires_visual_review"
            review = True
        elif label == "mixed_card_footage":
            # A shot-boundary unit can alternate live action with cards. Excluding
            # it on prose alone could discard required footage; retaining the legacy
            # decision is not an endorsement and is explicitly marked for review.
            review, reason = True, "mixed_card_footage_requires_review"
        elif (label == "card_description_requires_review"
              or (label == "card" and override is None and legacy_kind != "card"
                  and re.search(r"实拍|叠|压在|为背景|\blive[- ]action\b|\boverla(?:y|id)\b"
                                r"|\bsuperimposed\b", desc, re.I))):
            review, reason = True, "card_overlay_or_description_requires_review"
        elif label in {"card", "featureless"}:
            shot["kind"] = "card"
            reason = "explicit_" + label
        elif label in {"footage", "golden_only", "source"}:
            if legacy_kind == "card" or label in {"golden_only", "source"}:
                good_match = verified_source_match(
                    p2.get(sid, {}), match_confidence.get(f"{slug}/{sid}"), match_minimum)
                if label == "source" and not good_match:
                    raise ValueError("Explicit source override lacks verified match: " + sid)
                if label == "golden_only":
                    good_match = False
                shot["kind"] = "source" if good_match else "golden_only"
                reason = ("restored_footage_verified_source" if good_match
                          else "restored_footage_trailer_only")
                if not good_match and sid in fixed_p2:
                    fixed_p2[sid]["refined_src_s"] = ""
                    fixed_p2[sid]["verdict"] = "NOMATCH(s1 restored card; source unverified)"
            else:
                reason = "explicit_footage_already_retained"
        elif label == "unresolved":
            review, reason = True, "unresolved_correction_note"
        boundary = {"status": "not_applicable"}
        if sid in verified_spans:
            boundary = {"status": "validated_candidate_span", "valid": True}
        elif shot.get("kind") == "source" and fixed_p2.get(sid, {}).get("refined_src_s"):
            if source_manifest is None:
                boundary = {"status": "not_checked"}
            else:
                boundary = source_boundary_evidence(fixed_p2[sid], source_manifest)
                boundary["status"] = "checked"
                if not boundary["valid"]:
                    shot["kind"] = "golden_only"
                    fixed_p2[sid]["refined_src_s"] = ""
                    fixed_p2[sid]["verdict"] = "NOMATCH(s1 source boundary mismatch)"
                    reason = "source_boundary_mismatch_trailer_only"
        audit.append({
            "policy": POLICY_VERSION, "slug": slug, "shot_id": sid,
            "raw_kind": raw_kind, "legacy_kind": legacy_kind,
            "effective_kind": shot.get("kind"), "note_classification": label,
            "reason": reason, "review_required": review,
            "changed_from_legacy": shot.get("kind") != legacy_kind,
            "card_misflag": note, "description": desc,
            "luma": copy.deepcopy(measured), "override": copy.deepcopy(override),
            "measured_featureless": flat,
            "source_boundary": boundary,
            "verified_candidate_span": copy.deepcopy(verified_spans.get(sid)),
            "source_match": copy.deepcopy(p2.get(sid, {})),
            "match_confidence": match_confidence.get(f"{slug}/{sid}"),
            "source_route": ("excluded" if shot.get("kind") == "card" else
                             "film" if shot.get("kind") == "source"
                             and (sid in verified_spans or fixed_p2.get(sid, {}).get("refined_src_s"))
                             else "trailer_only"),
        })
    return fixed_inputs, fixed_p2, audit
