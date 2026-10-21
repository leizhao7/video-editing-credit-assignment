"""Bounded S4 contracts; structural validation does not certify visual judgments."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

CUT_BATCH_SIZE = 12
INPUT_FILES = {"units": "s3_units.json", "cut_list": "cut_list.json",
               "audio": "audio_measured.json", "picture": "picture_measured.json"}


class VisualReadCoverage:
    """Track successful native Claude image results, never their image/text data."""

    def __init__(self, task_dir, paths):
        self.task = Path(task_dir).resolve()
        self.required = set()
        for path in paths:
            resolved = self._path(path)
            if resolved is None or resolved in self.required:
                raise ValueError("Every S4 cut needs a distinct strip path inside the task directory")
            self.required.add(resolved)
        self.pending, self.seen = {}, set()

    def _path(self, value):
        if not isinstance(value, str) or not value:
            return None
        path = Path(value).expanduser()
        resolved = (path if path.is_absolute() else self.task / path).resolve()
        return resolved if resolved.is_relative_to(self.task) else None

    def feed(self, line):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(event, dict) or not isinstance(event.get("message"), dict):
            return
        content = event["message"].get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if event.get("type") == "assistant" and block.get("type") == "tool_use" and block.get("name") == "Read":
                arguments = block.get("input")
                identifier = block.get("id")
                if isinstance(identifier, str) and isinstance(arguments, dict):
                    path = self._path(arguments.get("file_path"))
                    if path in self.required:
                        self.pending[identifier] = path
            if event.get("type") == "user" and block.get("type") == "tool_result":
                identifier = block.get("tool_use_id")
                if not isinstance(identifier, str):
                    continue
                path = self.pending.pop(identifier, None)
                parts = block.get("content")
                if path is None or block.get("is_error") is True or not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    source = part.get("source")
                    if (isinstance(source, dict) and source.get("type") == "base64"
                            and isinstance(source.get("data"), str) and source["data"]
                            and isinstance(source.get("media_type"), str)
                            and source["media_type"].startswith("image/")):
                        self.seen.add(path)
                        break

    def failures(self):
        missing = self.required - self.seen
        return ([f"S4 visual access incomplete: {len(missing)}/{len(self.required)} cut strips lack "
                 "a matched successful Read image result"] if missing else [])


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_inputs(inputs, slug=None):
    """Validate the immutable identities used to join measured cut information."""
    if not isinstance(inputs, dict) or any(not isinstance(inputs.get(k), dict) for k in INPUT_FILES):
        return ["S4 requires units, cut_list, audio and picture JSON objects"]
    units = inputs["units"].get("units")
    cuts = inputs["cut_list"].get("cuts")
    if (not isinstance(units, list) or not units or not isinstance(cuts, list)
            or any(not isinstance(v, dict) for v in units + cuts)):
        return ["S4 units and cuts must be arrays of objects; units cannot be empty"]
    unit_ids = [u.get("id") for u in units]
    if any(not _text(i) for i in unit_ids) or len(set(unit_ids)) != len(unit_ids):
        return ["S4 input unit IDs must be nonempty unique strings"]
    indices = [c.get("idx") for c in cuts]
    if (any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in indices)
            or len(set(indices)) != len(indices) or indices != sorted(indices)):
        return ["S4 input cut indices must be unique positive integers in order"]
    errors = []
    if slug is not None:
        for key in ("units", "cut_list"):
            if inputs[key].get("film") != slug:
                errors.append(f"S4 {key} film must match the selected slug")
    for cut in cuts:
        if any(not _text(cut.get(k)) or cut[k] not in unit_ids for k in ("after", "before")):
            errors.append(f"Input cut {cut['idx']} has an unknown adjacent unit")
    for key in ("audio", "picture"):
        measured = inputs[key].get("cuts")
        if (not isinstance(measured, list) or any(not isinstance(c, dict) for c in measured)
                or any(isinstance(c.get("idx"), bool) or not isinstance(c.get("idx"), int) for c in measured)
                or [c.get("idx") for c in measured] != indices):
            errors.append(f"S4 {key} indices must exactly match cut_list once, in order")
    return errors


def validate_document(inputs, document, *, prefix=False, diversity=True):
    """Check exact cut coverage, measured continuity and finite transition values."""
    errors = validate_inputs(inputs)
    if errors:
        return errors
    if not isinstance(document, dict) or document.get("stage") != "S4":
        return ["S4 document must have stage: S4"]
    cuts = document.get("cuts")
    if not isinstance(cuts, list) or any(not isinstance(c, dict) for c in cuts):
        return ["S4 document cuts must be an array of objects"]
    expected = inputs["cut_list"]["cuts"]
    if prefix:
        if len(cuts) > len(expected):
            errors.append("S4 prefix contains more cuts than the input")
        expected = expected[:len(cuts)]
    identity = lambda c: (c.get("idx"), c.get("after"), c.get("before"))
    if ([identity(c) for c in cuts] != [identity(c) for c in expected]
            or any(isinstance(c.get("idx"), bool) or not isinstance(c.get("idx"), int) for c in cuts)):
        errors.append("S4 idx/after/before must match every expected cut once, in order")
    measured = {c["idx"]: c for c in inputs["picture"]["cuts"]}
    techniques = set()
    for cut in cuts:
        idx = cut.get("idx")
        transition = cut.get("transition")
        if not isinstance(transition, dict) or not _text(transition.get("type")):
            errors.append(f"Cut {idx} needs a nonempty transition.type")
        else:
            if transition["type"] == "no_change" and (
                    not isinstance(idx, int) or isinstance(idx, bool)
                    or measured.get(idx, {}).get("continuous") is not True):
                errors.append(f"Cut {idx}: no_change requires measured continuous=true")
            for key, value in transition.items():
                if key == "type":
                    continue
                if key in {"fade_out", "black_hold", "fade_in", "dur"} or isinstance(value, (int, float)):
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        errors.append(f"Cut {idx} transition.{key} must be a finite nonnegative number")
                elif not _text(value):
                    errors.append(f"Cut {idx} transition.{key} must be nonempty text or a finite nonnegative number")
        for field in ("technique", "note"):
            if not _text(cut.get(field)):
                errors.append(f"Cut {idx} needs a nonempty {field}")
        if _text(cut.get("technique")):
            techniques.add(cut["technique"])
    if diversity and not prefix and len(cuts) >= 20 and len(techniques) < 3:
        errors.append(f"Only {len(techniques)} technique labels across {len(cuts)} cuts; "
                      "the inherited whole-film review heuristic requires human review, not invented labels")
    return errors


def make_cut_batch(inputs, manifest, start, stop):
    """Copy consecutive cuts with their measured rows and relevant units only."""
    errors = validate_inputs(inputs)
    if errors:
        raise ValueError("; ".join(errors))
    cuts = inputs["cut_list"]["cuts"]
    if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int)
            or not isinstance(stop, int) or not 0 <= start < stop <= len(cuts)
            or stop - start > CUT_BATCH_SIZE):
        raise ValueError("S4 batch must contain 1 to 12 consecutive cuts")
    records = manifest.get("cuts")
    identity = lambda c: (c.get("idx"), c.get("after"), c.get("before"))
    if (not isinstance(records, list) or any(not isinstance(c, dict) for c in records)
            or [identity(c) for c in records] != [identity(c) for c in cuts]):
        raise ValueError("S4 visual manifest must match all input cut identities in order")
    batch = deepcopy(inputs)
    chosen = cuts[start:stop]
    wanted_units = {c[k] for c in chosen for k in ("after", "before")}
    batch["units"]["units"] = [deepcopy(u) for u in inputs["units"]["units"] if u["id"] in wanted_units]
    # Source counts describe the whole film, not this subset.
    batch["units"].pop("counts", None)
    for key in ("cut_list", "audio", "picture"):
        batch[key]["cuts"] = deepcopy(inputs[key]["cuts"][start:stop])
    batch["cut_list"]["n_cuts"] = stop - start
    batch["picture"]["n"] = stop - start
    batch["picture"]["n_continuous"] = sum(c.get("continuous") is True for c in batch["picture"]["cuts"])
    visual_subset = {k: deepcopy(v) for k, v in manifest.items()
                     if k not in {"cuts", "sheets", "inputs_paths", "draft_path"}}
    visual_subset["cuts"] = deepcopy(records[start:stop])
    # Each cut record directly names its mandatory strip and individual frames.
    # Omit global input/output pointers and sheet inventory. The caller supplies
    # paths to this batch's frozen inputs and its unique output directory.
    return batch, visual_subset


def build_cut_prompt(batch_dir, output_path, vocabulary):
    directory = Path(batch_dir)
    paths = "\n".join(f"- {json.dumps(str(directory / filename))}" for filename in INPUT_FILES.values())
    return f"""# Inspect one bounded batch of fine-cut boundaries

Write all newly authored notes and explanations in English. Preserve IDs,
supplied dialogue, visible card text, and proper names in their original form.

Read these frozen measured inputs for THIS batch only:
{paths}
Read the visual manifest at {json.dumps(str(directory / 'manifest.json'))}.
For EVERY listed cut, individually Read its `strip_path` before annotating it.
Each strip contains timestamped consecutive frames from nominal before/after
sampling windows relative to the supplied `timeline_s` and the manifest's declared
numerical tolerance. These side labels do not establish where the visible scene
actually changes; the first nominal after frame can still show the earlier scene.
Consider the full sequence. If the exact transition remains uncertain, describe
only supported facts and state that uncertainty in `note`; never re-measure or
change the inputs. A strip is an inspection aid, not proof of continuous motion,
exact effect duration, or audio. Inspect listed high-resolution frames or nearby `reference/golden.mp4`
frames when needed. Use the original-reference `timeline_s`, never the assembled
timeline time. Do not replace per-boundary inspection with a whole-video sample.

Do not modify these inputs, the global draft, or stages/s4_cuts.json. Do not
re-measure the reference with analysis scripts. Audio evidence comes only from
the supplied speech/music labels. Use `no_change` only when this cut's supplied
picture measurement has `continuous: true`. A pair of stills alone cannot
establish continuous motion. Be conservative about ambiguous imagery and do
not invent specific actions, identities, durations, or varied technique labels.
Do not delegate.

Write only {json.dumps(str(output_path))}: a JSON object with `stage: "S4"` and
`cuts`. Include every batch cut exactly once in input order, copying its `idx`,
`after`, and `before` exactly. Each also needs `transition` (object with nonempty
`type`), a nonempty `technique`, and a specific evidence-based `note` relating the
pictures to the supplied audio evidence. Include transition parameters only when
supported; numeric parameters must be finite nonnegative numbers, never guesses.
Use nonempty text for direction or other descriptive parameters.

The following vocabulary is inherited from the shared S4 prompt:
{vocabulary}

This batch contains at most 12 cuts. Review only these cuts, save the batch output,
and end with a short completion message. The caller validates and merges it.
The whole-film technique-diversity heuristic is NOT a per-batch quota; never
fabricate labels to meet it. Report insufficient evidence rather than pretending
that contract checks certify visual or interpretive accuracy.
"""


def atomic_write_bytes(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, document):
    atomic_write_bytes(path, (json.dumps(document, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n").encode("utf-8"))
