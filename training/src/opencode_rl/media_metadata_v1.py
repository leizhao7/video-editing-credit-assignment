"""Candidate lossless tabular metadata encoding; not deployed.

Only homogeneous multi-row frames/cells metadata are factored. All original
numeric values, ordering, image labels, image bytes, notes and stream facts stay.
This changes public text formatting and needs explicit shared-harness deployment;
a round-trip test is not a model-quality or training-speed guarantee.
"""
from copy import deepcopy
import json

VERSION = "ocrl-media-metadata-rows-v1"
KEYS = ("frames", "cells")
FORMAT_KEY = "metadata_table_format"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compact_result(result):
    if not isinstance(result, dict):
        raise ValueError("Media result must be an object")
    # Never overwrite a preexisting application field or interpret user keys.
    if FORMAT_KEY in result or any(k + "_table" in result for k in KEYS):
        return deepcopy(result)
    out = deepcopy(result)
    changed = False
    for key in KEYS:
        rows = result.get(key)
        if (not isinstance(rows, list) or not 2 <= len(rows) <= 96
                or not all(isinstance(r, dict) and r for r in rows)):
            continue
        columns = list(rows[0])
        if any(set(r) != set(columns) for r in rows):
            continue
        shared = {k: deepcopy(rows[0][k]) for k in columns
                  if all(dumps(r[k]) == dumps(rows[0][k]) for r in rows[1:])}
        fields = [k for k in columns if k not in shared]
        table = {"columns": fields, "shared": shared,
                 "rows": [[deepcopy(r[k]) for k in fields] for r in rows]}
        # Factoring must actually reduce public text, including schema overhead.
        if len(dumps(table)) + len(FORMAT_KEY) + len(VERSION) + 20 >= len(dumps(rows)):
            continue
        del out[key]
        out[key + "_table"] = table
        changed = True
    if changed:
        out[FORMAT_KEY] = VERSION
    return out


def restore_result(value):
    if not isinstance(value, dict):
        raise ValueError("Media result must be an object")
    out = deepcopy(value)
    if out.get(FORMAT_KEY) != VERSION:
        return out
    found = False
    for key in KEYS:
        name = key + "_table"
        if name not in out:
            continue
        if key in out:
            raise ValueError("Ambiguous original and tabular fields")
        table = out.pop(name)
        if not isinstance(table, dict) or set(table) != {"columns", "shared", "rows"}:
            raise ValueError("Invalid metadata table")
        columns, shared, rows = table["columns"], table["shared"], table["rows"]
        if (not isinstance(columns, list) or not all(isinstance(k, str) for k in columns)
                or len(set(columns)) != len(columns) or not isinstance(shared, dict)
                or set(columns) & set(shared) or not isinstance(rows, list)
                or not 2 <= len(rows) <= 96
                or any(not isinstance(r, list) or len(r) != len(columns) for r in rows)):
            raise ValueError("Invalid metadata table dimensions")
        out[key] = [{**deepcopy(shared), **dict(zip(columns, deepcopy(r)))} for r in rows]
        found = True
    if not found:
        raise ValueError("Format marker without a metadata table")
    del out[FORMAT_KEY]
    return out


def compact_tool_text(text):
    """Transform only the leading JSON object; preserve all following labels."""
    if not isinstance(text, str):
        raise ValueError("Text required")
    try:
        result, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return text
    if not isinstance(result, dict):
        return text
    packed = compact_result(result)
    if packed == result:
        return text
    if restore_result(packed) != result:
        raise ValueError("Metadata is not exactly reversible")
    return dumps(packed) + text[end:]
