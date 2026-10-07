"""Candidate only: avoid per-token memo lookups for plain scalar arrays.

Not wired into a runtime/image. Preserve deepcopy ownership, cycles, aliases,
custom hooks, explicit memo overrides and the exact captured scalar values.
"""
from copy import deepcopy as _stdlib_deepcopy

_ATOMIC = frozenset((type(None), bool, int, float, str, bytes))
_MISSING = object()


def deepcopy(value, memo=None):
    # A caller-supplied memo may override immutable scalars. A custom hook may
    # introduce such overrides later; once encountered, keep the general path.
    supplied = memo is not None
    return _copy(value, {} if memo is None else memo, [supplied])


def _copy(value, memo, opaque):
    known = memo.get(id(value), _MISSING)
    if known is not _MISSING:
        return known
    kind = type(value)
    if kind in _ATOMIC:
        return value
    if kind is list:
        if not opaque[0] and all(type(item) in _ATOMIC for item in value):
            # No scalar memo entries can exist yet. One C-level list copy
            # replaces id()/memo.get() per token; mutable list stays unshared.
            result = value.copy()
            memo[id(value)] = result
        else:
            result = []
            memo[id(value)] = result
            append = result.append
            for item in value:
                append(memo.get(id(item), item) if type(item) in _ATOMIC
                       else _copy(item, memo, opaque))
    elif kind is dict:
        result = {}
        memo[id(value)] = result
        for key, item in value.items():
            copied = memo.get(id(item), item) if type(item) in _ATOMIC else _copy(item, memo, opaque)
            copied_key = memo.get(id(key), key) if type(key) in _ATOMIC else _copy(key, memo, opaque)
            result[copied_key] = copied
    else:
        opaque[0] = True
        return _stdlib_deepcopy(value, memo)
    memo.setdefault(id(memo), []).append(value)
    return result
