"""Deep copies of captured JSON payloads without per-token Python recursion.

Only exact built-in containers/scalars take the fast path. Mutable containers
are still independently owned, aliases/cycles use a shared memo, and opaque
objects or subclasses retain the standard library's deepcopy behavior. Never
serialize, retokenize, coerce numeric values, or share mutable image metadata.
"""
from copy import deepcopy as _stdlib_deepcopy


_ATOMIC = frozenset((type(None), bool, int, float, str, bytes))
_MISSING = object()


def deepcopy(value, memo=None):
    """Preserve deepcopy ownership semantics, including explicit memo entries."""
    if memo is None:
        memo = {}
    known = memo.get(id(value), _MISSING)
    if known is not _MISSING:
        return known
    kind = type(value)
    if kind in _ATOMIC:
        return value
    if kind is list:
        result = []
        memo[id(value)] = result
        append = result.append
        # IDs, masks, logprobs and shapes dominate these payloads. Avoid a
        # recursive function call and dispatch for every immutable scalar.
        for item in value:
            append(memo.get(id(item), item) if type(item) in _ATOMIC
                   else deepcopy(item, memo))
    elif kind is dict:
        result = {}
        memo[id(value)] = result
        for key, item in value.items():
            # Match stdlib evaluation order: value first, then the key.
            copied = memo.get(id(item), item) if type(item) in _ATOMIC else deepcopy(item, memo)
            copied_key = memo.get(id(key), key) if type(key) in _ATOMIC else deepcopy(key, memo)
            result[copied_key] = copied
    else:
        return _stdlib_deepcopy(value, memo)
    # Match stdlib keep-alive ownership for temporary objects / reused memos.
    memo.setdefault(id(memo), []).append(value)
    return result
