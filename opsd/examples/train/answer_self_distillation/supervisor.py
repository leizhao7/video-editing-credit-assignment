"""Reuse the frozen 4i+4t supervisor, changing ONLY its entrypoint allowlist.

The old module/file is never mutated. Its runtime-checked implementation keeps
owned-child cleanup, inference admission, durable traces and completion checks.
"""
import inspect

from .answers import require
from .launch import MODULE


async def supervise(plan, contract, argv, commit):
    import append_agent_rl.launch as frozen
    require(argv[2] == MODULE, "Wrong OPSD trainer entrypoint")
    require(plan["opsd"]["runtime_profile"] == "frozen_v15", "Require the verified deployed supervisor")
    source = inspect.getsource(frozen.supervise)
    anchor = "argv[2] != 'append_agent_rl.train_entrypoint'"
    require(source.count(anchor) == 1, "Frozen supervisor entrypoint anchor changed")
    source = source.replace(anchor, f"argv[2] != {MODULE!r}")
    namespace = dict(vars(frozen))
    exec(compile(source, "<isolated-answer-opsd-supervisor>", "exec"), namespace)
    return await namespace["supervise"](plan, contract, argv, commit)
