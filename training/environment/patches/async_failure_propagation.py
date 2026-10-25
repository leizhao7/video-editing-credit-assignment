"""Propagate fully-async generator failures without killing the Ray driver.

Upstream calls ``os._exit(1)`` from an asyncio generator task.  That bypasses
normal exception propagation and turns a useful Python traceback into a Modal
connection/system error.  Under ``sample_full_batch`` we already have a
consumer wake-up event, so make its watcher stop sibling generators and carry
the original exception into the consumer instead.
"""
from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/train/fully_async_trainer.py"
MARKER = "S12_ASYNC_FAILURE_PROPAGATION"

OLD_TASKS = '''                generator_tasks = [
                    asyncio.create_task(self._run_generate_for_a_group_loop(generation_output_group_buffer))
                    for _ in range(self.num_parallel_generation_workers)
                ]
'''

NEW_TASKS = '''                # S12_ASYNC_FAILURE_PROPAGATION: reset the per-epoch
                # failure channel before starting producer tasks.
                self._generator_worker_error = None
                generator_tasks = [
                    asyncio.create_task(self._run_generate_for_a_group_loop(generation_output_group_buffer))
                    for _ in range(self.num_parallel_generation_workers)
                ]
'''

OLD_WATCHER = '''                    async def _watch_generators_done(tasks=generator_tasks, event=all_generators_done):
                        await asyncio.gather(*tasks, return_exceptions=True)
                        event.set()
'''

NEW_WATCHER = '''                    async def _watch_generators_done(tasks=generator_tasks, event=all_generators_done):
                        try:
                            done, pending = await asyncio.wait(
                                tasks, return_when=asyncio.FIRST_EXCEPTION
                            )
                            failure = None
                            for task in done:
                                if not task.cancelled():
                                    exc = task.exception()
                                    if exc is not None:
                                        failure = exc
                                        break
                            if failure is not None:
                                self._generator_worker_error = failure
                                for task in pending:
                                    task.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                        finally:
                            event.set()
'''

OLD_DRAIN = '''        while True:
            if not buffer.empty():
                return buffer.get_nowait()
            if all_generators_done.is_set():
                return None
'''

NEW_DRAIN = '''        while True:
            failure = getattr(self, "_generator_worker_error", None)
            if failure is not None:
                raise RuntimeError(
                    "Fully-async generation worker failed; aborting before training buffered data."
                ) from failure
            if not buffer.empty():
                return buffer.get_nowait()
            if all_generators_done.is_set():
                return None
'''

OLD_EXIT = '''        except Exception as e:
            logger.error(f"Generator worker errored out with exception: {e}")
            logger.error(f"Traceback: \\n{traceback.format_exc()}")
            sys.stderr.flush()  # flush before os._exit, which otherwise drops buffered output
            os._exit(1)
'''

NEW_EXIT = '''        except Exception as e:
            logger.error(f"Generator worker errored out with exception: {e}")
            logger.error(f"Traceback: \\n{traceback.format_exc()}")
            sys.stderr.flush()
            # S12_ASYNC_FAILURE_PROPAGATION: the watcher records this exception,
            # cancels sibling producers, wakes the consumer, and re-raises there.
            # Never terminate the whole Ray driver from an asyncio child task.
            raise
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[async-failure] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[async-failure] already patched", flush=True)
        return 0

    replacements = (
        (OLD_TASKS, NEW_TASKS, "tasks"),
        (OLD_WATCHER, NEW_WATCHER, "watcher"),
        (OLD_DRAIN, NEW_DRAIN, "drain"),
        (OLD_EXIT, NEW_EXIT, "exit"),
    )
    for old, _new, name in replacements:
        count = source.count(old)
        if count != 1:
            print(f"[async-failure] {name} anchor matched {count}; refusing", flush=True)
            return 1
    for old, new, _name in replacements:
        source = source.replace(old, new)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source)
    py_compile.compile(TARGET, doraise=True)
    print(f"[async-failure] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
