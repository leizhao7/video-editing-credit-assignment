"""Get the fused-CE patch into the Ray workers, without loading torch everywhere.

SkyRL runs its policy workers as Ray actors in separate processes. Patching
``HFModelWrapper`` from the launcher would leave every process that actually
holds a shard running the stock forward — the patch would appear to apply,
print its log line, and change nothing. ``sitecustomize`` is imported
automatically by every Python interpreter at startup, so putting it on
``PYTHONPATH`` covers driver and actors alike without needing Ray's
``runtime_env`` plumbing, which the SFT config does not expose.

The catch, learned by doing it the obvious way first: patching *at* startup
means importing ``model_wrapper``, and behind that torch and transformers, in
every process the interpreter starts. Ray spawns a handful of short-lived
helpers while bringing up a node, none of which need any of that, and the extra
seconds each one spends importing pushed node startup past its deadline:

    Exception: The current node timed out during startup. This could happen
    because some of the raylet failed to startup or the GCS has become
    overloaded.

So nothing is imported here. An ``__import__`` wrapper watches for the moment
``model_wrapper`` actually lands in ``sys.modules`` and patches it then. In a
raylet helper that moment never comes and this file costs one dict lookup per
import; in a policy worker it fires exactly once, before any forward runs.
"""
import builtins
import os
import sys

# ── 卡死取证：收到 SIGUSR1 就把**所有线程**的调用栈打到 stderr ──────────
# 保留 handler 只是让外部/人工发来的 SIGUSR1 变成无害的栈转储。
# _res_monitor 已改成只读 /proc/PID/wchan，永远不发信号：只有这样才能
# 保证看门狗本身没有杀死 runner/raylet/训练进程的能力。
try:
    import faulthandler as _fh
    import signal as _sig
    _fh.register(_sig.SIGUSR1, all_threads=True, chain=False)
    # 致命信号（段错误 / abort / 浮点异常）直接把 C 层栈打到 stderr。
    # SIGUSR1 那条是「外面叫我 dump」，这条是「我自己要死了」——
    # 后者才对应 Runner failed with exit code: 2 那种进程凭空消失。
    _fh.enable(all_threads=True)
except Exception:  # noqa: BLE001
    pass

_TARGET = "skyrl.backends.skyrl_train.workers.model_wrapper"

if os.environ.get("SKYRL_FUSED_CE", "1") != "0":
    _real_import = builtins.__import__
    _done = False
    _vl_done = False
    _trim_done = False
    _nan_done = False
    _freeze_done = False

    def _import_and_patch(name, *args, **kwargs):
        global _done, _trim_done, _nan_done, _freeze_done
        module = _real_import(name, *args, **kwargs)
        # 便宜的检查：一次 dict 查找加一次 getattr。绝不在这里主动 import
        # 任何东西 —— 主动 import 正是上一版拖垮 raylet 启动的原因。
        #
        # 条件是「类已经存在」而不是「模块已经在 sys.modules 里」：模块
        # 在**开始执行**时就被登记进 sys.modules，那时 HFModelWrapper 还
        # 没定义。按后者判断会在模块自身的 import 期间就开火，撞上
        #   AttributeError: partially initialized module ... has no
        #   attribute 'HFModelWrapper' (most likely due to a circular import)
        # 之后还有大量 import 会经过这里，所以模块装完自然会补上。
        _mod = sys.modules.get(_TARGET)
        if not _done and getattr(_mod, "HFModelWrapper", None) is not None:
            _done = True          # 先置位，apply() 里的 import 会重入这里
            try:
                import fused_ce
                fused_ce.apply()
            except Exception as e:  # noqa: BLE001
                # 到这一步 model_wrapper 已经在 sys.modules 里了，所以
                # 失败一定是补丁本身的问题，不是"这个进程不相干"。
                # 不能静默：静默的结果是 131k 上 OOM，和没打补丁一模一样。
                print(f"[sitecustomize] fused_ce 打补丁失败: {e!r}", file=sys.stderr)
                raise
        # Qwen3.5-9B RL freezes the pretrained visual encoder before FSDP and
        # AdamW are built.  Frames/features are unchanged; only visual backward
        # is removed.  This must run before the first HFModelWrapper instance.
        if (not _freeze_done
                and os.environ.get("SKYRL_FREEZE_VISION", "0") == "1"
                and getattr(_mod, "HFModelWrapper", None) is not None):
            _freeze_done = True
            try:
                import freeze_vision
                freeze_vision.apply()
            except Exception as e:  # noqa: BLE001
                print(f"[sitecustomize] vision freeze failed: {e!r}",
                      file=sys.stderr)
                raise
        if (not _trim_done
                and os.environ.get("SKYRL_TRIM_MICROBATCH", "0") == "1"
                and getattr(_mod, "HFModelWrapper", None) is not None):
            _trim_done = True
            try:
                import microbatch_trim
                microbatch_trim.apply()
            except Exception as e:  # noqa: BLE001
                print(f"[sitecustomize] microbatch trim 打补丁失败: {e!r}",
                      file=sys.stderr)
                raise
        # Expensive and intentionally diagnostic-only.  The exact-batch replay
        # sets the env flag before Ray starts, so each policy actor installs the
        # wrapper in its own interpreter.  Ordinary RL/SFT processes never
        # import this module and keep the stock performance path.
        _wm = sys.modules.get("skyrl.backends.skyrl_train.workers.worker")
        if (not _nan_done
                and os.environ.get("S12_NAN_DIAGNOSTIC", "0") == "1"
                and getattr(_wm, "PolicyWorkerBase", None) is not None):
            _nan_done = True
            try:
                import nan_diagnostic
                nan_diagnostic.apply()
            except Exception as e:  # noqa: BLE001
                print(f"[sitecustomize] NaN diagnostic patch failed: {e!r}",
                      file=sys.stderr)
                raise
        # 同一机制的第二个目标：Qwen3_5Model 出现后打 dummy visual 补丁
        # （无图 micro 的视觉塔 collective 对齐，见 vl_dummy_fix.py 的档案）。
        global _vl_done
        _qm = sys.modules.get("transformers.models.qwen3_5.modeling_qwen3_5")
        if (not _vl_done
                and os.environ.get("SKYRL_VL_DUMMY", "1") != "0"
                and getattr(_qm, "Qwen3_5Model", None) is not None):
            _vl_done = True
            try:
                import vl_dummy_fix
                vl_dummy_fix.apply()
            except Exception as e:  # noqa: BLE001
                print(f"[sitecustomize] vl_dummy 打补丁失败: {e!r}", file=sys.stderr)
                raise
        return module

    builtins.__import__ = _import_and_patch
