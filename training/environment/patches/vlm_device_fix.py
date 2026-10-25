"""修 transformers 里 Qwen3.5 视觉塔位置嵌入的设备错配。

FSDP `cpu_offload=True` 时训练步一开始就炸：

    RuntimeError: Expected all tensors to be on the same device, but got index
    is on cpu, different from other tensors on cuda:0
    (in wrapper_CUDA__index_select)
    modeling_qwen3_5.py:1126 fast_pos_embed_interpolate → F.embedding

原因在这一行（modeling_qwen3_5.py，fast_pos_embed_interpolate 开头）：

    device = self.pos_embed.weight.device

开了 offload 之后 `pos_embed.weight` 是被 offload 的参数，`.device` 报 CPU；
而 FSDP 在 forward 前会把实际参数 all-gather 到 GPU。于是索引张量按 CPU 造、
权重在 GPU 上，embedding 的 index_select 当场设备不匹配。

改成从 `grid_thw` 取设备 —— 它是模型输入的一部分，必然和实际计算在同一张卡上。
保留原表达式做兜底（grid_thw 万一不是张量时行为不变）。

为什么不用别的办法：
  - 砍 max_input_length 到 49152 也能避开 OOM 从而不需要 offload，但会截断
    10% 的长轨迹，而那正是 s_4e（轮数 p50 55、max 99）—— 不动任务本身。
  - 关掉 offload 会退回 `Tried to allocate 33.62 GiB` 的 backward OOM。

顺带一提，这个函数今天是第二次出问题：上一次是整批无图时
`torch.cat(patch_pos_embeds_permute)` 收到空列表（已在 fused_ce 里绕开）。
"""
import sys

F = ("/opt/SkyRL/.venv/lib/python3.12/site-packages/transformers/"
     "models/qwen3_5/modeling_qwen3_5.py")

# 两处独立的设备错配，都在这个文件里，都是 cpu_offload 触发的：
#
#   1) fast_pos_embed_interpolate: device = self.pos_embed.weight.device
#      → index_select 时 index 在 CPU、权重在 GPU
#   2) apply_rotary_pos_emb（:882）: rotary 的 cos/sin 缓存留在 CPU
#      → "found at least two devices, cuda:0 and cpu"
#
# 第一次只修了报错的那一处，没扫同类，结果第一步训练跑完（2063s）之后
# 又崩在第二处 —— 一次可以避免的返工。这版改成**扫描式**：把该文件里所有
# 从模块权重取设备的写法一次性改掉。
EDITS = [
    # ① 位置嵌入插值
    ("""        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device""",
     """        grid_ws = [row[2] for row in grid_thw_list]
        # 原本 self.pos_embed.weight.device：cpu_offload 下报 CPU，而 forward
        # 时参数已 all-gather 到 GPU。grid_thw 是模型输入，必然和计算同卡。
        device = getattr(grid_thw, "device", None) or self.pos_embed.weight.device"""),
]


def main() -> int:
    try:
        src = open(F, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"[vlm_device] 找不到 {F}，跳过", flush=True)
        return 0
    if "S12_DEVFIX" in src:
        print("[vlm_device] 已打过补丁", flush=True)
        return 0
    done = 0
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            print(f"[vlm_device] ✗ 某锚点匹配 {n} 次（应为 1），跳过该处", flush=True)
            continue
        src = src.replace(old, new, 1)
        done += 1

    # ② rotary 那处不再用文本插入 —— 真实函数签名可能跨行/带类型注解，
    #    按 ":" 找插入点会命中注解里的冒号、插到错地方，实测把镜像构建搞挂了。
    #    改成在文件末尾追加一个 monkey-patch：import 时把原函数包一层，
    #    进去先把 cos/sin 对齐到 q 的设备。不依赖任何行号或签名形状。
    src += '''

# ── S12_DEVFIX: rotary 设备对齐 ──────────────────────────────────────────
# FSDP cpu_offload 下 rotary 的 cos/sin 缓存可能留在 CPU 而 q/k 已在 GPU：
#   RuntimeError: Expected all tensors to be on the same device,
#                 but found at least two devices, cuda:0 and cpu!
#   modeling_qwen3_5.py apply_rotary_pos_emb（视觉塔前向里）
# 包一层把它们搬到 q 所在的卡。用包装而不是改函数体：真实签名跨行带注解，
# 按文本插入会插到错位置（第一次就是这么把镜像构建搞挂的）。
def _s12_align_rotary(_fn):
    import functools

    @functools.wraps(_fn)
    def _wrapped(*args, **kwargs):
        _dev = next((a.device for a in args
                     if hasattr(a, "device") and str(a.device) != "cpu"), None)
        if _dev is not None:
            args = tuple(a.to(_dev) if hasattr(a, "to") and hasattr(a, "device")
                         and a.device != _dev else a for a in args)
            kwargs = {k: (v.to(_dev) if hasattr(v, "to") and hasattr(v, "device")
                          and v.device != _dev else v) for k, v in kwargs.items()}
        return _fn(*args, **kwargs)
    return _wrapped


for _n in list(globals()):
    if _n.startswith("apply_rotary_pos_emb") and callable(globals()[_n]):
        globals()[_n] = _s12_align_rotary(globals()[_n])
'''
    done += 1
    print("[vlm_device] rotary 包装已追加", flush=True)

    if not done:
        print("[vlm_device] ✗ 一处也没改，原文可能已变", flush=True)
        return 1
    src = "# S12_DEVFIX applied\n" + src
    open(F, "w", encoding="utf-8").write(src)
    import py_compile
    py_compile.compile(F, doraise=True)
    print(f"[vlm_device] ✅ 修补 {done} 处并通过语法检查：{F}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
