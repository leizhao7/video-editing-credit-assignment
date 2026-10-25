"""让梯度裁剪在 FSDP2 cpu_offload 下能跑。

`cpu_offload=true` 时 `p.grad` 是本地分片在 CPU 上的 DTensor。SkyRL 的
`fsdp2_clip_grad_norm_` docstring 已经写明 "torch.nn.utils.clip_grad_norm_
can't run on cpu parameter DTensor"，但它的处置是

    total_norm = _get_total_norm(grads, ...)
    total_norm = total_norm.to(torch.cuda.current_device())   # ← 搬得太晚

集合通信发生在 `_get_total_norm` **里面**，不是之后。torch 的实现
（clip_grad.py:106）是：

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type)

`first_device` 取自第一个张量，即 CPU；stack 起来的是一串 `Partial` DTensor，
对它做归约就要 all_reduce，而 mesh 是按 "cuda" 建的：

    RuntimeError: No backend type associated with device type cpu

关键观察是**逐个梯度的 vector_norm 不通信** —— 它只在本地分片上算，产出
`Partial`，通信推迟到最后那一步。traceback 落在 106 行而不是更早，正好印证。
所以只要把那些**标量**范数搬到 GPU 再做最后一步，梯度本体一个都不用搬。

不搬本体是有意的：27B 分 8 卡，fp32 梯度是 13.5GB/rank，而 cpu_offload 存在的
理由正是显存不够（实测 98k token 反向 `Tried to allocate 33.62 GiB`）。为了修
裁剪再把 13.5GB 搬回 GPU，等于把 offload 白做了。搬标量的额外开销是零。

裁剪本身不受影响：torch 的 `_clip_grads_with_norm_` 会按每组梯度自己的设备
把 clip 系数 `.to(device)`，所以 total_norm 在 GPU、梯度在 CPU 是它支持的组合
—— 这也正是 SkyRL 原来那句 `.to(cuda)` 想要的效果，只是位置晚了一步。
"""
import sys

F = "/opt/SkyRL/skyrl/backends/skyrl_train/distributed/fsdp_utils.py"

MARK = "S12_CLIP_CPU_FIX"

OLD = (
    "    grads = [p.grad for p in parameters if p.grad is not None]\n"
    "    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)\n"
    "    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)\n"
)

NEW = (
    "    grads = [p.grad for p in parameters if p.grad is not None]\n"
    "    # " + MARK + "：见 /patch/clip_grad_cpu_fix.py。cpu_offload 下梯度的本地\n"
    "    # 分片在 CPU 上，而 _get_total_norm 最后一步（clip_grad.py:106）要对\n"
    "    # 一串 Partial DTensor 做归约 —— 那一步 all_reduce，CPU 张量撞 CUDA mesh。\n"
    "    # 逐个梯度的 vector_norm 只在本地算不通信，所以把**标量**范数搬到 GPU\n"
    "    # 再做最后一步就够，梯度本体（13.5GB/rank）一个都不用搬。\n"
    "    _s12_dev = torch.cuda.current_device()\n"
    "\n"
    "    def _s12_on_cpu(t):\n"
    "        _lt = getattr(t, '_local_tensor', None)\n"
    "        return (_lt if _lt is not None else t).device.type == 'cpu'\n"
    "\n"
    "    if grads and any(_s12_on_cpu(g) for g in grads):\n"
    "        _s12_norms = [torch.linalg.vector_norm(g, norm_type).to(_s12_dev)\n"
    "                      for g in grads]\n"
    "        total_norm = torch.linalg.vector_norm(torch.stack(_s12_norms), norm_type)\n"
    "        if error_if_nonfinite and torch.logical_or(total_norm.isnan(),\n"
    "                                                   total_norm.isinf()):\n"
    "            raise RuntimeError(\n"
    "                f'The total norm of order {norm_type} for gradients from '\n"
    "                '`parameters` is non-finite, so it cannot be clipped.')\n"
    "    else:\n"
    "        total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)\n"
    "    total_norm = total_norm.to(_s12_dev, non_blocking=True)\n"
)


def main() -> int:
    try:
        src = open(F, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"[clip_cpu] ✗ 找不到 {F}", flush=True)
        return 1
    if MARK in src:
        print("[clip_cpu] 已打过补丁", flush=True)
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"[clip_cpu] ✗ 锚点匹配到 {n} 处（应为 1），原文可能已变，不改",
              flush=True)
        return 1
    open(F, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    import py_compile
    py_compile.compile(F, doraise=True)
    print(f"[clip_cpu] ✅ 已修补并通过语法检查：{F}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
