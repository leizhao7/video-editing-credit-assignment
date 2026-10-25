"""Stop materialising the [seq, vocab] logits so sequences can be long again.

The cap on this run is not the model, it is the vocabulary. Qwen3.6's is
248,320 tokens, so the logits tensor costs seq × 248,320 × 2 bytes and is live
twice — forward and backward — which works out to 0.993 MB per token against
0.655 MB for the activations of all 64 layers. Logits are the larger half.

With ~87 GB free per H200 after the 54 GB of sharded weights, gradients and
optimizer state, that puts the ceiling near 52k tokens, hence the 49,152 the
run has been using. The teacher trajectories are median 44.5k, p99 126k. So
roughly half of them get truncated — and truncation takes the tail, which is
where the submitted edit lives. For action cloning that does not lose tokens,
it loses labels.

Removing the logits term moves the ceiling to about 87 / 0.655 ≈ 132k, which
covers p99.

Why a patch and not a flag
--------------------------
SkyRL already owns the kernel for this: ``LinearCrossEntropy`` in
``distributed/megatron/fused_linear_logprob_triton.py``, whose comment reads
"Triton doesn't materialize d_logits at all." It is only wired into the
Megatron backend, and the Megatron backend is not available to us — its bridge
deliberately routes Qwen3.5-family checkpoints to a text-only ``GPTModel``,
dropping the vision tower. Vision only survives on FSDP, and the FSDP forward
does ``logits_BSV = output["logits"]``.

The other obvious lever, Ulysses sequence parallelism, is closed by assertion:

    assert self.sequence_parallel_size == 1, \
        "Sequence parallelism is not supported with VLM vision inputs"

Scope
-----
This replaces ``HFModelWrapper.forward`` rather than editing SkyRL in place, so
it can be turned off for an A/B against the stock path. The replacement is
short because it only covers the configuration actually in use — VLM, no
packing, no sequence parallelism — and asserts each of those rather than
reimplementing the branches. If a future config drifts out of that envelope the
patch fails loudly instead of quietly computing something else.

The known-uncertain part is FSDP2: parameters are DTensors, and reading
``lm_head.weight`` outside that module's own forward can hand back a shard
rather than the full matrix. ``_full_weight`` deals with it, but this is the
kind of thing that gets settled by running it, not by reading about it.

    MILES_UNUSED=  SKYRL_FUSED_CE=1  python -m skyrl.train.main_sft ...
"""
from __future__ import annotations

import os

import torch
from loguru import logger

_APPLIED = False
_RAN = False
_CALLED = False


def _full_weight(w: torch.Tensor) -> torch.Tensor:
    """FSDP2 分片下拿完整的 lm_head 权重。

    DTensor.full_tensor() 是可微的 all-gather，梯度会正确散回分片。
    248,320 × 5120 × 2B ≈ 2.5 GB，在省下来的余量里可以忽略；不这么做的话
    kernel 拿到的是 [vocab/8, hidden]，算出来的 logprob 静默地只覆盖
    1/8 的词表。

    cpu_offload=true 下还得先把本地分片搬回 GPU。这里访问 head.weight 的时
    机在 self.model(...) **返回之后** —— FSDP2 那时已经 reshard，而
    CPUOffloadPolicy 的分片存储在 CPU 上。直接 full_tensor() 等于拿 CPU 张量
    去做 all-gather，撞上按 CUDA 建的 mesh：
        RuntimeError: No backend type associated with device type cpu
    非 offload 下分片仍在 GPU 上，所以同步单节点跑 130 步都没碰到这条。
    修法是过一道 from_local —— 它内部就按 device_mesh.device_type 做迁移
    （torch/distributed/tensor/_api.py:482-485），不用自己写 .to()：
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)
    shape/stride 显式传全局值。文档原话是"各 rank 本地形状不同时必须提供"，
    而 FSDP2 正是会 pad 分片的那种情况；lm_head 恰好 248320/8 整除不需要，
    但不能指望下一个模型也整除。
    """
    if not hasattr(w, "full_tensor"):
        return w
    _mesh = w.device_mesh
    _local = w.to_local()
    if _local.device.type != _mesh.device_type:
        # from_local 是 staticmethod，用 type(w) 取免掉一次 import
        w = type(w).from_local(_local, _mesh, w.placements,
                               shape=w.shape, stride=w.stride())
    return w.full_tensor()


def _fused_forward(
    self,
    sequences,
    num_actions,
    attention_mask=None,
    temperature: float = 1.0,
    return_output=False,
    compute_entropy=False,
    entropy_requires_grad=True,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
):
    """HFModelWrapper.forward，logits 不落地的版本。

    只覆盖当前实际在跑的那一种配置。其余分支不是不支持，是没验证过 ——
    与其抄一份可能抄错的，不如让它当场断掉。
    """
    from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
        LinearCrossEntropy,
    )
    from skyrl.backends.skyrl_train.training_batch import TensorList

    # ── buffer 设备对齐 ────────────────────────────────────────────────
    # FSDP 的 CPUOffloadPolicy 只管 **parameters, gradients, optimizer
    # states**（torch/distributed/fsdp/_fully_shard/_fsdp_api.py 的 docstring
    # 原话）：参数每次 forward 前 all-gather 到 GPU，**buffer 没有这个机制**。
    # 而 _get_managed_modules 的 DFS 又不递归进已被 fully_shard 包过的子模块，
    # 所以哪些 buffer 被移到 GPU 取决于包装顺序 —— 漏下的就留在 CPU。
    #
    # 结果是 CPU 上的 buffer 撞 GPU 上的激活，连撞三处、每次换个地方：
    #   1) index_select @ fast_pos_embed_interpolate（位置嵌入）
    #   2) 视觉塔 rotary 的 cos/sin
    #   3) mat2 is on cuda:0 / others on cpu  = inv_freq(CPU) @ position_ids(GPU)
    # 逐处打补丁是打地鼠 —— 病根一个，发作点无穷。
    #
    # 在前向入口一次性对齐：buffer 都很小（inv_freq 几 KB），搬一次几乎零成本，
    # 而且只在有 CPU buffer 时才动手，之后的 forward 直接跳过。
    # **每次 forward 都扫**，不是只扫一次：rotary 的 dynamic_rope_update 会在
    # 前向中重新创建 inv_freq（序列变长时重算），一次性对齐挡不住它。
    # 扫描本身很便宜 —— 27B 的 buffer 只有几十个、每个几 KB，全在设备上时
    # 循环里一次 .to() 都不会调用。
    _dev = getattr(sequences, "device", None)
    if _dev is not None and _dev.type != "cpu":
        _moved = 0
        _names = []
        _first = not getattr(self, "_s12_buf_warned", False)
        for _mn, _m in self.model.named_modules():
            for _bn, _b in list(_m._buffers.items()):
                if _b is not None and _b.device != _dev:
                    # 直接改 _buffers 而不是 setattr —— 保持注册关系，
                    # 也不触发 nn.Module.__setattr__ 的类型检查。
                    _m._buffers[_bn] = _b.to(_dev)
                    _moved += 1
                    # 首次把名字记下来：这是唯一能坐实"到底哪些 buffer 掉回
                    # CPU"的机会，光有计数无法回答根因。之后不再收集。
                    if _first and len(_names) < 6:
                        _names.append(f"{_mn}.{_bn}[{tuple(_b.shape)}]")
        if _moved and _first:
            self._s12_buf_warned = True
            print(f"[fused_ce] 对齐 {_moved} 个 buffer 到 {_dev}"
                  f"（FSDP cpu_offload 不搬 buffer；后续静默处理）"
                  f" 样例={_names}", flush=True)

    assert self.is_vlm, "fused_ce 补丁只为 VLM 路径写的"
    assert self.sequence_parallel_size == 1, "SP>1 未覆盖（VLM 下本来也被断言禁掉）"
    assert not self.remove_microbatch_padding, "packing 未覆盖（VLM 下本来也被禁掉）"

    if isinstance(pixel_values, TensorList):
        pixel_values = torch.cat(pixel_values.tensors, dim=0)
    if isinstance(image_grid_thw, TensorList):
        image_grid_thw = torch.cat(image_grid_thw.tensors, dim=0)

    # 整个 micro-batch 一张图都没有 —— 每条轨迹都没调过 view_image，直接
    # 作答。这在 GRPO 里是合法且常见的一类行为（而且 reward 低，正是要区分
    # 的），micro_forward=4 时四条凑一起并不罕见。
    #
    # 但空张量不能当"有图"往下传：模型见 pixel_values 非 None 就会进视觉塔，
    # 而 grid 是 0 行，最后炸在
    #   qwen3_5/modeling_qwen3_5.py:1141 fast_pos_embed_interpolate
    #   ValueError: torch.cat(): expected a non-empty list of Tensors
    # 一个 micro-batch 踩到，整步的 GPU 时间全废。
    #
    # 传 None 而不是空张量：模型据此完全跳过视觉塔，走纯文本路径 —— 这正是
    # 这批数据的真实语义。
    if image_grid_thw is not None and image_grid_thw.numel() == 0:
        pixel_values, image_grid_thw = None, None

    from opencode_rl.forward_guard import check_forward
    check_forward(sequences, attention_mask, mm_token_type_ids, temperature)
    if image_grid_thw is not None and mm_token_type_ids is None:
        mm_token_type_ids = (sequences == self.model.config.image_token_id).long()

    vlm_kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    if mm_token_type_ids is not None:
        vlm_kwargs["mm_token_type_ids"] = mm_token_type_ids

    # SkyRL left-pads every dense batch.  Its stock VLM branch computes the
    # correct cumsum-based positions but then discards them and calls the model
    # with ``position_ids=None``.  Qwen3.5 can reconstruct M-RoPE positions when
    # a batch contains images, but its *pure-text* fallback is a plain
    # ``arange(seq_len)`` and therefore counts left PAD columns.  vLLM rolls out
    # one unpadded request at a time, so those shifted positions create a large
    # rollout-vs-policy logprob gap even at staleness zero.
    #
    # For multimodal batches leave this None so Qwen3.5 computes its full 3-D
    # positions from image_grid_thw/mm_token_type_ids.  For text-only batches a
    # 2-D tensor is sufficient; Qwen3_5TextModel expands it over all four RoPE
    # axes internally.  Padding values are irrelevant under attention_mask but
    # setting them to zero also matches Qwen3.5's own get_rope_index path.
    model_position_ids = None
    if (image_grid_thw is None
            and attention_mask is not None
            and os.environ.get("SKYRL_QWEN35_POSITION_FIX", "1") == "1"):
        model_position_ids = attention_mask.long().cumsum(-1) - 1
        model_position_ids.masked_fill_(attention_mask == 0, 0)

    # 从根模块调用，把 lm_head 短路成恒等 —— 这样返回的 "logits" 其实是
    # hidden state，而 [S, 248320] 从来没被算出来过。
    #
    # 上一版是直接调内层的 self.model.model 来拿 hidden，那是错的：FSDP2
    # 把参数存成 DTensor，只在**根模块**的 pre-forward hook 里 all-gather
    # 成本地张量。跳过根模块就跳过了那次 unshard，视觉塔的 conv 权重仍是
    # 分片的 DTensor，撞上普通张量的 pixel_values：
    #   RuntimeError: aten.convolution.default: got mixed torch.Tensor
    #   and DTensor
    # 换句话说，为了拿 hidden 而绕开根模块，正好绕开了唯一的 unshard 时机。
    #
    # 改 forward 而不是替换模块：模块对象不变，FSDP 注册在上面的 hook 和
    # 参数归属都不受影响，只是这一次不做那个投影。
    head = self.model.lm_head
    _orig_head_forward = head.forward
    head.forward = lambda x, *_a, **_k: x
    try:
        out = self.model(sequences, attention_mask=attention_mask,
                         position_ids=model_position_ids, **vlm_kwargs)
    finally:
        head.forward = _orig_head_forward
    hidden = out["logits"] if "logits" in out else out[0]
    assert hidden.shape[-1] == head.weight.shape[-1], (
        f"短路失败：拿到 {tuple(hidden.shape)}，末维应当是 hidden_size "
        f"{head.weight.shape[-1]} 而不是词表")

    # LinearCrossEntropy currently flattens with ``view``.  A model wrapper
    # is allowed to return a strided view (the ordered VL dummy path used to
    # do exactly that after removing its masked suffix), so make the kernel's
    # stronger layout precondition explicit at this boundary.  This is a
    # no-op for the normal contiguous image path.
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()

    # 对齐 dtype。FSDP2 的混合精度把 fp32 主权重在 forward 时临时 cast 成
    # bf16 参与计算，而 full_tensor() 拿的是**存储的那一份**，即 fp32。
    # 不 cast 的话内核里的 GEMM 直接拒绝：
    #   tl.dot(_hidden, _weight.trans(), ...)
    #   AssertionError: Both operands must be same dtype. Got bf16 and fp32
    # 往 hidden 的 dtype 靠而不是反过来：bf16 才是这个模型的计算精度，
    # 把 hidden 抬成 fp32 会让省下来的显存又还回去一半。
    weight = _full_weight(head.weight).to(hidden.dtype)

    # The isolated exact-batch replay needs to distinguish a non-finite loss
    # gradient from a fused-CE backward failure and a deeper transformer-layer
    # failure.  Registering a normal Tensor hook here is compatible with
    # FSDP2/DTensor; torch anomaly mode is not.  Ordinary training never imports
    # the diagnostic helper and pays no hook or finite-scan cost.
    if os.environ.get("S12_NAN_DIAGNOSTIC", "0") == "1":
        import nan_diagnostic

        nan_diagnostic.watch_tensor(hidden, "fused_hidden")

    # reduction="none" 给逐 token logprob，和 logprobs_from_logits 的输出
    # 同形。dist_process_group=None：FSDP 没有词表并行，权重上面已经聚齐。
    from opencode_rl.response_head import response_suffix_linear_ce
    action_log_probs, entropy = response_suffix_linear_ce(
        hidden, weight, sequences, num_actions, temperature,
        linear_ce=LinearCrossEntropy.apply, compute_entropy=compute_entropy)
    if os.environ.get("S12_NAN_DIAGNOSTIC", "0") == "1":
        nan_diagnostic.watch_tensor(action_log_probs, "fused_log_probs")

    global _RAN
    if not _RAN:
        # 打补丁成功和补丁真的被走到，是两回事。前者只证明 apply() 跑过了；
        # 万一 worker 里没生效，缺了这一行就只会表现为在 131k 上 OOM，
        # 和没打补丁一模一样，查起来分不清。
        _RAN = True
        # print 而不是 logger：SkyRL 在启动后会 logger.remove() 重配 loguru
        # 的 sink，非 skyrl 模块的消息被过滤掉。apply() 那行能出现是因为它
        # 在 import 期、重配之前就打了。上一轮两个诊断点全部消失、而 smoke5
        # 的报错栈证明代码确实执行过 —— 丢的是日志不是执行。
        n_tok, vocab = sequences.shape[1], weight.shape[0]
        gib = 2**30
        # 同时写文件。日志这条路已经排查两轮（loguru 被 SkyRL 重配 sink 过滤，
        # 换 print 仍然不见），而 smoke5 的栈里有 /patch/fused_ce.py:143，
        # 证明代码在跑。所以问题在「怎么把话传出来」，不在「有没有执行」。
        # volume 上的文件不经过任何日志系统，modal volume ls 直接能看。
        try:
            import json
            import socket
            with open(f"/ckpt/_fused_ce_ran_{socket.gethostname()}_"
                      f"{torch.cuda.current_device()}.json", "w") as fh:
                json.dump({"tokens": int(n_tok), "vocab": int(vocab),
                           "hidden": list(hidden.shape),
                           "W": list(weight.shape),
                           "peak_gib": torch.cuda.max_memory_allocated() / gib,
                           "alloc_gib": torch.cuda.memory_allocated() / gib}, fh)
        except Exception as e:  # noqa: BLE001
            print(f"[fused_ce] marker 写失败: {e!r}", flush=True)
        # 显存实测。等一个恰好 131k 的 batch 出现是不可靠的 —— padded 长度
        # 取决于批内最长那条，中位数才 44.5k。但激活对 token 数是线性的，
        # 所以任一长度的实测值都能外推到 131k，这比守株待兔强。
        print(
            f"[fused_ce] 生效 tokens={n_tok} hidden={tuple(hidden.shape)} "
            f"W={tuple(weight.shape)} "
            f"alloc={torch.cuda.memory_allocated()/gib:.1f}GiB "
            f"peak={torch.cuda.max_memory_allocated()/gib:.1f}GiB "
            f"（省掉的 logits={n_tok * vocab * 2 / gib:.1f}GiB×2, "
            f"外推 131k 时激活≈"
            f"{torch.cuda.max_memory_allocated()/gib * 131072 / max(n_tok,1):.0f}GiB）",
            flush=True)

    # Response positions were selected before vocabulary projection.

    if not return_output:
        return action_log_probs
    # 刻意不带 "logits"。下游若真的要它，这里应该当场 KeyError，
    # 而不是拿到一个我们没算的东西。
    packed = {}
    if compute_entropy:
        packed["entropy"] = entropy.view(sequences.shape)
    return action_log_probs, packed


def apply_nvrtc_fallback() -> None:
    """让 torch.prod 在 B300 上不要炸。

    B300 是 sm_103。torch 2.11.0+cu128 的 arch list 是
    ``['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`` —— 没有 sm_103。
    大部分算子有预编译的 cubin 或能从 PTX JIT 上来，照常工作（融合 CE 的
    triton 内核在 B300 上实测数值全对）。但 PyTorch 有一部分 reduction 走
    NVRTC 运行时编译，那条路会拿设备的 compute capability 直接去要
    ``--gpu-architecture=compute_103``，被 CUDA 12.8 的 nvrtc 拒掉：

        extern "C" __global__ void reduction_prod_kernel(ReduceJitOp r)
        nvrtc: error: invalid value for --gpu-architecture (-arch)

    这就是当初判定「B300 不能用」的那个错误。它其实只是一个算子 —— VLM
    前向里对 ``image_grid_thw`` 之类 [N,3] 小整数张量求积，算的是每张图
    多少 patch。放 CPU 上做没有任何代价。

    只在真的撞上 nvrtc 才回退，不是无条件绕道：编译失败发生在 kernel 启动
    之前，所以此时张量状态是干净的，重算安全。别的原因导致的 RuntimeError
    原样抛出 —— 把无关的错误也吞掉，比这个问题本身更糟。
    """
    _orig = torch.Tensor.prod

    def _prod(self, *a, **kw):
        try:
            return _orig(self, *a, **kw)
        except RuntimeError as e:
            if "nvrtc" not in str(e):
                raise
            return _orig(self.cpu(), *a, **kw).to(self.device)

    torch.Tensor.prod = _prod

    _orig_fn = torch.prod

    def _prod_fn(inp, *a, **kw):
        try:
            return _orig_fn(inp, *a, **kw)
        except RuntimeError as e:
            if "nvrtc" not in str(e):
                raise
            return _orig_fn(inp.cpu(), *a, **kw).to(inp.device)

    torch.prod = _prod_fn


def apply() -> bool:
    """打上补丁。SKYRL_FUSED_CE=0 时不打，用来做 A/B。"""
    global _APPLIED
    if _APPLIED or os.environ.get("SKYRL_FUSED_CE", "1") == "0":
        return False
    from skyrl.backends.skyrl_train.workers import model_wrapper as mw

    # B300 用；H200 上永远不触发。包 try 是因为它是可选的加速回退，而
    # sitecustomize 那边刻意让打补丁失败 raise —— 那条对诊断主补丁是对的，
    # 但它把这个可选项变成了致命依赖：上一轮 8×B300 全部 worker 起不来，
    #   ActorUnschedulableError: worker startup repeatedly failed
    # 回退装不上最坏也只是在 B300 上撞回 nvrtc，不该拖垮整个进程。
    try:
        apply_nvrtc_fallback()
    except Exception as e:  # noqa: BLE001
        print(f"[fused_ce] nvrtc 回退未装上（不致命）: {e!r}", flush=True)
    mw.HFModelWrapper._stock_forward = mw.HFModelWrapper.forward
    mw.HFModelWrapper.forward = _fused_forward

    # 探针，装在 __call__ 上。上一轮 rc=0 跑完了，但 _fused_forward 里的
    # 那行日志一次都没出现 —— 于是分不清是「forward 没被替换成功」还是
    # 「这个模块根本没被调用」。__call__ 这层任何调用都躲不开，能把两者
    # 分开。它只打一次，然后原样委派。
    _stock_call = mw.HFModelWrapper.__call__

    def _probing_call(self, *a, **kw):
        global _CALLED
        if not _CALLED:
            _CALLED = True
            fn = type(self).forward
            print(
                f"[fused_ce] __call__ 命中 forward={getattr(fn, '__name__', fn)} "
                f"是补丁={fn is _fused_forward} cls={type(self).__name__}",
                flush=True)
        return _stock_call(self, *a, **kw)

    mw.HFModelWrapper.__call__ = _probing_call
    _APPLIED = True
    logger.info("fused_ce: HFModelWrapper.forward 已替换（logits 不落地）")
    return True

# OCRL_EXACT_MEDIA_V7
