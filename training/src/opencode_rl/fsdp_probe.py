"""Four-rank FSDP2 regression on recorded turns; no optimizer/checkpoint writes."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

from .contract import Contract, canonical
from .install_training_fixes import verify_installed
from .records import load_turn
from .store import TraceStore, atomic_bytes
from .training import assemble_images, attach_modality_ids, modality_ids, verify_training_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TensorList, pad_training_input_batch
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
    from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator
    sys.path.insert(0, '/patch')
    import fused_ce
    fused_ce.apply()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    if dist.get_world_size() != 4:
        raise RuntimeError('This regression requires four actual training ranks')
    implementation = verify_installed()
    capture, output = Path(args.capture), Path(args.output)
    contract = Contract(**json.loads((capture / 'contract.json').read_bytes()))
    store = TraceStore(capture / 'trace')
    model_path = Path('/hf/hub/models--Qwen--Qwen3.5-9B/snapshots') / contract.model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    generated = {key: [] for key in ('prompt_token_ids', 'response_ids', 'loss_masks', 'rollout_logprobs',
                                     'pixel_values', 'image_grid_thw', 'mm_token_type_ids')}
    # Five real rows force the actual SkyRL DP-padding branch to clone an image
    # row three times. Other ranks also see pure-text and generated-image rows.
    names = ['image', 'text', 'generated_image', 'image_cache', 'text']
    for name in names:
        turn = load_turn(store, store.attempt_requests(name)[0]['request_id'], contract)
        pixels, grid = assemble_images(store, turn, contract)
        for key, value in (('prompt_token_ids', turn['input']['prompt_token_ids']), ('response_ids', turn['response_ids']),
                           ('loss_masks', turn['loss_mask']), ('rollout_logprobs', turn['rollout_logprobs']),
                           ('pixel_values', pixels), ('image_grid_thw', grid), ('mm_token_type_ids', modality_ids(turn))):
            generated[key].append(value)
    tensors = convert_prompts_responses_to_batch_tensors(tokenizer.pad_token_id, generated['prompt_token_ids'],
        generated['response_ids'], [[0.] * len(r) for r in generated['response_ids']], generated['loss_masks'],
        generated['rollout_logprobs'], max_seq_len=contract.max_context)
    data = TrainingInputBatch(dict(zip(('sequences', 'attention_mask', 'response_mask', 'rewards', 'loss_mask', 'rollout_logprobs'), tensors)))
    data['pixel_values'] = TensorList(generated['pixel_values'])
    data['image_grid_thw'] = TensorList(generated['image_grid_thw'])
    data.metadata = {'response_length': max(map(len, generated['response_ids']))}
    data = pad_training_input_batch(data, 3)
    attach_modality_ids(data, generated)
    verify_training_batch(data, generated)
    batch = data.slice(2 * rank, 2 * rank + 2)
    reference = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
        local_files_only=True, device_map={'': f'cuda:{rank}'}, attn_implementation='sdpa')
    reference.config.use_cache = False
    for p in reference.model.visual.parameters():
        p.requires_grad_(False)
    mesh = init_device_mesh('cuda', (4,))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in reference.model.language_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=precision)
    fully_shard(reference, mesh=mesh, mp_policy=precision)
    actor = HFModelWrapper(reference, bf16=True, sequence_parallel_size=1, remove_microbatch_padding=False)
    actor.is_vlm = True
    # This is the installed SkyRL worker method, not a copied forward call.
    worker = NS(model=actor, cfg=NS(algorithm=NS(temperature=1.)))
    actual = PolicyWorkerBase._forward_micro_batch(worker, batch)['output']
    experience = BaseBatchIterator.batch_to_experience(batch)
    experience.to_device(torch.cuda.current_device())
    actor.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lp = actor(experience.sequences, experience.num_actions, attention_mask=experience.attention_mask,
                   temperature=1., pixel_values=experience.pixel_values, image_grid_thw=experience.image_grid_thw,
                   mm_token_type_ids=experience.mm_token_type_ids)
        mask = experience.loss_mask.to(lp.dtype)
        loss = -(lp * mask).sum() / max(1., float(mask.sum()))
    loss.backward()
    finite, count = True, 0
    for p in reference.parameters():
        if p.grad is not None:
            grad = p.grad.to_local() if hasattr(p.grad, 'to_local') else p.grad
            finite &= bool(torch.isfinite(grad).all())
            count += 1
    target = batch['rollout_logprobs'].cpu()
    active = batch['loss_mask'].cpu().bool()
    delta = (actual.float() - target).abs()[active]
    item = {'rank': rank, 'finite_gradients': finite, 'parameters_with_grad': count,
            'mean_abs_logprob_diff': float(delta.mean()) if delta.numel() else None,
            'max_abs_logprob_diff': float(delta.max()) if delta.numel() else None,
            'zero_loss_padding_rank': not bool(active.any()), 'loss': float(loss.detach())}
    atomic_bytes(output / f'rank_{rank}.json', canonical(item))
    gathered = [None] * 4
    dist.all_gather_object(gathered, item)
    if rank == 0:
        result = {'status': 'passed' if all(x['finite_gradients'] and x['parameters_with_grad'] for x in gathered) else 'failed',
                  'ranks': gathered, 'actual_worker_forward': True, 'actual_experience_conversion': True,
                  'actual_fsdp2_world_size': 4, 'generated_image_preserved': True, 'dp_padding_rows': 3,
                  'implementation': implementation, 'optimizer_steps': 0, 'training_started': False}
        atomic_bytes(output / 'result.json', canonical(result))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()
    if not all(x['finite_gradients'] and x['parameters_with_grad'] for x in gathered):
        raise RuntimeError('Distributed forward/backward regression failed')


if __name__ == '__main__':
    main()
