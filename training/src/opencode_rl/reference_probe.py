"""Replay already-captured raw engine data through the real SkyRL model wrapper.

No generation, no optimizer, no edited request tokens. Optional backward. Includes
actual SkyRL left-padding and multimodal batch handling, including fused CE.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

from .contract import Contract, canonical
from .records import load_turn
from .store import TraceStore, atomic_bytes
from .training import assemble_images, attach_modality_ids, modality_ids, verify_training_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--case', default='')
    parser.add_argument('--stability-check', action='store_true')
    args = parser.parse_args()
    capture, output = Path(args.capture), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from .install_training_fixes import verify_installed
    implementation = verify_installed()
    contract = Contract(**json.loads((capture / 'contract.json').read_bytes()))
    store = TraceStore(capture / 'trace')
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
    from skyrl.backends.skyrl_train.training_batch import TensorList
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    if args.fused:
        import sys
        sys.path.insert(0, '/patch')
        import fused_ce
        fused_ce.apply()
    model_dir = Path('/hf/hub/models--Qwen--Qwen3.5-9B/snapshots') / contract.model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    reference = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa').eval()
    reference.config.use_cache = False
    actor = HFModelWrapper(reference, bf16=True, sequence_parallel_size=1, remove_microbatch_padding=False)
    # The wrapper only auto-detects this when given a model path, not an already
    # loaded HF object. The object is the same Qwen VLM, without a second load.
    actor.is_vlm = True
    actor.eval()
    fn = type(actor).forward
    source_path = Path(inspect.getsourcefile(fn))
    wrapper = {'module': fn.__module__, 'name': fn.__name__, 'path': str(source_path),
               'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
    generated = {key: [] for key in ('prompt_token_ids', 'response_ids', 'loss_masks', 'rollout_logprobs',
                                    'pixel_values', 'image_grid_thw', 'mm_token_type_ids')}
    names = (args.case,) if args.case else ('text', 'image', 'image_cache', 'generated_image') if (capture / 'cases.json').exists() and 'generated_image' in json.loads((capture / 'cases.json').read_bytes()) else ('text', 'image', 'image_cache')
    turns = []
    for name in names:
        requests = store.attempt_requests(name)
        if len(requests) != 1:
            raise RuntimeError('Unexpected captured request count')
        turn = load_turn(store, requests[0]['request_id'], contract)
        turns.append(turn)
        pixels, grid = assemble_images(store, turn, contract)
        for key, value in (('prompt_token_ids', turn['input']['prompt_token_ids']),
                           ('response_ids', turn['response_ids']), ('loss_masks', turn['loss_mask']),
                           ('rollout_logprobs', turn['rollout_logprobs']),
                           ('pixel_values', pixels), ('image_grid_thw', grid), ('mm_token_type_ids', modality_ids(turn))):
            generated[key].append(value)
    tensors = convert_prompts_responses_to_batch_tensors(tokenizer.pad_token_id,
        generated['prompt_token_ids'], generated['response_ids'], [[0.] * len(r) for r in generated['response_ids']],
        generated['loss_masks'], generated['rollout_logprobs'], max_seq_len=contract.max_context)
    data = dict(zip(('sequences', 'attention_mask', 'response_mask', 'rewards', 'loss_mask', 'rollout_logprobs'), tensors))
    data.update({key: generated[key] for key in ('pixel_values', 'image_grid_thw')})
    attach_modality_ids(data, generated)
    verify_training_batch(data, generated)
    max_response = max(map(len, generated['response_ids']))
    def forward():
        return actor(data['sequences'].cuda(), num_actions=max_response,
            attention_mask=data['attention_mask'].cuda(), temperature=1.,
            mm_token_type_ids=data['mm_token_type_ids'].cuda(),
            pixel_values=TensorList([p.to('cuda', dtype=torch.bfloat16) for p in generated['pixel_values']]),
            image_grid_thw=TensorList([g.cuda() for g in generated['image_grid_thw']]))
    def flags():
        return {'matmul_precision': torch.get_float32_matmul_precision(),
                'tf32': torch.backends.cuda.matmul.allow_tf32,
                'bf16_reduced_precision': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
                'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
                'math_sdp': torch.backends.cuda.math_sdp_enabled(),
                'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
                'deterministic': torch.are_deterministic_algorithms_enabled(),
                'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    stability = {'performed': args.stability_check, 'before_flags': flags()}
    if args.stability_check:
        with torch.inference_mode():
            before_import = forward().float().cpu()
            repeat_before = forward().float().cpu()
    # Compare the actual HF M-RoPE implementation against the actual vLLM
    # implementation, using captured feature spans, not image-token guesses.
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    hf_positions, _ = reference.model.get_rope_index(data['sequences'],
        image_grid_thw=torch.cat(generated['image_grid_thw']),
        attention_mask=data['attention_mask'], mm_token_type_ids=data['mm_token_type_ids'])
    position_mismatches = 0
    for i, turn in enumerate(turns):
        features = [NS(modality='image', mm_position=NS(offset=f['offset'], length=f['length']),
                       data={'image_grid_thw': NS(data=store.load_tensor(f['data']['image_grid_thw']).reshape(3))})
                    for f in turn['input']['features']]
        expected_positions, _ = Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
            generated['prompt_token_ids'][i] + generated['response_ids'][i], features, reference.config)
        actual_positions = hf_positions[:, i, data['attention_mask'][i].bool()]
        position_mismatches += int((actual_positions != expected_positions).sum())
    if position_mismatches:
        raise RuntimeError(f'Actual HF/vLLM M-RoPE mismatch: {position_mismatches}')
    with torch.inference_mode():
        action_lp = forward().float().cpu()
        if args.stability_check:
            repeat_after = forward().float().cpu()
            stability['after_flags'] = flags()
            stability['cases'] = []
            for i, name in enumerate(names):
                n = len(generated['response_ids'][i])
                a, b, c, d = [v[i, -n:] for v in (before_import, repeat_before, action_lp, repeat_after)]
                target = a.new_tensor(generated['rollout_logprobs'][i])
                stability['cases'].append({'case': name,
                    'repeat_before_max_diff': float((a-b).abs().max()),
                    'import_and_position_check_max_diff': float((b-c).abs().max()),
                    'repeat_after_max_diff': float((c-d).abs().max()),
                    'before_mean_vs_inference': float((a-target).abs().mean()),
                    'before_max_vs_inference': float((a-target).abs().max())})
    backward = {'performed': False}
    if args.backward:
        if not args.fused:
            raise RuntimeError('Backward regression uses the actual fused training path')
        actor.train()
        reference.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        reference.zero_grad(set_to_none=True)
        lp = forward()
        mask = data['loss_mask'][:, -max_response:].to(lp.device)
        loss = -(lp * mask).sum() / mask.sum()
        loss.backward()
        count, finite = 0, True
        for parameter in reference.parameters():
            if parameter.grad is not None:
                count += 1
                finite &= bool(torch.isfinite(parameter.grad).all())
        backward = {'performed': True, 'loss': float(loss.detach()), 'parameters_with_grad': count, 'all_finite': finite}
        if not finite or not count:
            raise RuntimeError('Actual fused backward has missing/nonfinite gradients')
        reference.zero_grad(set_to_none=True)
    cases = []
    for i, name in enumerate(names):
        count = len(generated['response_ids'][i])
        actual = action_lp[i, -count:]
        behavior = actual.new_tensor(generated['rollout_logprobs'][i])
        delta = (actual - behavior).abs()
        case = {'case': name, 'tokens': count, 'images': len(generated['image_grid_thw'][i]),
                'mean_abs_logprob_diff': float(delta.mean()), 'max_abs_logprob_diff': float(delta.max()),
                'training_logprobs': actual.tolist(), 'inference_logprobs': behavior.tolist()}
        case['within_diagnostic_tolerance'] = case['mean_abs_logprob_diff'] < .03 and case['max_abs_logprob_diff'] < .2
        cases.append(case)
    result = {'status': 'passed' if all(c['within_diagnostic_tolerance'] for c in cases) else 'failed',
              'contract_sha256': contract.sha256, 'source_capture': str(capture), 'wrapper': wrapper,
              'implementation': implementation,
              'actual_skyrl_preprocessing': True, 'actual_skyrl_model_forward': True,
              'token_mismatches': 0, 'image_mismatches': 0, 'cases': cases,
              'position_mismatches': position_mismatches,
              'stability': stability,
              'training_started': False, 'backward_performed': args.backward, 'backward': backward,
              'fused': args.fused, 'fsdp_tested': False,
              'full_training_parity_proven': False}
    if args.stability_check:
        stable = stability['before_flags'] == stability['after_flags'] and all(
            row[key] == 0. for row in stability['cases'] for key in
            ('repeat_before_max_diff', 'import_and_position_check_max_diff', 'repeat_after_max_diff'))
        result['stability_passed'] = stable
        if not stable:
            result['status'] = 'failed'
    atomic_bytes(output / 'result.json', canonical(result))
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
