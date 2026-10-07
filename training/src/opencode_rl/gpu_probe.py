"""Bounded fixed-base-checkpoint numerical probe; no policy optimization."""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from .assets import from_assets
from .contract import MODEL_REVISION, canonical, digest
from .records import load_turn
from .runtime import configuration
from .store import TraceStore, atomic_bytes
from .training import assemble_images, modality_ids


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    model = Path('/hf/hub/models--Qwen--Qwen3.5-9B/snapshots') / MODEL_REVISION
    context = 32768 if args.regression else 8192
    cfg = configuration(context=context, output=64, temperature=1, top_p=1., thinking=True)
    contract = from_assets(model, Path(__file__).parent, config=cfg,
        sandbox_sha256=digest({'diagnostic': 'direct-native-chat-no-tool-execution'}),
        tools_sha256=digest({'diagnostic': 'no-tools'}), thinking=True, temperature=1, top_p=1.,
        top_k=0, max_context=context, max_output=64)
    atomic_bytes(root / 'contract.json', canonical(contract.to_dict()))
    server_log = (root / 'server.log').open('wb')
    server = subprocess.Popen(['/opt/SkyRL/.venv/bin/python', '-m', 'opencode_rl.fixed_server',
        '--contract', str(root / 'contract.json'), '--trace-root', str(root / 'trace'), '--model', str(model)],
        env={**os.environ, 'PYTHONPATH': '/opt:/opt/SkyRL', 'OCRL_FIXED_DIAGNOSTIC': '1'},
        stdout=server_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Fixed diagnostic server exited {server.returncode}: ' +
                                   (root / 'server.log').read_text(errors='replace')[-8000:])
            try:
                with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Fixed diagnostic server did not become healthy')
            time.sleep(2)
        print('GPU probe: native server ready', flush=True)
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG')
        image = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        store = TraceStore(root / 'trace')
        turns = []
        cases = [
            ('text', 'Reply briefly with the word hello.'),
            ('image', [{'type': 'text', 'text': 'What color is this image? Reply briefly.'},
                       {'type': 'image_url', 'image_url': {'url': image}}]),
            # Same image again exercises vLLM processor/cache-hit provenance.
            ('image_cache', [{'type': 'text', 'text': 'Name the main color, briefly.'},
                             {'type': 'image_url', 'image_url': {'url': image}}]),
        ]
        if args.regression:
            cases.extend([
                ('generated_image', [{'type': 'text', 'text': 'Name the color.'},
                                     {'type': 'image_url', 'image_url': {'url': image}}]),
                ('long_text', 'Context:' + ' fact' * 16384 + '\nReply briefly.'),
                ('long_image', [{'type': 'text', 'text': 'Context:' + ' fact' * 8192 + '\nName the color.'},
                                {'type': 'image_url', 'image_url': {'url': image}}]),
            ])
        atomic_bytes(root / 'cases.json', canonical([name for name, _ in cases]))
        for name, content in cases:
            body = {'model': 'qwen35-9b-policy', 'stream': True, 'messages': [{'role': 'user', 'content': content}],
                    'temperature': 1., 'top_p': 1., 'max_tokens': 64}
            if name == 'generated_image':
                # Controlled regression: force a normally rare policy output.
                # The recorded logprobs remain RAW, before this diagnostic bias.
                body.update(logit_bias={str(contract.image_token_id): 100}, max_tokens=8)
            headers = {'Content-Type': 'application/json', 'x-ocrl-contract': contract.sha256,
                       'x-ocrl-attempt': name, 'x-ocrl-agent': 'build', 'x-ocrl-primary': '1',
                       'x-ocrl-session': name, 'x-ocrl-call': uuid.uuid4().hex, 'x-ocrl-turn': '0'}
            request = urllib.request.Request('http://127.0.0.1:18081/v1/chat/completions', data=canonical(body), headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                wire = response.read()
            atomic_bytes(root / f'{name}.sse', wire)
            recorded = store.attempt_requests(name)
            if len(recorded) != 1:
                raise RuntimeError('Expected one recorded native engine request')
            turn = load_turn(store, recorded[0]['request_id'], contract)
            pixels, grid = assemble_images(store, turn, contract)
            turns.append((name, turn, pixels, grid))
            print(f'GPU probe: {name} captured {len(turn["input"]["prompt_token_ids"])} prompt / {len(turn["response_ids"])} response tokens', flush=True)
        # Release inference GPU allocations before loading the reference model.
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()
    if args.capture_only:
        atomic_bytes(root / 'result.json', canonical({'status': 'captured', 'cases': [name for name, _ in cases],
                                                      'training_started': False}))
        return
    import torch
    from transformers import AutoModelForImageTextToText
    reference = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True, attn_implementation='sdpa').eval()
    results = []
    with torch.inference_mode():
        for name, turn, pixels, grid in turns:
            prompt, response = turn['input']['prompt_token_ids'], turn['response_ids']
            ids = torch.tensor([prompt + response], dtype=torch.long, device='cuda')
            kwargs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'use_cache': False}
            if grid.numel():
                kwargs.update(pixel_values=pixels.to('cuda', dtype=reference.dtype), image_grid_thw=grid.to('cuda'),
                              mm_token_type_ids=torch.tensor([modality_ids(turn)], dtype=torch.long, device='cuda'))
            outputs = reference(**kwargs)
            logits = outputs.logits[0, len(prompt)-1:len(prompt)+len(response)-1].float()
            lp = logits.log_softmax(-1).gather(-1, ids[0, len(prompt):].unsqueeze(-1)).squeeze(-1).cpu()
            observed = lp.new_tensor(turn['rollout_logprobs'])
            difference = (lp - observed).abs()
            result = {'case': name, 'prompt_tokens': len(prompt), 'response_tokens': len(response),
                'images': len(grid), 'max_abs_logprob_diff': float(difference.max()),
                'mean_abs_logprob_diff': float(difference.mean()),
                'reference_logprobs': lp.tolist(), 'inference_logprobs': observed.tolist()}
            # BF16 cross-engine kernel differences are numerical, not byte equality.
            result['within_diagnostic_tolerance'] = result['mean_abs_logprob_diff'] < .03 and result['max_abs_logprob_diff'] < .2
            results.append(result)
            del outputs, logits
    result = {'status': 'passed' if all(r['within_diagnostic_tolerance'] for r in results) else 'failed',
              'contract_sha256': contract.sha256, 'cases': results, 'model_revision': MODEL_REVISION,
              'training_started': False, 'actual_opencode_model_rollout': False,
              'full_training_parity_proven': False,
              'note': 'Native engine vs HF numerical diagnostic only; full SkyRL actor/async parity remains gated.'}
    atomic_bytes(root / 'result.json', canonical(result))
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
