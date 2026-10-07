"""Bounded two-GPU real native NCCL reload, including an in-flight request.

No optimizer, no task dataset, no checkpoint write. Reload the same base weights
through the exact SkyRL sender/dispatch and require cache-safe policy versions.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace as NS
import uuid

from .assets import from_assets
from .contract import MODEL_REVISION, canonical, digest
from .records import load_turn
from .runtime import configuration
from .store import TraceStore, atomic_bytes


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def check(root, contract, model_path):
    import httpx
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
    from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
    from skyrl.backends.skyrl_train.weight_sync.broadcast_strategy import BroadcastInitInfo, BroadcastTransferStrategy
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
    url = 'http://127.0.0.1:18081'
    client = RemoteInferenceClient(proxy_url=url, server_urls=[url], data_parallel_size=1)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port()}', rank=0, world_size=1)
    reference = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, local_files_only=True).eval()
    parameters = [(name, value.detach()) for name, value in reference.named_parameters()]
    metadata = {'names': [n for n, _ in parameters], 'dtype_names': [str(t.dtype).split('.')[-1] for _, t in parameters],
                'shapes': [list(t.shape) for _, t in parameters]}
    info = BroadcastInitInfo(master_addr='127.0.0.1', master_port=port(), rank_offset=1, world_size=2,
                             override_existing_receiver=False)
    sender_client = RemoteInferenceClient(proxy_url=url, server_urls=[url], data_parallel_size=1)
    def make_sender():
        torch.cuda.set_device(0)
        return BroadcastTransferStrategy.create_sender(info, sender_client)
    receiver, sender = await asyncio.gather(client.init_weight_update_communicator(info), asyncio.to_thread(make_sender))
    print('NCCL probe: real communicator ready', flush=True)

    def chunks():
        for start in range(0, len(parameters), 16):
            group = parameters[start:start + 16]
            yield WeightChunk(names=[n for n, _ in group], dtypes=[str(t.dtype) for _, t in group],
                              shapes=[list(t.shape) for _, t in group], tensors=[t for _, t in group])
    def broadcast(*args, **kwargs):
        def run():
            torch.cuda.set_device(0)
            asyncio.run(sender.send_chunks(chunks(), metadata))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=300)
    cfg = NS(trainer=NS(strategy='fsdp2', policy=NS(model=NS(lora=NS(rank=0))),
                        fully_async=NS(enabled=True)),
             generator=NS(inference_engine=NS(offload_kv_for_weight_sync=False, weight_sync_backend='nccl')))
    dispatch = NS(_inference_engine_client=client, colocate_all=False, cfg=cfg,
                  _prepare_for_weight_sync=lambda: None, _finish_weight_sync=lambda: None,
                  ensure_active_adapter=lambda *a: None, _broadcast_to_inference_engines=broadcast)
    picture = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(picture, format='PNG')
    content = [{'type': 'text', 'text': 'Describe this image in detail.'},
               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(picture.getvalue()).decode()}}]
    async with httpx.AsyncClient(timeout=600) as http:
        async def request(name):
            headers = {'x-ocrl-contract': contract.sha256, 'x-ocrl-attempt': name, 'x-ocrl-session': name,
                       'x-ocrl-agent': 'build', 'x-ocrl-primary': '1', 'x-ocrl-call': uuid.uuid4().hex, 'x-ocrl-turn': '0'}
            body = {'model': 'qwen35-9b-policy', 'stream': True, 'temperature': 1., 'top_p': 1.,
                    'max_tokens': 256, 'seed': 7, 'messages': [{'role': 'user', 'content': content}]}
            response = await http.post(url + '/v1/chat/completions', json=body, headers=headers)
            response.raise_for_status()
            atomic_bytes(root / f'{name}.sse', response.content)
        before_task = asyncio.create_task(request('before'))
        deadline = time.monotonic() + 120
        while (await http.get(url + '/ocrl/status')).json()['readers'] == 0:
            if before_task.done() or time.monotonic() > deadline:
                raise RuntimeError('Did not observe an in-flight generation')
            await asyncio.sleep(.01)
        sync = asyncio.create_task(WorkerDispatch.save_weights_for_sampler(dispatch))
        while (await http.get(url + '/ocrl/status')).json()['phase'] == 'open':
            if sync.done():
                await sync
                raise RuntimeError('Sync did not overlap generation')
            await asyncio.sleep(.01)
        queued = asyncio.create_task(request('queued'))
        await asyncio.gather(before_task, sync, queued)
        state = (await http.get(url + '/ocrl/status')).json()
        if state['version'] != f'{MODEL_REVISION}:1' or state['readers'] != 0:
            raise RuntimeError(f'Bad committed policy state: {state}')
    store = TraceStore(root / 'trace')
    turns = {}
    for name in ('before', 'queued'):
        records = store.attempt_requests(name)
        if len(records) != 1:
            raise RuntimeError('Unexpected request count')
        turns[name] = load_turn(store, records[0]['request_id'], contract)
    if turns['before']['input']['weight_version'] != f'{MODEL_REVISION}:0' or turns['queued']['input']['weight_version'] != f'{MODEL_REVISION}:1':
        raise RuntimeError('Generated request crossed the weight boundary')
    if turns['before']['input']['prompt_token_ids'] != turns['queued']['input']['prompt_token_ids']:
        raise RuntimeError('Identical request context changed across weight sync')
    before, after = turns['before']['response_ids'], turns['queued']['response_ids']
    same_ids = before == after
    delta = None
    if same_ids:
        delta = max(abs(a-b) for a,b in zip(turns['before']['rollout_logprobs'], turns['queued']['rollout_logprobs']))
    result = {'status': 'passed' if same_ids and delta < .2 else 'failed', 'same_checkpoint_reload': True,
              'actual_skyrl_dispatch': True, 'actual_nccl_transfer': True, 'parameters_sent': len(parameters),
              'bytes_sent': sum(t.numel()*t.element_size() for _,t in parameters), 'inflight_generation_observed': True,
              'same_response_ids': same_ids, 'max_logprob_diff': delta, 'state': state,
              'versions': {name: turn['input']['weight_version'] for name, turn in turns.items()},
              'training_started': False, 'optimizer_steps': 0}
    atomic_bytes(root / 'result.json', canonical(result))
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('Native reload numerical regression failed')


def main():
    # Match Ray's per-worker GPU visibility: the sender and every nested
    # asyncio.to_thread CUDA call see only their own physical GPU as cuda:0.
    # Exposing both GPUs lets native packed-transfer threads default to the
    # receiver's GPU, unlike the real isolated training worker process.
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    model = Path('/hf/hub/models--Qwen--Qwen3.5-9B/snapshots') / MODEL_REVISION
    cfg = configuration(context=8192, output=256, temperature=1., top_p=1., thinking=True)
    contract = from_assets(model, Path(__file__).parent, config=cfg, sandbox_sha256=digest({'probe': 'nccl'}),
                          tools_sha256=digest({'probe': 'no-tools'}), thinking=True, temperature=1., top_p=1., top_k=0,
                          max_context=8192, max_output=256)
    atomic_bytes(root / 'contract.json', canonical(contract.to_dict()))
    command = ['/opt/SkyRL/.venv/bin/python', '-m', 'opencode_rl.async_server', '--contract', str(root / 'contract.json'),
        '--trace-root', str(root / 'trace'), '--', '--model', str(model), '--served-model-name', 'qwen35-9b-policy',
        '--host', '127.0.0.1', '--port', '18081', '--max-model-len', '8192', '--max-num-seqs', '4',
        '--gpu-memory-utilization', '.45', '--dtype', 'bfloat16', '--enforce-eager', '--generation-config', 'vllm',
        '--override-generation-config', '{"top_k":0}', '--enable-prefix-caching', '--logprobs-mode', 'raw_logprobs',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', '{"enable_thinking":true}', '--weight-transfer-config', '{"backend":"nccl"}',
        '--worker-extension-cls', 'skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap',
        '--limit-mm-per-prompt', '{"image":4,"video":0}', '--no-enable-log-requests']
    with (root / 'server.log').open('wb') as log:
        process = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': '/opt:/opt/SkyRL'},
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            import urllib.request
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Lease server exited: ' + (root / 'server.log').read_text()[-8000:])
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Lease server startup')
                    time.sleep(1)
            asyncio.run(check(root, contract, model))
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
