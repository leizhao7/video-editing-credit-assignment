"""Reviewed 4i+4t serving limits; never edits model-visible messages or sampling."""


def training_capacity():
    # User-requested high-throughput profile. More independent tools can run
    # while the finite per-engine stream/token gates bound inference queues.
    return {'rollout_concurrency': 192, 'streams_per_engine': 40,
            'generation_workers': 32, 'preparation_concurrency': 8,
            'read_cache_bytes': 64 * 1024 * 1024,
            'engine_feature_preparation_concurrency': 16,
            'engine_active_requests': 32, 'engine_reserved_tokens': 2097152,
            'max_num_seqs': 64, 'max_num_batched_tokens': 65536,
            'mm_processor_cache_gb': 16,
            'long_prefill_token_threshold': 8192, 'gpu_memory_utilization': .90,
            'stream_interval': 32, 'judge_concurrency': 32}


def scheduler_args(capacity):
    # vLLM 0.26 explicitly rejects concurrent-partial-prefill >1, despite
    # exposing the flags. The supported per-request chunk cap is used instead.
    return ['--max-num-seqs', str(capacity['max_num_seqs']),
            '--max-num-batched-tokens', str(capacity['max_num_batched_tokens']),
            '--gpu-memory-utilization', str(capacity['gpu_memory_utilization']),
            '--mm-processor-cache-gb', str(capacity['mm_processor_cache_gb']),
            '--enable-chunked-prefill', '--long-prefill-token-threshold',
            str(capacity['long_prefill_token_threshold']), '--scheduler-reserve-full-isl',
            '--watermark', '.05', '--stream-interval', str(capacity['stream_interval'])]


def serving_metrics(body):
    import math
    selected = {'vllm:num_requests_running', 'vllm:num_requests_waiting',
                'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total',
                'vllm:prompt_tokens_total', 'vllm:generation_tokens_total'}
    result = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        key, raw = line.rsplit(' ', 1)
        if key.split('{', 1)[0] not in selected:
            continue
        value = float(raw)
        if math.isfinite(value):
            result[key] = value  # Preserve labels, never sum unrelated KV percentages.
    return result


async def sample_capacity(engines):
    import asyncio
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
        async def one(engine):
            result = {'url': engine['url']}
            try:
                async with client.get(engine['url'] + '/ocrl/status') as response:
                    response.raise_for_status()
                    result['admission'] = await response.json()
                async with client.get(engine['url'] + '/metrics') as response:
                    response.raise_for_status()
                    result['vllm'] = serving_metrics(await response.text())
            except Exception as exc:
                result['unavailable'] = type(exc).__name__  # not a model zero or success
            return result
        return await asyncio.gather(*(one(e) for e in engines))
