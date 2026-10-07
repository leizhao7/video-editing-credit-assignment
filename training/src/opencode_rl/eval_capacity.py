"""Eight-GPU evaluation scheduling, independent of model-visible settings."""


def eight_gpu_plan(model):
    if model == 'qwen35-9b-policy':
        replicas, tp = 4, 2
    elif model == 'qwen36-27b-policy':
        replicas, tp = 2, 4
    else:
        raise ValueError('Unreviewed eight-GPU model')
    return {
        'gpus': 8, 'replicas': replicas, 'tp': tp, 'rollout_concurrency': 64,
        'tasks_per_engine': 64 // replicas, 'streams_per_engine': 32,
        'max_num_seqs': 64, 'max_num_batched_tokens': 65536,
        'gpu_memory_utilization': .90, 'mm_processor_cache_gb': 16,
        'ports': list(range(18081, 18081 + replicas)),
        'devices': [list(range(i * tp, (i + 1) * tp)) for i in range(replicas)],
        'cuda_graphs': True, 'prefix_caching': True,
    }
