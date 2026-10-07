"""Native preprocessing cache without order-dependent, tensor-less IPC hits.

The recorder deliberately yields before EngineCore admission to persist input
tensors. vLLM's sender/receiver cache assumes that preprocessing and admission
keep the same order, which is not true across those yields (or cancellation).
Use vLLM's own bounded processor-only cache in the API process instead. Hits
reuse the exact processed tensors, not a second image preprocessing pass, and
every EngineCore request is self-contained. Receiver caching can remain enabled:
the reviewed receiver accepts full items on both hits and misses.
"""
from functools import wraps
import threading

from .contract import ParityError


def install_cache_hooks(registry_type, renderer_type):
    """Called only after vLLM source verification, before renderer construction."""
    if getattr(registry_type, '_ocrl_self_contained_mm', False):
        raise ParityError('Multimodal cache hooks already installed')
    original_cache = registry_type.processor_cache_from_config
    processor_only = registry_type.processor_only_cache_from_config
    original_process = renderer_type._process_multimodal
    original_clear = renderer_type.clear_mm_cache
    # v0.26 runs MM preprocessing in the renderer thread pool. Its LRU lookup,
    # HF processing and merge form one transaction, not independent operations:
    # otherwise another request can evict an item between is_cached and get.
    # No I/O coroutine or engine generation is awaited while holding this lock.
    # Tokenization and the asyncio event loop remain outside it.
    processing_lock = threading.RLock()

    @wraps(original_cache)
    def cache(registry, config):
        mm = config.model_config.multimodal_config
        if mm is not None and mm.mm_processor_cache_gb > 0 and mm.mm_processor_cache_type != 'lru':
            raise ParityError('Self-contained image provenance requires reviewed LRU caching')
        return processor_only(registry, config)

    @wraps(original_process)
    def process(renderer, *args, **kwargs):
        with processing_lock:
            return original_process(renderer, *args, **kwargs)

    @wraps(original_clear)
    def clear(renderer, *args, **kwargs):
        with processing_lock:
            return original_clear(renderer, *args, **kwargs)

    registry_type.processor_cache_from_config = cache
    renderer_type._process_multimodal = process
    renderer_type.clear_mm_cache = clear
    registry_type._ocrl_self_contained_mm = True

    def restore():
        registry_type.processor_cache_from_config = original_cache
        renderer_type._process_multimodal = original_process
        renderer_type.clear_mm_cache = original_clear
        del registry_type._ocrl_self_contained_mm
    return restore
