"""Dedicated native SkyRL/vLLM server with traced requests and weight leases.

Invoke with --contract/--trace-root followed by -- and the ordinary vLLM server
arguments. This is not a training launcher. The sandbox can reach only the
separate chat-only relay, never this server's administrative routes.
"""
import argparse
import json
import os
from pathlib import Path

from .contract import Contract, ParityError
from .store import TraceStore
from .vllm_hooks import install
from .weight_sync import AsyncPolicy, add_routes
from .visual_budget import vllm_image_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--trace-root', required=True)
    parser.add_argument('--weight-version', type=int, default=0)
    parser.add_argument('--active-requests', type=int, default=32)
    parser.add_argument('--reserved-tokens', type=int, default=2097152)
    parser.add_argument('--feature-preparation-concurrency', type=int, default=16)
    args, server_args = parser.parse_known_args()
    if server_args[:1] == ['--']:
        server_args = server_args[1:]
    contract = Contract(**json.loads(Path(args.contract).read_bytes()))
    if args.weight_version < 0:
        raise ParityError('Invalid initial policy version')
    if not 1 <= args.active_requests <= 64 or args.reserved_tokens < contract.max_context:
        raise ParityError('Admission must allow one full-context request without truncation')
    if not 1 <= args.feature_preparation_concurrency <= 64:
        raise ParityError('Feature preparation must remain bounded')
    os.environ['VLLM_SERVER_DEV_MODE'] = '1'
    from skyrl.backends.skyrl_train.inference_servers import vllm_server_actor as server
    options = server._build_standalone_cli_args(server_args)
    if options.host not in ('127.0.0.1', 'localhost'):
        raise ParityError('Weight-update server must be private; expose only the sandbox chat relay')
    if options.max_model_len != contract.max_context or options.logprobs_mode != 'raw_logprobs':
        raise ParityError('Serving context/logprob definition differs from contract')
    if options.generation_config != 'vllm' or options.override_generation_config.get('top_k') != contract.top_k:
        raise ParityError('Explicit neutral generation config and contract top_k required')
    if options.default_chat_template_kwargs.get('enable_thinking') is not contract.thinking:
        raise ParityError('Serving thinking mode differs from contract')
    if not options.enable_prefix_caching:
        raise ParityError('Audited async server requires explicit prefix-cache lifecycle')
    # Same user-selected 999-image admission as evaluation; do not inherit a
    # stale 192-image launcher setting. Avoid excessive dummy MM profiling.
    options.limit_mm_per_prompt = vllm_image_limits()
    options.skip_mm_profiling = True
    store = TraceStore(args.trace_root)
    policy = AsyncPolicy(f'{contract.model_revision}:{args.weight_version}', audit_root=store.root / 'weight_sync',
                         max_readers=args.active_requests, token_budget=args.reserved_tokens)
    install(store, contract, policy,
            feature_preparation_concurrency=args.feature_preparation_concurrency)
    original = server.VLLMServerActor._add_custom_endpoints

    def endpoints(app, engine, cli_args):
        original(app, engine, cli_args)
        add_routes(app, engine, policy)

    server.VLLMServerActor._add_custom_endpoints = staticmethod(endpoints)
    import asyncio
    asyncio.run(server._build_and_serve_vllm_server(options))


if __name__ == '__main__':
    main()
