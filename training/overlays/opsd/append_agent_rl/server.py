"""Private independent raw-token inference server; importing never launches."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from opencode_rl.contract import ParityError
from opencode_rl.store import TraceStore
from opencode_rl.vllm_hooks import install
from opencode_rl.weight_sync import AsyncPolicy, add_routes as weight_routes
from opencode_rl.visual_budget import vllm_image_limits
from .contract import AppendContract, loop_source_sha256
from .routes import add_routes

SOURCE_PINS = {
    'entrypoints/chat_utils.py': '16b9fd8c6ed75c8593778be2ddc5a56053259aec85f300063c607894381e28ec',
    'entrypoints/openai/chat_completion/protocol.py': 'aea283940efbe7b8c91aed6e4e146a9d7f714095cf46c6df3080f00e05d3277c',
    'model_executor/models/qwen3_vl.py': '4f31566372fca55b708cffcf24229bd4481d5e9f4a77994ee3a71ba29f85936d',
    'entrypoints/scale_out/token_in_token_out/serving.py': 'eec69b775af748e3bb59d83e6aebaf25aad4bc04d780b2a7d4ff7f8d792c8dad',
    'entrypoints/scale_out/token_in_token_out/protocol.py': 'b41ab47df0e95fa0e7ad01c9002f55e533523a38bd046beff3aa3558d02688a3',
    'entrypoints/scale_out/token_in_token_out/mm_serde.py': '31f30054faeeaf3ac4fa6c989df4eb7daaab55741dda33472e483ccc95929e02',
    'entrypoints/scale_out/derender/serving.py': '9d7d17a3efe926dadea835c74c95e4cdf6f0c9a7a27a7adb9cabbea7482b542e',
    'renderers/online_renderer.py': 'dbc8e032939cdb41c0c0c1f2e07d42c2bb6d9c59b16e5c195973f81826dcd5e9',
    'renderers/online_derenderer.py': '880bb39e29d943414cf7bb28d5c83e3f1abb4c41af9e5703f2fcad3a48a2dfff',
    'parser/engine/registered_adapters.py': '7badf1d4918917b08a82afd89fdd9201ff012a22838406fbfc2d739070d0ff08',
    'tool_parsers/qwen3_engine_tool_parser.py': '3cf83a2a9408d72c79082825464b2c4dea1147ff390289dfb8936c5501114be9',
}


def verify_sources(root):
    for name, sha in SOURCE_PINS.items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != sha:
            raise ParityError('Unreviewed raw-token serving source: ' + name)


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
    contract = AppendContract(**json.loads(Path(args.contract).read_bytes()))
    if contract.loop_sha256 != loop_source_sha256():
        raise ParityError('Unpinned append-loop source')
    if (args.weight_version < 0 or not 1 <= args.active_requests <= 64 or
            args.reserved_tokens < contract.max_context or not 1 <= args.feature_preparation_concurrency <= 64):
        raise ParityError('Invalid bounded serving capacity')
    import vllm
    verify_sources(Path(vllm.__file__).parent)
    os.environ['VLLM_SERVER_DEV_MODE'] = '1'
    from skyrl.backends.skyrl_train.inference_servers import vllm_server_actor as server
    options = server._build_standalone_cli_args(server_args)
    if (options.host not in ('127.0.0.1', 'localhost') or options.max_model_len != contract.max_context or
            options.logprobs_mode != 'raw_logprobs' or options.generation_config != 'vllm' or
            options.override_generation_config.get('top_k') != -1 or not options.enable_prefix_caching or
            options.default_chat_template_kwargs.get('enable_thinking') is not contract.thinking or
            options.tool_call_parser != 'qwen3_xml' or not options.enable_auto_tool_choice):
        raise ParityError('Serving options differ from the append-only Qwen contract')
    options.limit_mm_per_prompt = vllm_image_limits()
    options.skip_mm_profiling = True
    store = TraceStore(args.trace_root)
    policy = AsyncPolicy(f'{contract.model_revision}:{args.weight_version}', audit_root=store.root / 'weight_sync',
                         max_readers=args.active_requests, token_budget=args.reserved_tokens)
    install(store, contract, policy, feature_preparation_concurrency=args.feature_preparation_concurrency)
    original = server.VLLMServerActor._add_custom_endpoints
    def endpoints(app, engine, cli_args):
        original(app, engine, cli_args)
        weight_routes(app, engine, policy)
        add_routes(app, store, contract, policy, render_concurrency=args.feature_preparation_concurrency)
    server.VLLMServerActor._add_custom_endpoints = staticmethod(endpoints)
    asyncio.run(server._build_and_serve_vllm_server(options))


if __name__ == '__main__':
    main()
