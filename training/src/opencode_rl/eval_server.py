"""Fixed weights, production-sized evaluation with the RL native token recorder."""
import argparse
import json
import os
from pathlib import Path
from .eval_contract import load_contract
from .store import TraceStore
from .vllm_hooks import StaticPolicy, install
from .visual_budget import vllm_image_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--trace-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--max-num-batched-tokens', type=int, default=16384)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--prefix-caching', action='store_true')
    parser.add_argument('--mm-processor-cache-gb', type=int, default=4)
    parser.add_argument('--disable-custom-all-reduce', action='store_true')
    args = parser.parse_args()
    if (not 1 <= args.max_num_seqs <= 128 or not 8192 <= args.max_num_batched_tokens <= 65536
            or not .5 <= args.gpu_memory_utilization <= .92 or args.port not in range(18081, 18085)
            or not 0 <= args.mm_processor_cache_gb <= 16):
        parser.error('Serving capacity outside reviewed range')
    contract = load_contract(json.loads(Path(args.contract).read_bytes()))
    os.environ['VLLM_SERVER_DEV_MODE'] = '0'
    install(TraceStore(args.trace_root), contract, StaticPolicy(contract.model_revision))
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    cli_env_setup()
    options = make_arg_parser(FlexibleArgumentParser()).parse_args([
        '--model', args.model, '--served-model-name', args.name, '--host', '127.0.0.1', '--port', str(args.port),
        '--tensor-parallel-size', str(args.tp), '--max-model-len', str(contract.max_context),
        '--max-num-seqs', str(args.max_num_seqs), '--max-num-batched-tokens', str(args.max_num_batched_tokens),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--dtype', 'bfloat16', *([] if args.cuda_graphs else ['--enforce-eager']),
        *(['--enable-prefix-caching'] if args.prefix_caching else []),
        *(['--disable-custom-all-reduce'] if args.disable_custom_all_reduce else []),
        '--mm-processor-cache-gb', str(args.mm_processor_cache_gb),
        '--skip-mm-profiling', '--generation-config', 'vllm',
        '--override-generation-config', json.dumps({'top_k': contract.top_k}),
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', json.dumps({'enable_thinking': contract.thinking}),
        '--limit-mm-per-prompt', json.dumps(vllm_image_limits()), '--no-enable-log-requests',
    ])
    validate_parsed_serve_args(options)
    uvloop.run(run_server(options))


if __name__ == '__main__':
    main()
