"""Fixed-checkpoint diagnostic server; deliberately no RL weight-update routes."""
import argparse
import json
import os
from pathlib import Path

from .contract import Contract, ParityError
from .store import TraceStore
from .vllm_hooks import StaticPolicy, install
from .visual_budget import vllm_image_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--trace-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=18081)
    args = parser.parse_args()
    if os.environ.get('OCRL_FIXED_DIAGNOSTIC') != '1':
        raise ParityError('This server is not a production training entrypoint')
    contract = Contract(**json.loads(Path(args.contract).read_bytes()))
    os.environ['VLLM_SERVER_DEV_MODE'] = '0'
    install(TraceStore(args.trace_root), contract, StaticPolicy(contract.model_revision))
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    cli_env_setup()
    vparser = make_arg_parser(FlexibleArgumentParser())
    options = vparser.parse_args([
        '--model', args.model, '--served-model-name', 'qwen35-9b-policy', '--host', '127.0.0.1', '--port', str(args.port),
        '--max-model-len', str(contract.max_context), '--max-num-seqs', '2', '--gpu-memory-utilization', '0.45',
        '--dtype', 'bfloat16', '--enforce-eager', '--generation-config', 'vllm',
        '--override-generation-config', json.dumps({'top_k': contract.top_k}),
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', json.dumps({'enable_thinking': contract.thinking}),
        '--skip-mm-profiling', '--limit-mm-per-prompt', json.dumps(vllm_image_limits()),
        '--no-enable-log-requests',
    ])
    validate_parsed_serve_args(options)
    uvloop.run(run_server(options))


if __name__ == '__main__':
    main()
