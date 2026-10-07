"""Build a contract from actual, local, pinned runtime/model assets."""
from pathlib import Path
import hashlib
import json

from .contract import Contract, MODEL_REVISION, ParityError, digest


def files_hash(root, names):
    root = Path(root)
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    if not files:
        raise ParityError('Cannot hash an empty asset set')
    return digest(files)


def from_assets(model_dir, package_dir, *, config, sandbox_sha256, tools_sha256,
                thinking, temperature, top_p, top_k, max_context, max_output):
    root, package = Path(model_dir), Path(package_dir)
    if root.name != MODEL_REVISION:
        raise ParityError('Expected the reviewed base model snapshot, not an unidentified checkpoint')
    cfg = json.loads((root / 'config.json').read_bytes())
    vision = cfg['vision_config']
    tokenizer_files = [p.name for p in root.iterdir()
                       if p.name in ('tokenizer.json', 'tokenizer_config.json', 'added_tokens.json', 'special_tokens_map.json')]
    if not {'tokenizer.json', 'tokenizer_config.json'} <= set(tokenizer_files):
        raise ParityError('Incomplete tokenizer snapshot')
    templates = [p.name for p in root.iterdir() if 'chat_template' in p.name]
    if not templates:
        templates = ['tokenizer_config.json']
    processor_files = [p.name for p in root.iterdir()
                       if p.name in ('preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json', 'config.json')]
    if 'preprocessor_config.json' not in processor_files:
        raise ParityError('Incomplete processor snapshot')
    return Contract(binary_sha256=hashlib.sha256(Path('/usr/local/bin/opencode').read_bytes()).hexdigest(),
        config_sha256=digest(config), tokenizer_sha256=files_hash(root, tokenizer_files),
        template_sha256=files_hash(root, templates), processor_sha256=files_hash(root, processor_files),
        tools_sha256=tools_sha256, sandbox_sha256=sandbox_sha256,
        integration_sha256=files_hash(package, [p.name for p in package.iterdir() if p.suffix in ('.py', '.mjs')]),
        max_context=max_context, max_output=max_output, thinking=thinking, temperature=temperature,
        top_p=top_p, top_k=top_k, image_token_id=cfg['image_token_id'],
        vision_patch_dim=vision['in_channels'] * vision['temporal_patch_size'] * vision['patch_size'] ** 2,
        spatial_merge_size=vision['spatial_merge_size'])
