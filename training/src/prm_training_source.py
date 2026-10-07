"""Workspace-owned, explicit source packaging for the opt-in append PRM image.

No Modal import, deployment, credentials or API calls. Hashes cover the complete
contract-relevant final packages, not only overlays. The runtime allowlist is
deliberately narrow: no tests, experiment inputs, downloads or analysis reports.
"""
from pathlib import Path
import hashlib
import json
import re

VERSION = 'append-prm-training-source-v1'
MANIFEST_NAME = 'prm_training_source_manifest.json'
RUNTIME_FILES = (
    'judge.py', 'judge_v3.py', 'judge_v5.py', 'context_v5.py', 'provenance_v5.py',
    'artifact_facts.py', 'numeric_facts.py', 'evidence_context.py', 'dataset.py',
    'transport.py', 'transport_v3.py', 'live_config_v5.py', 'live_packet_v5.py',
    'live_judge_v5.py', 'live_metrics_v5.py', 'gradient_probe_v5.py',
    'gradient_source_guard_v5.py', 'sandbox_admission_v5.py', 'cloud_preflight_v5.py',
)
OVERLAY_FILES = (
    'append_agent_rl/loop.py', 'append_agent_rl/controller.py',
    'append_agent_rl/task_rollout.py', 'append_agent_rl/generator.py',
    'append_agent_rl/train_entrypoint.py', 'append_agent_rl/launch.py',
    'opencode_rl/train_config.py', 'opencode_rl/train_entrypoint.py',
    'opencode_rl/train_generator.py',
    'opencode_rl/training.py', 'opencode_rl/prefix_transport.py',
    'opencode_rl/skyrl_adapter.py', 'opencode_rl/prefix_backward.py',
)
SUPPORT_FILES = ('prm_training_source.py', 'modal_append_prm_train.py')
SHELL = '/opt/append_timeout_bin/bash'
OWNER_SHA = '9b5d05aee1f07a323e80dc0980a6fe165b8fb8ba613fd9870ce42c1a3e8ceb38'
AUXILIARY_FILES = {'append_timeout_bin/bash': {
    'source': 'append_memory_timeout_command_owner.py', 'sha256': OWNER_SHA, 'mode': 0o555}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or symlinked source: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate upstream source key: ' + key)
        result[key] = value
    return result


def _validate_upstream_hashes(value):
    if not isinstance(value, dict) or set(value) != set(OVERLAY_FILES):
        raise ValueError('Upstream source hash keys differ from the exact overlay set')
    if any(not isinstance(sha, str) or not re.fullmatch(r'[a-f0-9]{64}', sha)
           for sha in value.values()):
        raise ValueError('Invalid upstream source SHA256')
    return dict(sorted(value.items()))


def _package_sources(root, package):
    root = Path(root)
    if package == 'append_agent_rl':
        files = [p for p in root.rglob('*') if p.is_file() and
                 p.suffix in ('.py', '.ts') and 'tests' not in p.relative_to(root).parts
                 and not p.name.endswith('.test.ts')]
    elif package == 'opencode_rl':
        files = [p for p in root.iterdir() if p.is_file() and p.suffix in ('.py', '.mjs')]
    else:
        raise ValueError('Unknown source package')
    if not files:
        raise ValueError('Empty source package: ' + package)
    return {str(p.relative_to(root)): p for p in sorted(files)}


def build_spec(project, work):
    """Return immutable expected manifest + explicit (local, remote) mounts.

    Call only when all owned overlays are complete. No filesystem writes. A
    missing overlay is a hard failure, never a silent fallback to legacy source.
    """
    project, work = Path(project), Path(work)
    overlay = work / 'prm_training_overlay'
    actual_overlay = {str(p.relative_to(overlay)) for p in overlay.rglob('*.py')
                      if '__pycache__' not in p.parts}
    if actual_overlay != set(OVERLAY_FILES):
        raise ValueError('Overlay set differs from reviewed explicit allowlist')
    upstream_path = overlay / 'upstream_hashes.json'
    _hash(upstream_path)  # Require a real, non-symlinked provenance file.
    upstream = _validate_upstream_hashes(json.loads(upstream_path.read_bytes(),
        object_pairs_hook=_unique_json_pairs))
    for name, expected in upstream.items():
        if _hash(project / name) != expected:
            raise ValueError('Upstream source changed; rebase/review overlay before packaging: ' + name)
    mounts, files, package_hashes = [], {}, {}
    for name in OVERLAY_FILES:
        source = overlay / name
        _hash(source)
        mounts.append((str(source), '/opt/' + name))
    for package in ('append_agent_rl', 'opencode_rl'):
        sources = _package_sources(project / package, package)
        for name in OVERLAY_FILES:
            prefix = package + '/'
            if name.startswith(prefix):
                sources[name[len(prefix):]] = overlay / name
        hashes = {name: _hash(source) for name, source in sorted(sources.items())}
        package_hashes[package] = digest(hashes)
        files.update({package + '/' + name: sha for name, sha in hashes.items()})
    for name in RUNTIME_FILES:
        source = work / 'prm_v2' / name
        files['prm_v2/' + name] = _hash(source)
        mounts.append((str(source), '/opt/prm_v2/' + name))
    # prm_v2 is intentionally a namespace package; no generated __init__ needed.
    for name in SUPPORT_FILES:
        files[name] = _hash(work / name)
        mounts.append((str(work / name), '/opt/' + name))
    for name, item in AUXILIARY_FILES.items():
        sha = _hash(work / item['source'])
        if sha != item['sha256']:
            raise ValueError('Qualified Bash owner source changed')
        files[name] = sha
        mounts.append((str(work / item['source']), '/opt/' + name))
    manifest = {'version': VERSION, 'files': dict(sorted(files.items())),
        'package_hashes': package_hashes, 'overlay_files': list(OVERLAY_FILES),
        'runtime_files': list(RUNTIME_FILES), 'support_files': list(SUPPORT_FILES),
        'auxiliary_files': {name: dict(item) for name, item in AUXILIARY_FILES.items()},
        'upstream_hashes': upstream,
        'scope': 'Exact final append/opencode contract sources plus explicit PRM runtime and launcher; no model/data bytes.'}
    return {'manifest': manifest, 'sha256': digest(manifest), 'mounts': mounts}


def verify_manifest(manifest, *, root='/opt'):
    """Check mounted bytes and exact contract source sets against expected hash."""
    root = Path(root)
    if not isinstance(manifest, dict) or manifest.get('version') != VERSION:
        raise ValueError('Unsupported PRM source manifest')
    # The remote image has only the final overlay, not the original project.
    # Preserve validated provenance without pretending to re-read upstream there.
    _validate_upstream_hashes(manifest.get('upstream_hashes'))
    if (manifest.get('overlay_files') != list(OVERLAY_FILES) or
            manifest.get('runtime_files') != list(RUNTIME_FILES) or
            manifest.get('support_files') != list(SUPPORT_FILES) or
            manifest.get('auxiliary_files') != AUXILIARY_FILES):
        raise ValueError('Source allowlist changed')
    required = ({'prm_v2/' + name for name in RUNTIME_FILES} | set(SUPPORT_FILES) |
                set(OVERLAY_FILES) | set(AUXILIARY_FILES))
    if not isinstance(manifest.get('files'), dict) or not required <= set(manifest['files']):
        raise ValueError('Incomplete source manifest')
    for name, sha in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or _hash(root / relative) != sha:
            raise ValueError('Mounted PRM source changed: ' + name)
    for package in ('append_agent_rl', 'opencode_rl'):
        sources = _package_sources(root / package, package)
        actual = {name: _hash(path) for name, path in sources.items()}
        expected = {name[len(package) + 1:]: sha for name, sha in manifest['files'].items()
                    if name.startswith(package + '/')}
        if actual != expected or digest(actual) != manifest['package_hashes'][package]:
            raise ValueError('Final contract source set changed: ' + package)
    actual_prm = {p.name for p in (root / 'prm_v2').iterdir() if p.suffix == '.py'}
    if actual_prm != set(RUNTIME_FILES):
        raise ValueError('Unexpected PRM module in image')
    for name, item in AUXILIARY_FILES.items():
        if ((root / name).stat().st_mode & 0o777) != item['mode']:
            raise ValueError('Qualified Bash owner executable mode changed')
    return {'source_sha256': digest(manifest), **manifest['package_hashes']}


def install_manifest(manifest, root='/opt'):
    """Image-build function: verify first, write one immutable receipt afterward."""
    verified = verify_manifest(manifest, root=root)
    target = Path(root) / MANIFEST_NAME
    raw = canonical(manifest)
    try:
        with target.open('xb') as handle:
            handle.write(raw)
    except FileExistsError:
        if target.read_bytes() != raw:
            raise ValueError('Different source manifest already installed')
    return verified


def verify_runtime(plan, *, root='/opt'):
    """Public GPU/Ray/grade guard; binds on-disk bytes to immutable plan."""
    manifest = json.loads((Path(root) / MANIFEST_NAME).read_bytes())
    result = verify_manifest(manifest, root=root)
    if (plan.get('process_source_sha256') != result['source_sha256'] or
            plan.get('process_source_manifest') != manifest):
        raise ValueError('Plan differs from mounted PRM source receipt')
    contract = plan['contract']
    if (contract['integration_sha256'] != result['opencode_rl'] or
            contract['loop_sha256'] != result['append_agent_rl']):
        raise ValueError('Contract was not generated from the final overlay image')
    if plan.get('config', {}).get('shell') != SHELL:
        raise ValueError('Plan omitted the qualified command-owner shell')
    _verify_import_origins(root)
    return result


def _verify_import_origins(root):
    """Reject /patch or site-packages shadowing the manifest's /opt modules."""
    import sys
    for name, module in tuple(sys.modules.items()):
        package = name.split('.')[0]
        if package not in ('append_agent_rl', 'opencode_rl', 'prm_v2'):
            continue
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to((Path(root) / package).resolve()):
            raise ValueError('Import shadowed verified runtime: ' + name)


def mounted_manifest(root='/opt'):
    manifest = json.loads((Path(root) / MANIFEST_NAME).read_bytes())
    verify_manifest(manifest, root=root)
    return manifest
