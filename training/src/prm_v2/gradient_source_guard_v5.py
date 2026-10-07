"""Fail-closed, opt-in pins for the audited SkyRL gradient-loss closure.

Importing this module uses only the stdlib and does not import Torch, SkyRL,
Ray, or a runtime API. Verification only inspects ALREADY LOADED modules.
Pins cover the loss, IS/mask helpers, KL/reduction, native normalization and
unscaled FSDP backward. NativeMicro, tensor dispatch/denominators, FSDP layout,
configuration, third-party Torch operations and the probe callbacks must be
qualified by the caller; this is not a hostile-Python security boundary.

The only accepted alternative is the reviewed active-token IS *metrics* patch.
Function ASTs deliberately ignore formatting/comments, not executable code.
Runtime bytecode is also compared with the exact lexical code object compiled
from the complete original module, without executing that module. Python 3.12
uses module-import metadata when choosing call opcodes, so compiling a detached
function would reject valid imported source. The selected function still must
match its pinned AST, and replacing __code__ while retaining old source fails.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import sys
import textwrap
import types

PPO = 'skyrl.backends.skyrl_train.utils.ppo_utils'
TORCH_UTILS = 'skyrl.backends.skyrl_train.utils.torch_utils'
OFFPOLICY = 'skyrl.backends.skyrl_train.utils.off_policy_correction_utils'
TRAINER = 'skyrl.train.trainer'
STRATEGY = 'skyrl.backends.skyrl_train.distributed.fsdp_strategy'
WORKER = 'skyrl.backends.skyrl_train.workers.worker'

# Frozen against work/skyrl-src; regenerated only after a source review.
PINS = {
    PPO + ':rollout_is_policy_loss': ('93a9c4ebc7a1e5885743add5d0bf2c3adfb3fdfd3d8709c7102e66e0976f84ee',),
    PPO + ':reduce_loss': ('d4551605bd10c1b31c4390e023bd620e889a53888cc1586cd54491248576cd51',),
    PPO + ':compute_approx_kl': ('8b98871501f39e028505b9e107ac7b6d77655aa8302777924aed86bcd2289b53',),
    PPO + ':apply_loss_reduction_to_advantages_minibatch': ('272abca1f9c4faf578c96903a0fb7c9ad1def835005b7b10a5f69a0e70ef729d',),
    TORCH_UTILS + ':masked_mean': ('4672f89a12f960fcbbd3de8d2b10404fb8ad53aea16074b5eac4239285224c28',),
    TORCH_UTILS + ':safe_exp_delta': ('8b5b0b2a962c4146f0e1cba8a07791ff0eebbcee0056c07736d74563bc2e968d',),
    OFFPOLICY + ':off_policy_correction_enabled': ('c087c104f902cc1bb0d656b16ec0cdce92b54b8804d937d4d39a413b9c0f95dc',),
    OFFPOLICY + ':compute_tis_ratio': ('88c5e1e3c93254b3d38fdf8a362b5afda16d8be0fe101b8549769886ee236d44',),
    OFFPOLICY + ':compute_outlier_token_mask': ('9100d8072ec46d4fadad8922396011568aadd85df40e4a57d0c56deb0ac0da0b',),
    OFFPOLICY + ':compute_token_mask': ('ee036d8cfabf07b6a4e70547d5ef1b7e336a52f548613343b88e90fb2581edef',),
    OFFPOLICY + ':compute_sequence_mask': ('aaa6f574ef094b77ea51fbbc4dcfd26806ef8a8ab0545e7290f45a19e55cbada',),
    OFFPOLICY + ':compute_off_policy_correction': (
        '73136c96499aeafa9308d311b31544969a1543eec9107a41f1b4f14f2add8d08',
        '1d791abfcd29f7228eba622112cb686d99d64f17918765982197e82cab0c9ac7'),
    OFFPOLICY + ':apply_off_policy_correction': ('abdd57cc023575be63a4482486ae37b388f8319d22fda1ea286bb9d7f923a375',),
    TRAINER + ':RayPPOTrainer._normalize_advantages': ('7283fb76d87cb6c21a8121980fb307e4ffc091d2bb9c9abb28c23a759055eaca',),
    STRATEGY + ':FSDPStrategy.backward': ('160f32e1ade137194079cf0879331d686f93b0d50dddf532682cd7c1e7869013',),
}

_REGISTRY_WRAPPER = '''def factory(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
'''
_NO_GRAD_WRAPPER = '''def factory(ctx_factory, func):
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)
    return decorate_context
'''


class SourceGuardError(RuntimeError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def _tree(value):
    if isinstance(value, ast.AST):
        return [type(value).__name__, {key: _tree(item) for key, item in ast.iter_fields(value)
            if key not in ('type_comment',) and not (key == 'type_params' and not item)}]
    if isinstance(value, list):
        return [_tree(item) for item in value]
    return value


def function_ast(source, qualname):
    """Select a precise lexical path, not a search for a same-named function."""
    node = ast.parse(textwrap.dedent(source))
    for part in qualname.split('.'):
        matches = [item for item in node.body
                   if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and item.name == part]
        if len(matches) != 1:
            raise SourceGuardError('Missing or ambiguous function: ' + qualname)
        node = matches[0]
    if not isinstance(node, ast.FunctionDef):
        raise SourceGuardError('Expected ordinary Python function: ' + qualname)
    return node


def source_digest(source, qualname):
    return _hash(_tree(function_ast(source, qualname)))


def verify_source(key, source, qualname=None):
    """Offline source fixture check, useful without installing Torch/SkyRL."""
    if key not in PINS:
        raise SourceGuardError('Unreviewed source key: ' + key)
    digest = source_digest(source, qualname or key.rsplit(':', 1)[-1])
    if digest not in PINS[key]:
        raise SourceGuardError('Unreviewed function source: ' + key + ' sha256=' + digest)
    return digest


def _code_tree(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return _code_tree(value)
        if isinstance(value, bytes):
            return ['bytes', value.hex()]
        if isinstance(value, tuple):
            return ['tuple', [constant(item) for item in value]]
        if value is Ellipsis:
            return ['ellipsis']
        if isinstance(value, frozenset):
            return ['frozenset', sorted([constant(item) for item in value], key=repr)]
        if value is None or isinstance(value, (str, int, float, complex, bool)):
            return [type(value).__name__, repr(value)]
        raise SourceGuardError('Unknown code constant')
    return dict(code=code.co_code.hex(), constants=[constant(x) for x in code.co_consts],
                names=code.co_names, varnames=code.co_varnames, freevars=code.co_freevars,
                cellvars=code.co_cellvars, argcount=code.co_argcount,
                posonly=code.co_posonlyargcount, kwonly=code.co_kwonlyargcount,
                # Future annotation flags / lexical nesting do not affect execution.
                flags=code.co_flags & 0xFFFF & ~inspect.CO_NESTED,
                exceptions=getattr(code, 'co_exceptiontable', b'').hex())


def _compiled_code(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    compiled = compile(ast.fix_missing_locations(module), '<reviewed-gradient-source>', 'exec',
                       dont_inherit=True, optimize=sys.flags.optimize)
    return next(item for item in compiled.co_consts
                if isinstance(item, types.CodeType) and item.co_name == node.name)


def _compiled_module_code(source, qualname, *, filename='<reviewed-gradient-module>'):
    """Compile, never execute, the full module and select a unique lexical path.

    Imports, future declarations and enclosing classes are compiler context,
    not runtime dependencies: this does not import Torch/SkyRL or run decorators.
    Do not turn this into a recursive same-name search or detach the target AST.
    """
    function_ast(source, qualname)  # Reject missing/ambiguous AST lexical paths.
    code = compile(source, filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    path = []
    for part in qualname.split('.'):
        path.append(part)
        matches = [item for item in code.co_consts if isinstance(item, types.CodeType)
                   and item.co_name == part and item.co_qualname == '.'.join(path)]
        if len(matches) != 1:
            raise SourceGuardError('Missing or ambiguous runtime lexical code: ' + qualname)
        code = matches[0]
    return code


def _check_code(actual, expected, key):
    if _code_tree(actual) != _code_tree(expected):
        raise SourceGuardError('Runtime code differs from reviewed source: ' + key)


def _plain_function(fn, key):
    if type(fn) is not types.FunctionType or hasattr(fn, '__wrapped__') or fn.__closure__:
        raise SourceGuardError('Unexpected callable/wrapper/closure: ' + key)
    module_name, qualname = key.rsplit(':', 1)
    module = _loaded(module_name)
    # The key selects the trusted already-loaded namespace. A spoofed
    # fn.__module__ or co_filename alone cannot redirect source verification.
    if (fn.__globals__ is not vars(module) or fn.__module__ != module_name or
            fn.__qualname__ != qualname or fn.__code__.co_qualname != qualname):
        raise SourceGuardError('Runtime function lexical/module binding changed: ' + key)
    try:
        filename = inspect.getsourcefile(module)
        if filename is None or fn.__code__.co_filename != filename:
            raise SourceGuardError('Runtime function source filename changed: ' + key)
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise SourceGuardError('Runtime source unavailable: ' + key) from exc
    node = function_ast(source, qualname)
    digest = verify_source(key, source, qualname)
    _check_code(fn.__code__, _compiled_module_code(source, qualname, filename=filename), key)
    defaults = tuple(ast.literal_eval(x) for x in node.args.defaults) or None
    kwdefaults = {arg.arg: ast.literal_eval(value) for arg, value in
                  zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None} or None
    if fn.__defaults__ != defaults or fn.__kwdefaults__ != kwdefaults:
        raise SourceGuardError('Runtime argument defaults changed: ' + key)
    return digest


def _loaded(name):
    module = sys.modules.get(name)
    if not isinstance(module, types.ModuleType):
        raise SourceGuardError('Required runtime module is not already loaded: ' + name)
    return module


def _closures(fn):
    try:
        return dict(zip(fn.__code__.co_freevars,
                        [cell.cell_contents for cell in (fn.__closure__ or ())]))
    except ValueError as exc:
        raise SourceGuardError('Empty runtime wrapper closure') from exc


def _unwrap_one(fn, *, no_grad=False):
    if type(fn) is not types.FunctionType or type(getattr(fn, '__wrapped__', None)) is not types.FunctionType:
        raise SourceGuardError('Expected exactly one audited decorator')
    body = fn.__wrapped__
    template = _NO_GRAD_WRAPPER if no_grad else _REGISTRY_WRAPPER
    parent = _compiled_code(function_ast(template, 'factory'))
    code = next(item for item in parent.co_consts if isinstance(item, types.CodeType))
    _check_code(fn.__code__, code, 'no_grad wrapper' if no_grad else 'policy registry wrapper')
    closure = _closures(fn)
    expected_names = {'ctx_factory', 'func'} if no_grad else {'func'}
    if set(closure) != expected_names or closure['func'] is not body:
        raise SourceGuardError('Decorator forwards to an unexpected function')
    if fn.__defaults__ is not None or fn.__kwdefaults__ is not None:
        raise SourceGuardError('Unexpected decorator argument defaults')
    if no_grad:
        torch = _loaded('torch')
        factory = closure['ctx_factory']
        if (not inspect.ismethod(factory) or type(factory.__self__) is not torch.no_grad
                or factory.__func__ is not torch.no_grad.clone):
            raise SourceGuardError('Normalizer decorator is not native torch.no_grad')
    return body


def _identity(actual, expected, key):
    if actual is not expected:
        raise SourceGuardError('Runtime binding changed: ' + key)


def _globals(fn, dependencies):
    for name, expected in dependencies.items():
        _identity(fn.__globals__.get(name), expected, fn.__name__ + '.' + name)


def _receipt(hashes):
    result = dict(hashes)
    result['receipt_sha256'] = _hash(hashes)
    return result


def verify_worker(worker):
    """Check actual worker bindings without a forward pass, collectives or Ray API."""
    ppo, off, tensor = (_loaded(name) for name in (PPO, OFFPOLICY, TORCH_UTILS))
    strategy_module, worker_module, torch = (_loaded(name) for name in (STRATEGY, WORKER, 'torch'))
    _identity(worker.policy_loss_fn, ppo.rollout_is_policy_loss, 'worker.policy_loss_fn')
    loss = _unwrap_one(worker.policy_loss_fn)
    hashes = {PPO + ':rollout_is_policy_loss': _plain_function(loss, PPO + ':rollout_is_policy_loss')}
    for owner, names in ((ppo, ('reduce_loss', 'compute_approx_kl',
                                'apply_loss_reduction_to_advantages_minibatch')),
                         (tensor, ('masked_mean', 'safe_exp_delta')),
                         (off, ('off_policy_correction_enabled', 'compute_tis_ratio',
                                'compute_outlier_token_mask', 'compute_token_mask',
                                'compute_sequence_mask', 'compute_off_policy_correction',
                                'apply_off_policy_correction'))):
        for name in names:
            fn = getattr(owner, name)
            key = owner.__name__ + ':' + name
            hashes[key] = _plain_function(fn, key)
            # Verify only globals actually referenced by this function.
            dependencies = {'torch': torch, 'masked_mean': tensor.masked_mean,
                            'safe_exp_delta': tensor.safe_exp_delta}
            dependencies.update({n: getattr(off, n) for n in (
                'off_policy_correction_enabled', 'compute_tis_ratio', 'compute_outlier_token_mask',
                'compute_token_mask', 'compute_sequence_mask', 'compute_off_policy_correction')})
            _globals(fn, {n: obj for n, obj in dependencies.items() if n in fn.__code__.co_names})
    _globals(loss, dict(torch=torch, safe_exp_delta=tensor.safe_exp_delta,
                       masked_mean=tensor.masked_mean, reduce_loss=ppo.reduce_loss,
                       apply_off_policy_correction=off.apply_off_policy_correction))
    # NativeMicro's imports must be the same helpers, not stale/rebound copies.
    _identity(worker_module.compute_approx_kl, ppo.compute_approx_kl, 'NativeMicro.compute_approx_kl')
    _identity(worker_module.masked_mean, tensor.masked_mean, 'NativeMicro.masked_mean')
    method = worker.strategy.backward
    if not inspect.ismethod(method) or method.__self__ is not worker.strategy:
        raise SourceGuardError('Strategy backward is not a bound native method')
    _identity(method.__func__, strategy_module.FSDPStrategy.backward, 'strategy.backward')
    key = STRATEGY + ':FSDPStrategy.backward'
    hashes[key] = _plain_function(method.__func__, key)
    return _receipt(hashes)


def verify_normalizer(native_callable):
    """Pass super()._normalize_advantages, not the overlay's override."""
    trainer, ppo, torch = (_loaded(name) for name in (TRAINER, PPO, 'torch'))
    fn = native_callable.__func__ if inspect.ismethod(native_callable) else native_callable
    _identity(fn, trainer.RayPPOTrainer._normalize_advantages, 'native _normalize_advantages')
    body = _unwrap_one(fn, no_grad=True)
    key = TRAINER + ':RayPPOTrainer._normalize_advantages'
    hashes = {key: _plain_function(body, key)}
    name = 'apply_loss_reduction_to_advantages_minibatch'
    helper = getattr(ppo, name)
    hashes[PPO + ':' + name] = _plain_function(helper, PPO + ':' + name)
    _globals(body, {'torch': torch, name: helper})
    _globals(helper, {'torch': torch})
    return _receipt(hashes)
