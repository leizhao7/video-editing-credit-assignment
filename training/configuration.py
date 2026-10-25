"""Validate explicit operator-owned cloud configuration without connecting."""
from pathlib import Path
import os
import json
import re
import tomllib

PROFILES = ('grpo', 'prm_fixed', 'prm_decay', 'opsd_reference', 'opsd_suggestions')
FIELDS = {'app', 'profile', 'release_volume', 'model_volume', 'logs_volume',
          'checkpoint_volume', 'asr_cache', 'judge_secret', 'process_secret',
          'logger', 'wandb_secret', 'residency_limits', 'demonstrator_tokens_per_minute'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}')


def load_config(path=None):
    if path is None and os.environ.get('EDITING_TRAINING_CONFIG_JSON'):
        return validate_config(json.loads(os.environ['EDITING_TRAINING_CONFIG_JSON']))
    path = path or os.environ.get('EDITING_TRAINING_CONFIG')
    if not path:
        raise ValueError('Set EDITING_TRAINING_CONFIG to an explicit cloud TOML configuration')
    value = tomllib.loads(Path(path).expanduser().read_text())
    return validate_config(value)


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError('Cloud configuration must be an object')
    if set(value) != FIELDS:
        raise ValueError('Cloud config has missing or unknown fields: ' + str(sorted(set(value) ^ FIELDS)))
    if value['profile'] not in PROFILES or value['logger'] not in ('console', 'wandb'):
        raise ValueError('Unsupported training profile or logger')
    for key in ('app', 'release_volume', 'model_volume', 'logs_volume', 'checkpoint_volume', 'asr_cache', 'judge_secret'):
        if not isinstance(value[key], str) or not NAME.fullmatch(value[key]):
            raise ValueError('Invalid resource name: ' + key)
    volumes = [value[k] for k in ('release_volume', 'model_volume', 'logs_volume', 'checkpoint_volume')]
    if len(set(volumes)) != len(volumes):
        raise ValueError('Release, models, logs and checkpoints require distinct volumes')
    for key in ('process_secret', 'wandb_secret'):
        if not isinstance(value[key], str) or (value[key] and not NAME.fullmatch(value[key])):
            raise ValueError('Invalid secret name: ' + key)
    if (value['profile'].startswith('prm_') or value['profile'] == 'opsd_suggestions') and not value['process_secret']:
        raise ValueError('This profile needs an explicit process/demonstrator secret')
    if value['logger'] == 'wandb' and not value['wandb_secret']:
        raise ValueError('W&B logging needs an explicit W&B secret')
    limits = value['residency_limits']
    if not isinstance(limits, list) or len(limits) != 4 or any(type(n) is not int or not 1 <= n <= 128 for n in limits):
        raise ValueError('Supply four positive bounded residency limits')
    quota = value['demonstrator_tokens_per_minute']
    if type(quota) is not int or not 1 <= quota <= 100000000:
        raise ValueError('Invalid demonstrator input-token quota')
    return value
