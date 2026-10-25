"""Trusted CPU entry; uses the exact same stage graders as OpenCode validation."""
import json
from pathlib import Path
import shutil
import tempfile

from .contract import ParityError
from .grading_client import input_identity
from .train_dataset import SOURCE_FILES, sha_file


def service_contract(service):
    result = service._service_contract()
    result['grader_sha256'] = {**result['grader_sha256'],
        '@service': result['service_sha256'], '@stage_bridge': sha_file(__file__)}
    return result


def grade_stage(payload, service):
    import s12_task
    from grading_contract import submission_checks
    stage = payload['stage']
    if stage in ('s_3', 's_4e'):
        # Executes the newly baked v5 image, not a mutable named live service.
        judged = service.grade_judged
        result = (judged.get_raw_f() if hasattr(judged, 'get_raw_f') else judged)(payload)
        result['service_contract'] = service_contract(service)
        return result
    if stage not in ('s_1', 's_2'):
        raise ParityError('Unknown stage')
    source = Path(service._task_source(payload))
    trusted = json.loads((source / 'task.json').read_bytes())
    actual = str(trusted['stage']).split('/')[0].lower()
    if actual.replace('s1', 's_1').replace('s2', 's_2') != stage:
        raise ParityError('Trusted task stage mismatch')
    raw = str(payload.get('deliverable_json') or '')
    result = {'format': 's12-remote-grade-result-v1', 'task': payload['task'], 'stage': stage,
              'service_contract': service_contract(service), 'judge_files': {},
              'input_contract': {**input_identity(payload), 'source_sha256': {
                  n: sha_file(source / n) for n in SOURCE_FILES if (source / n).is_file()}}}
    with tempfile.TemporaryDirectory(prefix='ocrl-deterministic-') as temp:
        folder = Path(temp)
        for name in ('task.json', 'task_prompt.md', 'probe.json'):
            shutil.copy2(source / name, folder / name)
        shutil.copytree(source / 'verifier', folder / 'verifier')
        (folder / 'initial_state').symlink_to(source / 'initial_state', target_is_directory=True)
        (folder / 'output').mkdir()
        try:
            def invalid_constant(value):
                raise ValueError(f'Non-finite JSON constant: {value}')
            decoded = json.loads(raw, parse_constant=invalid_constant)
            if submission_checks():
                s12_task.validate_deliverable_data(decoded, stage)
                key = 'selected' if stage == 's_1' else 'order'
                if len(decoded[key]) != len(set(decoded[key])):
                    raise ValueError(f'Duplicate clip IDs in {key}')
        except (ValueError, TypeError) as exc:
            return {**result, 'score': 0., 'invalid_submission': True, 'invalid_reason': str(exc)}
        (folder / 'output' / s12_task.out_name(stage)).write_text(raw)
        score, detail = s12_task.grade(str(folder), stage)
        return {**result, 'score': score, 'detail': detail, 'invalid_submission': False}
