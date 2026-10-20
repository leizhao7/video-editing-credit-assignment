"""Synthetic CPU regressions; never import the real Modal SDK or run media tools."""
import importlib.util
import inspect
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ['modal_tn_generic', 'modal_transcribe', 'modal_stems', 'modal_film_transcribe']


@pytest.fixture
def load(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected network or external process')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setenv('VIDEO_EDITING_MODAL_VOLUME', 'synthetic-user-volume')
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs): return cls()
        def apt_install(self, *args, **kwargs): return self
        def pip_install(self, *args, **kwargs): return self
        def env(self, *args, **kwargs): return self
    class App:
        def __init__(self, name): pass
        def function(self, **kwargs): return lambda fn: fn
        def local_entrypoint(self, **kwargs): return lambda fn: fn
    volume = NS(commit=Mock())
    def from_name(name, **kwargs):
        assert name == 'synthetic-user-volume' and kwargs == {'create_if_missing': False}
        return volume
    monkeypatch.setitem(sys.modules, 'modal', NS(Image=Image, App=App, Volume=NS(from_name=from_name)))
    def get(name):
        spec = importlib.util.spec_from_file_location('_synthetic_' + name, ROOT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.DATA_ROOT = str(tmp_path / 'volume')
        Path(module.DATA_ROOT).mkdir(exist_ok=True)
        return module
    return get


@pytest.mark.parametrize('name', FILES)
def test_import_is_declaration_only_and_pipeline_inputs_remain_required(load, name):
    module = load(name)
    fn = getattr(module, dict(zip(FILES, ['detect', 'run', 'split', 'transcribe']))[name])
    key = 'slug' if name == 'modal_film_transcribe' else 'film'
    assert inspect.signature(fn).parameters[key].default is inspect.Parameter.empty
    module.vol.commit.assert_not_called()


@pytest.mark.parametrize('name', FILES)
@pytest.mark.parametrize('value', ['', '..', '../outside', '/absolute', 'nested/file', 'nested\\file'])
def test_unsafe_identifiers_rejected_before_processing(load, name, value):
    module = load(name)
    with pytest.raises(ValueError):
        if name == 'modal_tn_generic': module.detect(value, 'source')
        elif name == 'modal_film_transcribe': module.transcribe(value)
        elif name == 'modal_transcribe': module.run(value)
        else: module.split(value)
    module.vol.commit.assert_not_called()


@pytest.mark.parametrize('name', FILES)
def test_symlink_cannot_escape_volume(load, tmp_path, name):
    module = load(name)
    outside = tmp_path / 'outside'; outside.mkdir()
    (Path(module.DATA_ROOT) / 'alias').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): module._data_path('alias', 'output.json')


@pytest.mark.parametrize('name', FILES)
def test_metadata_write_is_atomic_and_closes_temporary_file(load, tmp_path, name):
    module = load(name)
    target = tmp_path / 'metadata.json'; target.write_text('{"previous": true}')
    with pytest.raises(ValueError): module._write_json(target, {'bad': float('nan')})
    assert json.loads(target.read_text()) == {'previous': True}
    assert not list(tmp_path.glob('.metadata-*'))
    module._write_json(target, {'segments': [], 'n_words': 0})
    assert json.loads(target.read_text()) == {'segments': [], 'n_words': 0}


@pytest.mark.parametrize('bad', ['', '\n,\n', '0/0', '0', '-24', 'nan', 'inf', 'invalid'])
def test_invalid_frame_rate_rejected(load, bad):
    with pytest.raises(ValueError): load('modal_tn_generic')._parse_fps(bad)


def test_fraction_frame_rate_and_trailing_csv_fields(load):
    assert load('modal_tn_generic')._parse_fps('24000/1001,\n\n') == pytest.approx(24000/1001)


class Pipe:
    def __init__(self, chunks): self.chunks = iter(chunks); self.closed = False
    def read(self, count):
        value = next(self.chunks, b'')
        if isinstance(value, Exception): raise value
        return value
    def close(self): self.closed = True


class Process:
    def __init__(self, chunks, code=0):
        self.stdout = Pipe(chunks); self.code = code; self.finished = False; self.killed = False
    def wait(self): self.finished = True; return self.code
    def poll(self): return self.code if self.finished else None
    def kill(self): self.killed = True


def test_decoder_preserves_partial_pipe_reads(load, monkeypatch):
    module = load('modal_tn_generic')
    raw = np.arange(2*48*27*3, dtype=np.uint8).tobytes()
    process = Process([raw[:5], raw[5:4000], raw[4000:]])
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: process)
    frames = module._decode_frames('synthetic-input', '')
    assert frames.shape == (2, 27, 48, 3) and frames.tobytes() == raw
    assert process.finished and process.stdout.closed


@pytest.mark.parametrize('chunks,code,error', [([], 0, RuntimeError), ([b'incomplete'], 0, RuntimeError),
                                            ([], 2, subprocess.CalledProcessError),
                                            ([OSError('pipe failed')], 0, OSError)])
def test_decoder_failure_is_reported_and_child_reaped(load, monkeypatch, chunks, code, error):
    module = load('modal_tn_generic'); process = Process(chunks, code)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: process)
    with pytest.raises(error): module._decode_frames('synthetic-input', '')
    assert process.finished and process.stdout.closed
    if chunks and isinstance(chunks[0], Exception): assert process.killed


@pytest.mark.parametrize('name', ['modal_transcribe', 'modal_film_transcribe'])
def test_empty_transcription_is_valid_schema_not_an_index_error(load, monkeypatch, name):
    module = load(name)
    film = Path(module.DATA_ROOT) / 'synthetic'; film.mkdir()
    if name == 'modal_film_transcribe': (film / 'source_audio.opus').touch()
    calls, temporary = [], []
    class Whisper:
        def __init__(self, model, **kwargs):
            assert model == 'large-v3' and kwargs == {'device': 'cuda', 'compute_type': 'float16'}
        def transcribe(self, wav, **kwargs):
            assert kwargs == {'word_timestamps': True, 'vad_filter': True, 'beam_size': 5, 'language': 'en'}
            return iter([]), NS(language='en', language_probability=1.)
    monkeypatch.setitem(sys.modules, 'faster_whisper', NS(WhisperModel=Whisper))
    def run(command, **kwargs):
        assert kwargs['check'] is True
        calls.append(command); temporary.append(Path(command[-1]).parent)
        return NS(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    result = module.run('synthetic') if name == 'modal_transcribe' else module.transcribe('synthetic')
    path = film / ('golden_transcript.json' if name == 'modal_transcribe' else 'source_transcript.json')
    output = json.loads(path.read_text())
    assert output['segments'] == [] and output['n_words'] == output['n_segments'] == 0
    assert result['words'] == 0 and all(not p.exists() for p in temporary)
    module.vol.commit.assert_called_once()


@pytest.mark.parametrize('name', ['modal_transcribe', 'modal_film_transcribe', 'modal_stems'])
def test_ffmpeg_failure_cleans_temp_and_does_not_commit(load, monkeypatch, name):
    module = load(name)
    film = Path(module.DATA_ROOT) / 'synthetic'; film.mkdir()
    if name == 'modal_film_transcribe': (film / 'source_audio.opus').touch()
    monkeypatch.setitem(sys.modules, 'faster_whisper', NS(WhisperModel=Mock()))
    temporary = []
    def fail(command, **kwargs):
        assert kwargs['check'] is True; temporary.append(Path(command[-1]).parent)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        if name == 'modal_transcribe': module.run('synthetic')
        elif name == 'modal_stems': module.split('synthetic')
        else: module.transcribe('synthetic')
    assert all(not p.exists() for p in temporary)
    assert not list(film.glob('*.json'))
    module.vol.commit.assert_not_called()


def test_stems_require_all_four_outputs_before_publishing(load, monkeypatch):
    module = load('modal_stems')
    def run(command, **kwargs):
        assert kwargs['check'] is True
        if 'demucs' in command:
            target = Path(command[command.index('-o')+1]) / 'htdemucs' / 'mix'
            target.mkdir(parents=True); (target / 'vocals.wav').touch()
        return NS(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='four stems'): module.split('synthetic')
    assert not (Path(module.DATA_ROOT) / 'synthetic/golden_stems').exists()
    module.vol.commit.assert_not_called()


def test_remote_batch_reports_empty_or_failed_jobs(load):
    module = load('modal_film_transcribe')
    module.list_ready = NS(remote=lambda: [])
    with pytest.raises(RuntimeError, match='No source'): module.main()
    module.list_ready = NS(remote=lambda: ['synthetic'])
    module.transcribe = NS(starmap=lambda jobs: [{'slug': 'synthetic', 'ok': False}])
    with pytest.raises(RuntimeError, match='failed'): module.main()


def test_local_cli_rejects_missing_or_conflicting_action(load):
    module = load('modal_film_transcribe')
    for args in ([], ['--upload', '--fetch', '--all'], ['--extract', '--all', '-j', '0']):
        with pytest.raises(SystemExit) as error: module._local_main(args)
        assert error.value.code == 2


def test_local_fetch_propagates_failure(load, monkeypatch, tmp_path):
    module = load('modal_film_transcribe'); monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    target = tmp_path / 'exports/synthetic'; target.mkdir(parents=True)
    (target / 'probe.json').write_text(json.dumps({'slug':'synthetic'}))
    def fail(command, **kwargs):
        assert kwargs['check'] is True
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError): module._local_main(['--fetch', '--film', 'synthetic'])


@pytest.mark.parametrize('boundaries', [[], [10, 30]])
@pytest.mark.parametrize('missing_clip', [False, True])
def test_shot_export_preserves_single_and_multiple_shot_contract(load, monkeypatch, boundaries, missing_clip):
    import torch
    module = load('modal_tn_generic')
    film = Path(module.DATA_ROOT) / 'synthetic'; film.mkdir(); (film / 'source.mp4').touch()
    monkeypatch.setattr(module, '_decode_frames', lambda *a: np.zeros((50, 27, 48, 3), dtype=np.uint8))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    class Detector:
        def to(self, device): assert device == 'cuda'; return self
        def eval(self): return self
        def __call__(self, windows):
            assert windows.shape == (1, 100, 27, 48, 3)
            logits = torch.full((1, 100, 1), -20.)
            for frame in boundaries: logits[0, 25+frame, 0] = 20.
            return logits, None
    monkeypatch.setitem(sys.modules, 'transnetv2_pytorch', NS(TransNetV2=Detector))
    commands = []
    def run(command, **kwargs):
        assert kwargs['check'] is True
        if command[0] == 'ffprobe': return NS(stdout='25/1,\n')
        commands.append(command)
        for index in range(len(boundaries)+1):
            if missing_clip and index == 0: continue
            target = command[-1] % index if boundaries else command[-1]
            Path(target).write_bytes(b'synthetic-encoded-output')
        return NS(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    if missing_clip:
        with pytest.raises(RuntimeError, match='missing expected shot'):
            module.detect('synthetic', 'source', th=.30, crf='26')
        module.vol.commit.assert_not_called()
        return
    result = module.detect('synthetic', 'source', th=.30, crf='26')
    command = commands[0]
    assert command[command.index('-crf')+1] == '26'
    assert command[command.index('-vf')+1] == 'scale=854:-2'
    if boundaries:
        assert command[command.index('-segment_times')+1] == '0.400,1.200'
        assert command[command.index('-force_key_frames')+1] == '0.400,1.200'
        assert command[-1].endswith('_s%04d.mp4')
    else:
        assert '-segment_times' not in command and '-force_key_frames' not in command
        assert command[-1].endswith('_s0000.mp4')
    assert result == {'shots': len(boundaries)+1}
    output = json.loads((film / 'source_shots.json').read_text())
    assert set(output) == {'fps', 'n', 'bounds'} and output['fps'] == 25.
    assert output['bounds'] == [0., *[x/25 for x in boundaries], 2.]
    manifest = (film / 'source_clips/manifest.csv').read_text().splitlines()
    assert manifest[0] == 'id,start_s,end_s,dur_s,file' and len(manifest) == len(boundaries)+2
    module.vol.commit.assert_called_once()


def test_local_extract_preserves_previous_output_when_probe_or_encoder_fails(load, monkeypatch, tmp_path):
    module = load('modal_film_transcribe'); monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    exports = tmp_path / 'exports/synthetic'; exports.mkdir(parents=True)
    sources = tmp_path / 'sources'; sources.mkdir(); (sources / 'input.mp4').touch()
    (exports / 'probe.json').write_text(json.dumps({'slug':'synthetic', 'dir':str(sources), 'source':'input.mp4'}))
    cache = tmp_path / 'exports/_film_audio'; cache.mkdir()
    previous = cache / 'synthetic.opus'; previous.write_bytes(b'previous-small-output')
    calls = []
    def run(command, **kwargs):
        assert kwargs['check'] is True; calls.append(command)
        if command[0] == 'ffprobe':
            return NS(stdout=json.dumps({'streams':[{'tags':{'language':'spa'}}, {'tags':{'language':'eng'}}]}))
        assert command[command.index('-map')+1] == '0:a:1'
        assert command[command.index('-ar')+1] == '16000'
        assert command[command.index('-b:a')+1] == '24k'
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError): module._local_main(['--extract', '--film', 'synthetic'])
    assert len(calls) == 2 and previous.read_bytes() == b'previous-small-output'
    assert not list(cache.glob('.extract-*'))
    def bad_probe(command, **kwargs): raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', bad_probe)
    with pytest.raises(subprocess.CalledProcessError): module._local_main(['--extract', '--film', 'synthetic'])
    assert previous.read_bytes() == b'previous-small-output'


@pytest.mark.parametrize('action', ['--extract', '--upload', '--fetch'])
def test_local_empty_input_does_not_report_success(load, monkeypatch, tmp_path, action):
    module = load('modal_film_transcribe'); monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='No '): module._local_main([action, '--all'])


@pytest.mark.parametrize('action', ['--extract', '--upload', '--fetch'])
@pytest.mark.parametrize('target', [[], ['--film', 'sample', '--all'], ['--film', '../sample'], ['--film', '']])
def test_local_actions_require_one_safe_explicit_target(load, action, target):
    with pytest.raises(SystemExit) as error:
        load('modal_film_transcribe')._local_main([action, *target])
    assert error.value.code == 2


def test_local_extract_uses_exact_probe_source_and_exact_film(load, monkeypatch, tmp_path):
    module = load('modal_film_transcribe')
    monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    sources = tmp_path / 'user inputs'; sources.mkdir()
    raw = tmp_path / 'external raw input.mov'; raw.write_bytes(b'private-source-placeholder')
    source = sources / 'Z raw source.mov'; source.symlink_to(raw)
    for name in ['_proxy.mp4', 'A_TRAILER.mp4', 'unrelated.mkv']:
        (sources / name).touch()
    target = tmp_path / 'exports/sample'; target.mkdir(parents=True)
    (target / 'probe.json').write_text(json.dumps({'slug': 'sample', 'dir': str(sources), 'source': source.name}))
    other = tmp_path / 'exports/sample-sequel'; other.mkdir()
    (other / 'probe.json').write_text('invalid unrelated metadata')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert kwargs['check'] is True
        if command[0] == 'ffprobe':
            assert command[-1] == str(source)
            return NS(stdout=json.dumps({'streams': [{'tags': {'language': 'eng'}}]}))
        assert command[command.index('-i') + 1] == str(source)
        assert command[command.index('-map') + 1] == '0:a:0'
        for flag, value in [('-ac', '1'), ('-ar', '16000'), ('-c:a', 'libopus'), ('-b:a', '24k')]:
            assert command[command.index(flag) + 1] == value
        Path(command[-1]).write_bytes(b'synthetic-extracted-audio')
        return NS(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    assert module._local_main(['--extract', '--film', 'sample']) == 0
    assert len(commands) == 2
    assert (tmp_path / 'exports/_film_audio/sample.opus').read_bytes() == b'synthetic-extracted-audio'
    assert not (tmp_path / 'exports/_film_audio/sample-sequel.opus').exists()


@pytest.mark.parametrize('source', [None, '', '..', '../raw.mp4', '/raw.mp4', 'nested\\raw.mp4', 'missing.mp4'])
def test_local_source_never_falls_back_to_another_video(load, tmp_path, source):
    module = load('modal_film_transcribe')
    (tmp_path / 'available-source.mp4').touch()
    with pytest.raises((ValueError, FileNotFoundError)):
        module._probe_source({'source': source, 'dir': str(tmp_path)})


@pytest.mark.parametrize('action', ['--extract', '--fetch'])
def test_local_batch_rejects_colliding_probe_slug_before_any_command(load, monkeypatch, tmp_path, action):
    module = load('modal_film_transcribe')
    monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    (tmp_path / 'raw.mp4').touch()
    for slug in ['sample', 'sample-sequel']:
        target = tmp_path / 'exports' / slug; target.mkdir(parents=True)
        (target / 'probe.json').write_text(json.dumps({'slug': 'sample', 'dir': str(tmp_path), 'source': 'raw.mp4'}))
    # The fixture rejects all media and service commands, including partial batch work.
    with pytest.raises(ValueError, match='export directory'):
        module._local_main([action, '--all'])


@pytest.mark.parametrize('action', ['--upload', '--fetch'])
def test_local_transfer_selects_exact_slug_or_explicit_all(load, monkeypatch, tmp_path, action):
    module = load('modal_film_transcribe')
    monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    cache = tmp_path / 'exports/_film_audio'; cache.mkdir(parents=True)
    for slug in ['sample', 'sample-sequel']:
        (cache / f'{slug}.opus').touch()
        target = tmp_path / 'exports' / slug; target.mkdir()
        (target / 'probe.json').write_text(json.dumps({'slug': slug}))
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: commands.append(command))
    assert module._local_main([action, '--film', 'sample']) == 0
    assert len(commands) == 1
    if action == '--upload':
        assert commands[0][-1] == '/sample/source_audio.opus'
    else:
        assert commands[0][-2] == '/sample/source_transcript.json'
    commands.clear()
    assert module._local_main([action, '--all']) == 0
    assert len(commands) == 2


@pytest.mark.parametrize('action', ['--extract', '--upload', '--fetch'])
def test_unknown_exact_local_film_fails_before_any_command(load, monkeypatch, tmp_path, action):
    module = load('modal_film_transcribe')
    monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        module._local_main([action, '--film', 'missing'])


def test_local_generated_output_symlink_cannot_escape_root(load, monkeypatch, tmp_path):
    module = load('modal_film_transcribe')
    root = tmp_path / 'private'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    exports = root / 'exports'; exports.mkdir()
    (exports / 'sample').symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv('VIDEO_EDITING_DATA_ROOT', str(root))
    with pytest.raises(ValueError, match='escapes'):
        module._local_main(['--fetch', '--film', 'sample'])
    (exports / '_film_audio').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        module._local_main(['--upload', '--all'])


def test_remote_only_selects_exact_slug_and_rejects_no_match(load):
    module = load('modal_film_transcribe')
    module.list_ready = NS(remote=lambda: ['sample', 'sample-sequel'])
    jobs_seen = []
    def starmap(jobs):
        jobs_seen.extend(jobs)
        return [{'slug': slug, 'ok': True} for slug, _ in jobs]
    module.transcribe = NS(starmap=starmap)
    module.main(only='sample')
    assert jobs_seen == [('sample', 'large-v3')]
    with pytest.raises(RuntimeError, match='No source'):
        module.main(only='sam')


@pytest.mark.parametrize('fps', ['23.976', '30000/1001', '1/1000', '120'])
def test_valid_frame_rates_are_not_overrestricted(load, fps):
    from fractions import Fraction
    assert load('modal_tn_generic')._parse_fps(fps) == float(Fraction(fps))
