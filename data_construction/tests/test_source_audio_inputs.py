"""Synthetic source selection regressions; no real media or model execution."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import cut_takes
import release_config


class SourceAudioInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='source audio inputs ')
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))

        def forbidden(*args, **kwargs):
            raise AssertionError('Unexpected network or media process')
        for target, name in [(socket, 'create_connection'), (socket.socket, 'connect'),
                             (subprocess, 'run'), (subprocess, 'Popen')]:
            self.stack.enter_context(mock.patch.object(target, name, forbidden))

        self.base = Path(self.temp.name)
        self.root = self.base / 'private'; self.root.mkdir()
        source_dir = self.base / 'source inputs'; source_dir.mkdir()
        raw = self.base / 'external raw footage.mov'; raw.touch()
        self.source = source_dir / 'Z raw source.mov'; self.source.symlink_to(raw)
        for name in ['_proxy.mp4', 'A_TRAILER.mp4', 'unrelated.mkv']:
            (source_dir / name).touch()
        self.exports = self.root / 'exports/sample'; self.exports.mkdir(parents=True)
        self.probe = {'slug': 'sample', 'dir': str(source_dir), 'source': self.source.name}
        self.write_probe()

    def write_probe(self):
        (self.exports / 'probe.json').write_text(json.dumps(self.probe), encoding='utf-8')

    def prepare_attempt(self, cached_audio=False):
        self.stack.enter_context(mock.patch.object(release_config, 'DATA_ROOT', str(self.root)))
        self.stack.enter_context(mock.patch.object(
            sys, 'argv', ['cut_takes.py', '--film', 'sample', '--handle', '3']))
        (self.exports / 'line_origins.json').write_text(json.dumps({
            'n_lines': 1, 'n_found': 1,
            'lines': [{'found': True, 'film_s': 10., 'film_e': 12., 'text': 'Synthetic line',
                       's': 1., 'e': 3., 'film_text': 'Synthetic line', 'score': .9}],
        }), encoding='utf-8')
        (self.exports / 'takes.json').write_text('{"old_success": true}', encoding='utf-8')
        take = self.exports / 'takes/t01.m4a'
        take.parent.mkdir(exist_ok=True)
        if cached_audio:
            take.write_bytes(b'previously-extracted-audio')
        else:
            take.unlink(missing_ok=True)
        return take

    def test_cut_takes_uses_exact_raw_source_and_accepts_user_input_symlink(self):
        self.assertEqual(cut_takes.film_src('sample', self.root), str(self.source))

    def test_cut_takes_never_falls_back_to_available_video(self):
        for source_name in [None, '', '..', '../raw.mp4', '/raw.mp4',
                            'nested\\raw.mp4', 'missing.mp4']:
            with self.subTest(source_name=source_name):
                self.probe['source'] = source_name
                self.write_probe()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    cut_takes.film_src('sample', self.root)

    def test_cut_takes_rejects_unsafe_slug_before_reading(self):
        for slug in ['', '..', '../outside', '/absolute', 'nested/file', 'nested\\file']:
            with self.subTest(slug=slug), self.assertRaisesRegex(ValueError, 'slug'):
                cut_takes.film_src(slug, self.root)

    def test_cut_takes_rejects_probe_slug_collision(self):
        self.probe['slug'] = 'other'
        self.write_probe()
        with self.assertRaisesRegex(ValueError, 'export directory'):
            cut_takes.film_src('sample', self.root)

    def test_cut_takes_rejects_output_symlink_escape(self):
        (self.root / 'exports/alias').symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            cut_takes.film_src('alias', self.root)

    def test_cut_command_and_speech_settings_preserved(self):
        self.prepare_attempt()
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            self.assertEqual(command[command.index('-i') + 1], str(self.source))
            for flag, value in [('-ss', '7.000'), ('-t', '8.000'), ('-ac', '2'),
                                ('-ar', '48000'), ('-c:a', 'aac'), ('-b:a', '128k')]:
                self.assertEqual(command[command.index(flag) + 1], value)
            Path(command[-1]).write_bytes(b'synthetic-extracted-take')
            return NS(returncode=0)

        test = self
        class Whisper:
            def __init__(self, model, **kwargs):
                test.assertEqual(model, 'small.en')
                test.assertEqual(kwargs, {'device': 'cpu', 'compute_type': 'int8'})

            def transcribe(self, path):
                test.assertEqual(Path(path).read_bytes(), b'synthetic-extracted-take')
                return iter([NS(start=3., end=5., text='Synthetic line')]), NS()

        self.stack.enter_context(mock.patch.object(subprocess, 'run', run))
        self.stack.enter_context(mock.patch.dict(sys.modules, {'faster_whisper': NS(WhisperModel=Whisper)}))
        cut_takes.main()
        self.assertEqual(len(commands), 1)
        output = json.loads((self.exports / 'takes.json').read_text())
        self.assertEqual(output['film'], 'sample')
        self.assertEqual(output['n_takes'], 1)
        take = output['takes'][0]
        self.assertEqual(take['words_start_at_s'], 3.)
        self.assertEqual(take['take_dur_s'], 8.)
        self.assertEqual(take['speech_spans_s'], [{'s': 3., 'e': 5., 'text': 'Synthetic line'}])

    def test_asr_startup_direct_and_generator_failures_raise_without_success_manifest(self):
        for phase in ['startup', 'direct', 'iteration']:
            with self.subTest(phase=phase):
                take = self.prepare_attempt(cached_audio=True)
                failure = TypeError('synthetic incompatible audio dependency')

                def segments():
                    yield NS(start=3., end=5., text='Partial speech before failure')
                    raise failure

                model = mock.Mock()
                factory = mock.Mock(return_value=model)
                if phase == 'startup':
                    factory.side_effect = failure
                elif phase == 'direct':
                    model.transcribe.side_effect = failure
                else:
                    model.transcribe.return_value = (segments(), NS())
                with mock.patch.dict(sys.modules, {'faster_whisper': NS(WhisperModel=factory)}):
                    with self.assertRaisesRegex(RuntimeError, 'Speech transcription failed.*requirements.txt') as error:
                        cut_takes.main()
                self.assertIs(error.exception.__cause__, failure)
                self.assertFalse((self.exports / 'takes.json').exists())
                self.assertEqual(take.read_bytes(), b'previously-extracted-audio')

    def test_real_empty_or_blank_transcript_is_a_successful_empty_result(self):
        for segments in [[], [NS(start=3., end=5., text='   ')]]:
            with self.subTest(segments=segments):
                self.prepare_attempt(cached_audio=True)
                model = mock.Mock()
                model.transcribe.return_value = (iter(segments), NS())
                with mock.patch.dict(sys.modules, {'faster_whisper': NS(WhisperModel=mock.Mock(return_value=model))}):
                    cut_takes.main()
                output = json.loads((self.exports / 'takes.json').read_text())
                self.assertEqual(output['n_takes'], 0)
                self.assertEqual(output['takes'], [])
                self.assertNotIn('old_success', output)

    def test_converter_failure_missing_or_empty_output_cannot_report_success(self):
        for outcome in ['failure', 'missing', 'empty']:
            with self.subTest(outcome=outcome):
                take = self.prepare_attempt()
                factory = mock.Mock()

                def run(command, **kwargs):
                    if outcome == 'failure':
                        Path(command[-1]).write_bytes(b'partial-corrupt-output')
                    elif outcome == 'empty':
                        Path(command[-1]).touch()
                    return NS(returncode=1 if outcome == 'failure' else 0, stderr='synthetic encoder error')

                with mock.patch.object(subprocess, 'run', run), \
                     mock.patch.dict(sys.modules, {'faster_whisper': NS(WhisperModel=factory)}):
                    with self.assertRaisesRegex(RuntimeError, 'Audio conversion'):
                        cut_takes.main()
                factory.assert_not_called()
                self.assertFalse(take.exists())
                self.assertFalse((self.exports / 'takes.json').exists())
                self.assertEqual(list(take.parent.iterdir()), [])

    def test_missing_source_invalidates_previous_success_at_attempt_start(self):
        self.prepare_attempt(cached_audio=True)
        self.probe['source'] = 'missing.mp4'
        self.write_probe()
        with self.assertRaises(FileNotFoundError):
            cut_takes.main()
        self.assertFalse((self.exports / 'takes.json').exists())


if __name__ == '__main__':
    unittest.main()
