"""Bounded, offline editing observations; never select clips or return rewards.

Independent OpenCode overlay. Legacy observation workers and graders stay intact.
All decoded pictures are delivered as MCP images, not just filesystem paths.
"""
from __future__ import annotations

import base64
import bisect
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .visual_budget import UNIT_PIXELS, MAX_EDGE
from .media_duration_v1 import normalize_media_metadata

MAX_CELLS = 16
MAX_SCAN_SECONDS = 180

ITEM_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['path'], 'properties': {
    'path': {'type': 'string'}, 'frame_index': {'type': 'integer', 'minimum': 0},
    'timestamp_s': {'type': 'number', 'minimum': 0},
    'crop': {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'minItems': 4, 'maxItems': 4}}}


def schema(name, description, properties, required=()):
    return {'name': name, 'description': description, 'parameters': {'type': 'object',
        'additionalProperties': False, 'properties': properties, 'required': list(required)}}


SCHEMAS = [
    schema('contact_sheet', 'View up to 16 images/video frames together as ONE labelled contact sheet returned directly as an image. Each cell lists its number, filename, actual frame index and decoded time; full paths and PTS are in JSON. Choose each source with path plus frame_index OR timestamp_s (video default 0); still images need only path. Optional normalized crop [x,y,w,h]. Official sheets charge source-frame count AND pixel size, not just one image. Revisit small cells with inspect_frames in individual/crop mode. No clip selection or semantic judging.',
        {'items': {'type': 'array', 'items': ITEM_SCHEMA, 'minItems': 1, 'maxItems': 16},
         'columns': {'type': 'integer', 'minimum': 1, 'maximum': 4},
         'cell_width': {'type': 'integer', 'minimum': 160, 'maximum': 448}}, ('items',)),
    schema('inspect_frames', 'Precisely inspect a cut/action or crop. Select frame_indices (zero based), timestamps_s (nearest decoded frame), OR center_s with before/after frame counts (defaults 4 before + center + 3 after). Returns actual decoded PTS and file-relative time, never fabricated FPS times. Default layout=sheet (up to 16 frames, edge=320 per cell); individual gives up to 4 pictures at edge=896, max1792. crop=[x,y,w,h] is normalized on source image before resizing; useful for faces/text/details. Still images with path/crop and no time are supported. This is sampled imagery, not native video playback; indexing limited to 600-second files and bounded CPU time.',
        {'path': {'type': 'string'}, 'frame_indices': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 1, 'maxItems': 16},
         'timestamps_s': {'type': 'array', 'items': {'type': 'number', 'minimum': 0}, 'minItems': 1, 'maxItems': 16},
         'center_s': {'type': 'number', 'minimum': 0}, 'before': {'type': 'integer', 'minimum': 0, 'maximum': 15},
         'after': {'type': 'integer', 'minimum': 0, 'maximum': 15}, 'crop': ITEM_SCHEMA['properties']['crop'],
         'layout': {'type': 'string', 'enum': ['sheet', 'individual']}, 'edge': {'type': 'integer', 'minimum': 160, 'maximum': 1792}}, ('path',)),
    schema('analyze_audio', 'Offline audio measurements: window LUFS/true peak/LRA, original per-channel RMS/peaks/clipping, channel silence intervals, transient/onset candidates and uncertain periodic pulse estimate. No downmix for level measurement. Choose audio_stream as zero-based AUDIO stream ordinal (default0); up to8 channels, duration<=30s. No cumulative episode audio-call/seconds quota; each window remains at most30s and the whole-tool deadline still applies. No ASR or semantic LLM; use inspect_audio for English transcription. Onsets are not guaranteed musical beats; measurements are not rewards.',
        {'path': {'type': 'string'}, 'start_s': {'type': 'number', 'minimum': 0},
         'duration_s': {'type': 'number', 'minimum': .05, 'maximum': 30},
         'audio_stream': {'type': 'integer', 'minimum': 0, 'maximum': 7}}, ('path',)),
    schema('validate_media', 'Offline technical self-check of preview media and/or your JSON plan. Reports decode failures, metadata and black/freeze/silence candidate intervals, which may be intentional and are NOT scores. path selects media; scan defaults to first180s and reports partial coverage (max180s/call,360s/episode for S3/S4E). expected contains ONLY your supplied public technical requirements; absent fields are not guessed from hidden answers. plan_path checks selected/order/cuts or edit JSON references, duplicates and source ranges against assets_dir (default initial_state); it cannot certify aesthetic quality or full JSON/render equivalence. Never modifies files or calls a grader.',
        {'path': {'type': 'string'}, 'start_s': {'type': 'number', 'minimum': 0},
         'duration_s': {'type': 'number', 'minimum': .05, 'maximum': 180},
         'plan_path': {'type': 'string'}, 'assets_dir': {'type': 'string'},
         'expected': {'type': 'object', 'additionalProperties': False, 'properties': {
             'duration_s': {'type': 'number', 'minimum': 0}, 'duration_tolerance_s': {'type': 'number', 'minimum': 0, 'maximum': 2},
             'width': {'type': 'integer', 'minimum': 1}, 'height': {'type': 'integer', 'minimum': 1},
             'fps': {'type': 'number', 'minimum': 0}, 'has_audio': {'type': 'boolean'}}}}),
]


def number(value, name, lo=0, hi=None, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < lo or (hi is not None and value > hi):
        raise ValueError(f'{name} outside [{lo}, {hi}]')
    if integer and type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    return value


def keys(args, allowed, required=()):
    if type(args) is not dict or set(args) - set(allowed) or not set(required) <= set(args):
        raise ValueError('Missing/unknown tool arguments')


def rate(value):
    try:
        result = float(Fraction(value))
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class EditingTools:
    def __init__(self, work=Path('/work')):
        self.work = Path(work).resolve()
        self.indices = {}

    def path(self, value, *, directory=False):
        if type(value) is not str or not value or len(value) > 1024:
            raise ValueError('Invalid workspace path')
        path = Path(value)
        path = path if path.is_absolute() else self.work / path
        if '..' in path.parts or not path.is_relative_to(self.work):
            raise ValueError('Path must stay inside the task workspace')
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink paths are not accepted')
        if directory:
            if not path.is_dir():
                raise ValueError('Not a directory')
        elif not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024 * 1024:
            raise ValueError('Missing/empty file or file exceeds 512 MiB')
        return path

    @staticmethod
    def run(argv, *, timeout=45, max_bytes=64 * 1024 * 1024, allow_error=False):
        # Redirect to bounded-lifetime files so verbose corrupt streams cannot
        # accumulate unbounded in-process buffers. Inspect size before reading.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            try:
                code = proc.wait(timeout=timeout)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            if out.tell() > max_bytes or err.tell() > 8 * 1024 * 1024:
                raise ValueError('Media diagnostic output exceeded bounds')
            out.seek(0)
            err.seek(0)
            body, stderr = out.read(), err.read().decode(errors='replace')
        if code and not allow_error:
            raise ValueError(f'Media command failed: {stderr[-1200:]}')
        return body, stderr, code

    def probe(self, path):
        out, _, _ = self.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error',
            '-protocol_whitelist', 'file,pipe', '-show_entries',
            'format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,sample_rate,channels,channel_layout,duration,nb_frames,start_time',
            '-of', 'json', str(path)], max_bytes=1024 * 1024)
        data = json.loads(out)
        duration = finite(data.get('format', {}).get('duration'))
        if duration is None or duration <= 0:
            raise ValueError('Cannot establish actual media duration')
        return normalize_media_metadata({'duration_s': duration, 'streams': data.get('streams', [])})

    def frame_index(self, path, *, max_duration_s=600):
        stat = path.stat()
        identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity in self.indices:
            cached = self.indices[identity]
            if max_duration_s is not None and cached['duration_s'] > max_duration_s:
                raise ValueError('Frame indexing requires a video no longer than 600 seconds; split longer files first')
            return cached
        meta = self.probe(path)
        video = next((s for s in meta['streams'] if s['codec_type'] == 'video'), None)
        if video is None or (max_duration_s is not None and meta['duration_s'] > max_duration_s):
            raise ValueError('Frame indexing requires a video no longer than 600 seconds; split longer files first')
        out, _, _ = self.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error',
            '-threads', '1', '-protocol_whitelist', 'file,pipe', '-select_streams', 'v:0',
            '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time,pts_time,pkt_duration_time,duration_time',
            '-of', 'json', str(path)], max_bytes=16 * 1024 * 1024)
        raw = json.loads(out).get('frames', [])
        pts = [finite(f.get('best_effort_timestamp_time', f.get('pts_time'))) for f in raw]
        if not pts or len(pts) > 100_000 or any(p is None for p in pts) or any(a >= b for a, b in zip(pts, pts[1:])):
            raise ValueError('Missing/non-monotone display timestamps; exact frame selection unavailable')
        fps = rate(video.get('avg_frame_rate'))
        last = finite(raw[-1].get('duration_time', raw[-1].get('pkt_duration_time')))
        tail = last if last is not None and last > 0 else (pts[-1] - pts[-2] if len(pts) > 1 else (1 / fps if fps else None))
        if tail is None:
            raise ValueError('Cannot establish last frame duration')
        result = {'pts': pts, 'times': [p - pts[0] for p in pts], 'origin_pts_s': pts[0],
                  'duration_s': pts[-1] - pts[0] + tail, 'video': video}
        if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size:
            raise ValueError('Video changed while indexing')
        if len(self.indices) >= 32:
            self.indices.pop(next(iter(self.indices)))
        self.indices[identity] = result
        return result

    @staticmethod
    def crop_box(crop, width, height):
        if crop is None:
            return None
        if type(crop) is not list or len(crop) != 4:
            raise ValueError('crop is normalized [x, y, width, height]')
        x, y, w, h = [number(v, 'crop', 0, 1) for v in crop]
        if w <= 0 or h <= 0 or x + w > 1.0000001 or y + h > 1.0000001:
            raise ValueError('Crop rectangle outside image')
        box = (round(x * width), round(y * height), round((x + w) * width), round((y + h) * height))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Crop is smaller than a pixel')
        return box

    def frames(self, items, edge, *, max_duration_s=600):
        from PIL import Image, ImageOps
        output, groups = [None] * len(items), {}
        for position, item in enumerate(items):
            keys(item, ('path', 'frame_index', 'timestamp_s', 'crop'), ('path',))
            if 'frame_index' in item and 'timestamp_s' in item:
                raise ValueError('Select frame_index OR timestamp_s')
            path = self.path(item['path'])
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                if 'frame_index' in item or 'timestamp_s' in item:
                    raise ValueError('A still image has no video timestamp')
                with Image.open(path) as source:
                    if source.width * source.height > 24_000_000 or getattr(source, 'n_frames', 1) != 1:
                        raise ValueError('Oversized/animated image')
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    box = self.crop_box(item.get('crop'), *image.size)
                    if box:
                        image = image.crop(box)
                    image.thumbnail((edge, edge))
                output[position] = (image, {'path': str(path.relative_to(self.work)), 'crop': item.get('crop'),
                    'frame_index': None, 'actual_time_s': None, 'actual_pts_s': None})
                continue
            info = self.frame_index(path, max_duration_s=max_duration_s)
            if 'frame_index' in item:
                index = number(item['frame_index'], 'frame_index', 0, len(info['pts']) - 1, integer=True)
            else:
                time = number(item.get('timestamp_s', 0), 'timestamp_s', 0)
                if time >= info['duration_s']:
                    raise ValueError('Timestamp outside decoded video duration')
                right = bisect.bisect_left(info['times'], time)
                choices = [i for i in (right - 1, right) if 0 <= i < len(info['times'])]
                index = min(choices, key=lambda i: (abs(info['times'][i] - time), i))
            key = (path, tuple(item['crop']) if 'crop' in item else None)
            groups.setdefault(key, []).append((position, index, info, item))
        for (path, crop), requests in groups.items():
            indexes = sorted({r[1] for r in requests})
            info = requests[0][2]
            box = self.crop_box(list(crop) if crop is not None else None, info['video']['width'], info['video']['height'])
            select = '+'.join(f'eq(n\\,{index})' for index in indexes)
            filters = [f'select={select}', 'showinfo']
            if box:
                filters.append(f'crop={box[2]-box[0]}:{box[3]-box[1]}:{box[0]}:{box[1]}:exact=1')
            # A larger requested edge preserves available detail; it must not
            # invent pixels (and visual-token cost) by enlarging a tiny crop.
            filters.append(f'scale=min(iw\\,{edge}):min(ih\\,{edge}):force_original_aspect_ratio=decrease')
            with tempfile.TemporaryDirectory(prefix='ocrl-frames-') as tmp:
                _, log, _ = self.run([shutil.which('ffmpeg') or 'ffmpeg', '-nostdin', '-hide_banner',
                    '-loglevel', 'info', '-threads', '1', '-filter_threads', '1', '-copyts', '-noautorotate',
                    '-protocol_whitelist', 'file,pipe', '-i', str(path), '-map', '0:v:0', '-an',
                    '-vf', ','.join(filters), '-fps_mode', 'passthrough', '-frames:v', str(len(indexes)),
                    '-threads', '1', str(Path(tmp) / '%03d.png')])
                actual = [float(s) for s in re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-\d.eE+]+)', log)]
                files = sorted(Path(tmp).glob('*.png'))
                if len(actual) != len(indexes) or len(files) != len(indexes):
                    raise ValueError('Decoded frame count did not match requested frames')
                if any(abs(t - info['pts'][i]) > 0.001 for i, t in zip(indexes, actual)):
                    raise ValueError('Actual decoded PTS differs from frame index; refusing misleading labels')
                images = {i: Image.open(p).convert('RGB') for i, p in zip(indexes, files)}
                for position, index, _, item in requests:
                    output[position] = (images[index].copy(), {'path': str(path.relative_to(self.work)),
                        'requested_time_s': item.get('timestamp_s'), 'frame_index': index,
                        'actual_time_s': info['times'][index], 'actual_pts_s': info['pts'][index],
                        'crop': item.get('crop')})
        return output

    @staticmethod
    def visual_source_minimum(name, args):
        """No media reads: never overestimate a boundary-clipped neighborhood."""
        if name == 'contact_sheet':
            items = args.get('items')
            if type(items) is not list or not 1 <= len(items) <= MAX_CELLS:
                raise ValueError('Choose 1–16 source frames/images')
            return len(items)
        for key in ('frame_indices', 'timestamps_s'):
            if key in args:
                values = args[key]
                if type(values) is not list or not 1 <= len(values) <= MAX_CELLS:
                    raise ValueError('Choose 1–16 source frames/images')
                return len(values)
        return 1

    def inspect_video(self, args):
        """Legacy call shape, exact video-relative times, one grouped decode.

        Unlike the advanced frame tool, whole-clip overview has no new 600s
        restriction. File size, frame count, subprocess and MCP deadlines still
        bound work. Repeated requests reuse an mtime/size-keyed frame index.
        """
        keys(args, ('path', 'timestamps_s', 'start_s', 'end_s', 'num_frames'), ('path',))
        explicit = args.get('timestamps_s') is not None
        if explicit:
            if any(args.get(k) is not None for k in ('start_s', 'end_s', 'num_frames')):
                raise ValueError('use timestamps_s OR start_s/end_s/num_frames, not both')
            times = args['timestamps_s']
            if type(times) is not list or not 1 <= len(times) <= 8:
                raise ValueError('timestamps_s must contain 1–8 timestamps')
            times = [number(t, 'timestamp_s') for t in times]
        else:
            count = number(args.get('num_frames', 8), 'num_frames', 1, 8, integer=True)
            start = number(args.get('start_s', 0), 'start_s')
        path = self.path(args['path'])
        info = self.frame_index(path, max_duration_s=None)
        duration = info['duration_s']
        if not explicit:
            end = number(args.get('end_s', duration), 'end_s')
            if not 0 <= start < end <= duration:
                raise ValueError(f'require 0 <= start_s < end_s <= decoded video duration {duration:.6f}s')
            times = [start + (end-start) * (i+.5)/count for i in range(count)]
        if any(t >= duration for t in times):
            raise ValueError(f'timestamps must be < decoded video duration {duration:.6f}s')
        frames = self.frames([{'path': args['path'], 'timestamp_s': t} for t in times],
                             448, max_duration_s=None)
        images, records = [], []
        for picture, record in frames:
            stream = io.BytesIO()
            picture.save(stream, format='JPEG', quality=82)
            body = stream.getvalue()
            if len(body) > 256 * 1024:
                raise ValueError('Sanitized frame exceeds byte budget')
            images.append({'jpeg_b64': base64.b64encode(body).decode(),
                'label': (f"{args['path']} requested t={record['requested_time_s']:.6f}s; "
                          f"actual t={record['actual_time_s']:.6f}s, PTS={record['actual_pts_s']:.6f}s, "
                          f"frame={record['frame_index']}"),
                'width': picture.width, 'height': picture.height})
            records.append(record)
        media = self.probe(path)
        media.setdefault('container_duration_s', media['duration_s'])
        media['duration_s'] = duration
        media['duration_basis'] = 'decoded_video_frame_span'
        return {'status': 'ok', 'media': media, 'timestamps_s': times,
                'actual_timestamps_s': [r['actual_time_s'] for r in records],
                'frames': records, 'images': images,
                'sampling': 'explicit_timestamps' if explicit else 'uniform_window',
                'note': 'Nearest actual decoded frames, not continuous playback. Times are relative to the first video frame; original PTS is preserved. Audio/container overhang is not extra video.'}

    def gallery(self, items, *, columns=4, cell_width=320, sheet=True):
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        if type(items) is not list or not 1 <= len(items) <= MAX_CELLS:
            raise ValueError('Choose 1–16 source frames/images')
        columns = number(columns, 'columns', 1, 4, integer=True)
        cell_width = number(cell_width, 'cell_width', 160, 448 if sheet else MAX_EDGE, integer=True)
        if not sheet and len(items) > 4:
            raise ValueError('Individual high-resolution mode allows at most 4 images')
        frames = self.frames(items, cell_width)
        records = [dict(meta, cell=i + 1) for i, (_, meta) in enumerate(frames)]
        if sheet:
            cols = min(columns, len(frames))
            rows = math.ceil(len(frames) / cols)
            if rows > 4:
                raise ValueError('Keep contact sheets to at most 4 rows; increase columns or split the request')
            cell_h, label_h = round(cell_width * 9 / 16), 42
            canvas = Image.new('RGB', (cols * cell_width, rows * (cell_h + label_h)), '#171b23')
            draw = ImageDraw.Draw(canvas)
            font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
            font = ImageFont.truetype(font_path, 12) if Path(font_path).exists() else ImageFont.load_default(size=12)
            for i, (image, meta) in enumerate(frames):
                x, y = (i % cols) * cell_width, (i // cols) * (cell_h + label_h)
                thumb = ImageOps.contain(image, (cell_width, cell_h))
                canvas.paste(thumb, (x + (cell_width - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
                name = Path(meta['path']).name.encode('ascii', 'replace').decode()[:40]
                label = f"{i+1:02d}  {name}"
                if meta['frame_index'] is not None:
                    label += f"\nf={meta['frame_index']}  t={meta['actual_time_s']:.3f}s"
                draw.text((x + 4, y + cell_h + 3), label, font=font, fill='white')
            canvas.thumbnail((MAX_EDGE, MAX_EDGE))
            pictures = [(canvas, 'Contact sheet; cell numbers map to the exact source/PTS records below')]
        else:
            pictures = [(im, json.dumps(meta, ensure_ascii=False)) for im, meta in frames]
        image_results = []
        for picture, label in pictures:
            stream = io.BytesIO()
            picture.save(stream, format='JPEG', quality=90)
            image_results.append({'label': label, 'jpeg_b64': base64.b64encode(stream.getvalue()).decode(),
                                  'width': picture.width, 'height': picture.height})
        units = max(len(items), sum(math.ceil(im.width * im.height / UNIT_PIXELS) for im, _ in pictures))
        return {'status': 'ok', 'cells': records, 'images': image_results, '_visual_source_units': units,
                'note': 'Pictures are actual observations, not continuous playback. Time is relative to first decoded video frame; PTS is also preserved. No AI upscaling or invented detail. Video crops use stored pixel orientation (auto-rotation disabled). Tiny sheet cells can be revisited in crop/individual mode.'}

    def contact_sheet(self, args):
        keys(args, ('items', 'columns', 'cell_width'), ('items',))
        return self.gallery(args['items'], columns=args.get('columns', 4), cell_width=args.get('cell_width', 320))

    def inspect_frames(self, args):
        keys(args, ('path', 'frame_indices', 'timestamps_s', 'center_s', 'before', 'after', 'crop', 'layout', 'edge'), ('path',))
        modes = [k for k in ('frame_indices', 'timestamps_s', 'center_s') if k in args]
        if len(modes) > 1:
            raise ValueError('Choose frame_indices, timestamps_s, OR center_s')
        if ('before' in args or 'after' in args) and 'center_s' not in args:
            raise ValueError('before/after only apply to center_s')
        if args.get('layout', 'sheet') not in ('sheet', 'individual'):
            raise ValueError('layout must be sheet or individual')
        base = {'path': args['path'], **({'crop': args['crop']} if 'crop' in args else {})}
        if 'center_s' in args:
            info = self.frame_index(self.path(args['path']))
            center = number(args['center_s'], 'center_s', 0)
            if center >= info['duration_s']:
                raise ValueError('Cut timestamp outside video')
            pivot = min(bisect.bisect_left(info['times'], center), len(info['times']) - 1)
            before = number(args.get('before', 4), 'before', 0, 15, integer=True)
            after = number(args.get('after', 3), 'after', 0, 15, integer=True)
            chosen = range(max(0, pivot - before), min(len(info['times']), pivot + after + 1))
            items = [{**base, 'frame_index': i} for i in chosen]
        elif modes:
            mode = modes[0]
            values = args[mode]
            if type(values) is not list or not 1 <= len(values) <= MAX_CELLS:
                raise ValueError('Choose 1–16 frames')
            key = 'frame_index' if mode == 'frame_indices' else 'timestamp_s'
            items = [{**base, key: value} for value in values]
        else:
            items = [base]
        sheet = args.get('layout', 'sheet') == 'sheet'
        return self.gallery(items, sheet=sheet, cell_width=args.get('edge', 320 if sheet else 896))

    def analyze_audio(self, args):
        import numpy as np
        keys(args, ('path', 'start_s', 'duration_s', 'audio_stream'), ('path',))
        path = self.path(args['path'])
        start = number(args.get('start_s', 0), 'start_s')
        requested = number(args.get('duration_s', 20), 'duration_s', .05, 30)
        index = number(args.get('audio_stream', 0), 'audio_stream', 0, 7, integer=True)
        meta = self.probe(path)
        streams = [s for s in meta['streams'] if s['codec_type'] == 'audio']
        if not streams:
            return {'status': 'no_audio_stream', 'media': meta}
        if index >= len(streams) or start >= meta['duration_s']:
            raise ValueError('Audio stream/window does not exist')
        stream = streams[index]
        channels = stream.get('channels', 0)
        if not 1 <= channels <= 8:
            raise ValueError('Supports at most 8 original audio channels')
        duration = requested
        raw, log, _ = self.run([shutil.which('ffmpeg') or 'ffmpeg', '-nostdin', '-hide_banner', '-v', 'info',
            '-threads', '1', '-filter_threads', '1', '-protocol_whitelist', 'file,pipe', '-ss', str(start),
            '-i', str(path), '-t', str(duration), '-map', f'0:a:{index}', '-vn', '-sn', '-dn',
            '-af', 'ashowinfo,ebur128=peak=true', '-ar', '48000', '-f', 'f32le', 'pipe:1'])
        samples = np.frombuffer(raw, dtype='<f4')
        if len(samples) % channels:
            raise ValueError('Incomplete channel sample frame')
        samples = samples.reshape(-1, channels)
        if not len(samples):
            return {'status': 'no_audio_samples', 'media': meta}
        if not np.isfinite(samples).all():
            raise ValueError('Non-finite decoded audio samples')
        pts = re.findall(r'\[Parsed_ashowinfo[^\]]*\].*?\bpts_time:([-\d.eE+]+)', log)
        if not pts:
            raise ValueError('Audio decoded without sample timestamp evidence')
        requested_start = start
        start += float(pts[0])  # Input -ss rebases PTS; retain late-starting tracks.
        def db(value):
            return 20 * math.log10(value) if value > 0 else None
        metrics = []
        silence = []
        for channel in range(channels):
            signal = samples[:, channel].astype(np.float64)
            peak = float(np.max(np.abs(signal)))
            metrics.append({'channel': channel, 'sample_peak_dbfs': db(peak),
                'rms_dbfs': db(float(np.sqrt(np.mean(signal ** 2)))),
                'near_clip_fraction': float(np.mean(np.abs(signal) >= .999))})
            flags = np.max(np.abs(signal[:len(signal)//480*480].reshape(-1, 480)), axis=1) < 10 ** (-50/20)
            edges = np.diff(np.r_[False, flags, False].astype(int))
            intervals = [(a/100 + start, b/100 + start) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= 10]
            silence.append({'channel': channel, 'intervals_s': intervals[:128], 'truncated': len(intervals) > 128})
        actual = len(samples) / 48000
        mono = np.mean(samples, axis=1)
        bins = len(mono) // 480
        envelope = np.sqrt(np.mean(mono[:bins*480].reshape(-1, 480).astype(np.float64)**2, axis=1))
        onset = np.maximum(np.diff(envelope, prepend=envelope[0]), 0)
        threshold = max(float(np.median(onset) + 4 * np.median(np.abs(onset - np.median(onset)))), .002)
        peaks = [i for i in range(1, len(onset)-1) if onset[i] > threshold and onset[i] >= onset[i-1] and onset[i] > onset[i+1]]
        chosen = []
        for i in peaks:
            if not chosen or i - chosen[-1] >= 10:
                chosen.append(i)
            elif onset[i] > onset[chosen[-1]]:
                chosen[-1] = i
        tempo, regularity = None, None
        if len(chosen) >= 4 and len(onset) >= 300:
            corr = np.correlate(onset, onset, mode='full')[len(onset)-1:]
            lo, hi = 30, min(150, len(corr)-1)
            lag = int(np.argmax(corr[lo:hi+1]) + lo)
            regularity = float(corr[lag] / max(corr[0], 1e-12))
            if regularity >= .25:
                tempo = 6000 / lag
        def last(pattern):
            values = re.findall(pattern, log)
            return finite(values[-1]) if values else None
        is_silent = float(np.max(np.abs(samples))) < 1e-6
        return {'status': 'digital_silence' if is_silent else 'ok', 'media': meta, 'audio_stream': index,
            'channel_layout': stream.get('channel_layout', 'unknown'), 'requested_start_s': requested_start,
            'start_s': start, 'decoded_duration_s': actual,
            'integrated_lufs': None if is_silent or actual < .4 else last(r'I:\s*([-\w.+]+)\s+LUFS'),
            'true_peak_dbtp': None if is_silent else last(r'Peak:\s*([-\w.+]+)\s+dBFS'),
            'loudness_range_lu': None if actual < 3 else last(r'LRA:\s*([-\w.+]+)\s+LU'),
            'channels': metrics, 'silence_intervals': silence,
            'onset_candidates_s': [start + i/100 for i in chosen[:128]], 'onsets_truncated': len(chosen) > 128,
            'periodic_pulse_bpm_estimate': tempo, 'pulse_periodicity': regularity,
            'note': 'Offline signal evidence, not a score. Original channels preserved. Silence threshold -50 dBFS/100 ms; intervals use 10 ms bins. Onsets are not guaranteed beats; pulse estimates have octave ambiguity. LUFS/LRA describe this window only. No speech/music semantic service is called.'}

    def validate_plan(self, value, assets_dir):
        """Only public file references/ranges; never import a grader/verifier."""
        plan = json.loads(self.path(value).read_bytes())
        if type(plan) is not dict:
            raise ValueError('Edit plan must be an object')
        root = self.path(assets_dir, directory=True)
        issues, checked = [], []
        durations = {}
        def issue(message):
            issues.append(message)
        def asset(folder, ident):
            if type(ident) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', ident):
                raise ValueError('Invalid media ID')
            matches = [p for p in (root / folder).glob(ident + '.*') if p.suffix.lower() in ('.mp4', '.mov', '.mkv', '.wav', '.m4a', '.mp3')]
            if len(matches) != 1:
                raise ValueError(f'Missing/ambiguous asset {folder}/{ident}')
            path = self.path(str(matches[0]))
            if path not in durations:
                durations[path] = self.probe(path)['duration_s']
            checked.append(str(path.relative_to(self.work)))
            return durations[path]
        recognized = False
        for field in ('selected', 'order'):
            if field not in plan:
                continue
            recognized = True
            ids = plan[field]
            if type(ids) is not list or not ids or len(ids) > 1000:
                issue(f'{field} must be a nonempty ID list')
                continue
            if len({str(x) for x in ids}) != len(ids):
                issue(f'Duplicate IDs in {field}')
            for ident in ids:
                try:
                    asset('clips', ident)
                except ValueError as exc:
                    issue(str(exc))
            if field == 'order':
                available = {p.stem for p in (root / 'clips').glob('*') if p.suffix.lower() in ('.mp4', '.mov', '.mkv')}
                if set(map(str, ids)) != available:
                    issue('order does not use every supplied clip exactly once')
        total = None
        if 'cuts' in plan:
            recognized = True
            cuts = plan['cuts']
            if type(cuts) is not list or not cuts or len(cuts) > 1000:
                issue('cuts must be a nonempty list')
            else:
                total, seen = 0., set()
                for n, cut in enumerate(cuts):
                    try:
                        if type(cut) is not dict:
                            raise ValueError('cut must be an object')
                        ident = cut.get('clip')
                        duration = asset('clips', ident)
                        if ident in seen:
                            issue(f'Duplicate cut for {ident}')
                        seen.add(ident)
                        start = number(cut.get('in_s'), 'in_s')
                        end = number(cut.get('out_s'), 'out_s')
                        if not start < end <= duration + .001:
                            raise ValueError('in/out outside actual clip duration')
                        total += end - start
                    except ValueError as exc:
                        issue(f'cuts[{n}]: {exc}')
                available = {p.stem for p in (root / 'clips').glob('*') if p.suffix.lower() in ('.mp4', '.mov', '.mkv')}
                if seen != available:
                    issue('cuts does not cover every supplied clip')
        if any(k in plan for k in ('dialogue', 'joins', 'music', 'production_sound', 'cards')):
            recognized = True
            for field in ('dialogue', 'joins', 'production_sound', 'cards'):
                if field in plan and (type(plan[field]) is not list or len(plan[field]) > 1000):
                    issue(f'{field} must be a list with at most 1000 entries')
                    return {'valid': False, 'issues': issues, 'checked_assets': sorted(set(checked)), 'planned_duration_s': None}
            for n, line in enumerate(plan.get('dialogue', [])):
                try:
                    duration = asset('takes', line.get('take'))
                    start = number(line.get('in_s'), 'in_s')
                    end = number(line.get('out_s'), 'out_s')
                    number(line.get('at_s'), 'at_s', -3600, 3600)
                    number(line.get('gain_db', 0), 'gain_db', -120, 120)
                    if not start < end <= duration + .001:
                        raise ValueError('Dialogue source range outside take')
                except (ValueError, AttributeError) as exc:
                    issue(f'dialogue[{n}]: {exc}')
            joins = plan.get('joins', [])
            ids = []
            for n, join in enumerate(joins):
                try:
                    ids.append(number(join.get('join'), 'join', 1, 1000, integer=True))
                    if join.get('picture') not in ('cut', 'dip_to_black', 'dissolve', 'other'):
                        issue(f'joins[{n}]: unknown picture transition')
                    if join.get('sound') not in ('hard', 'carries_over', 'prelap', 'under'):
                        issue(f'joins[{n}]: missing/unknown sound transition')
                    for field in ('black_hold_s', 'fade_in_s', 'fade_out_s'):
                        if field in join:
                            number(join[field], field, 0, 180)
                except (ValueError, AttributeError) as exc:
                    issue(f'joins[{n}]: {exc}')
            shots = list((root / 'shots').glob('*.mp4'))
            if shots and (len(ids) != len(set(ids)) or set(ids) != set(range(1, len(shots)))):
                issue('joins must cover every supplied shot boundary exactly once')
            for n, production in enumerate(plan.get('production_sound', [])):
                try:
                    asset('shots', production.get('shot'))
                    if type(production.get('keep')) is not bool:
                        raise ValueError('keep must be boolean')
                    number(production.get('gain_db', 0), 'gain_db', -120, 120)
                except (ValueError, AttributeError) as exc:
                    issue(f'production_sound[{n}]: {exc}')
            for n, card in enumerate(plan.get('cards', [])):
                try:
                    asset('shots', card.get('shot'))
                    if type(card.get('text')) is not str or len(card['text']) > 4096:
                        raise ValueError('Card text must be a bounded string')
                except (ValueError, AttributeError) as exc:
                    issue(f'cards[{n}]: {exc}')
            if 'music' in plan:
                try:
                    # An offset may be negative (delayed entry); do not infer a
                    # hidden preferred alignment or falsely reject J/L cuts.
                    offset = number(plan['music'].get('starts_at_s'), 'starts_at_s', -3600, 3600)
                    duration = self.probe(self.path(str(root / 'music.wav')))['duration_s']
                    if offset >= duration:
                        raise ValueError('Music offset starts after the original music ends')
                except (ValueError, AttributeError) as exc:
                    issue(f'music: {exc}')
        if not recognized:
            issue('Unknown plan format; no structural validation performed')
        return {'valid': not issues, 'issues': issues[:128], 'issues_truncated': len(issues) > 128,
                'checked_assets': sorted(set(checked)), 'planned_duration_s': total,
                'note': 'Structural/file/range checks only. Does not judge selection, aesthetics, speech intelligibility, or prove that rendered content implements the JSON.'}

    def validate_media(self, args):
        keys(args, ('path', 'start_s', 'duration_s', 'expected', 'plan_path', 'assets_dir'))
        if 'path' not in args and 'plan_path' not in args:
            raise ValueError('Provide a media path and/or plan_path')
        result = {'status': 'ok', 'technical_checks': [], 'candidates': {}, 'not_a_reward': True}
        if 'plan_path' in args:
            plan_path = self.path(args['plan_path'])
            if plan_path.stat().st_size > 1024 * 1024:
                raise ValueError('Edit JSON exceeds 1 MiB')
            result['plan'] = self.validate_plan(args['plan_path'], args.get('assets_dir', 'initial_state'))
        if 'path' not in args:
            return result
        path = self.path(args['path'])
        meta = self.probe(path)
        result['media'] = meta
        expected = args.get('expected', {})
        keys(expected, ('duration_s', 'duration_tolerance_s', 'width', 'height', 'fps', 'has_audio'))
        video = next((s for s in meta['streams'] if s['codec_type'] == 'video'), None)
        audio = next((s for s in meta['streams'] if s['codec_type'] == 'audio'), None)
        values = {'duration_s': meta['duration_s'], 'width': video.get('width') if video else None,
                  'height': video.get('height') if video else None,
                  'fps': rate(video.get('avg_frame_rate')) if video else None, 'has_audio': audio is not None}
        for key, target in expected.items():
            if key == 'duration_tolerance_s':
                continue
            if key == 'has_audio':
                if type(target) is not bool:
                    raise ValueError('has_audio must be boolean')
                ok = values[key] == target
            else:
                number(target, key, 0)
                tolerance = number(expected.get('duration_tolerance_s', .1), 'duration_tolerance_s', 0, 2) if key == 'duration_s' else (.01 if key == 'fps' else 0)
                ok = values[key] is not None and abs(values[key] - target) <= tolerance
            result['technical_checks'].append({'field': key, 'expected': target, 'actual': values[key], 'ok': ok})
        planned = result.get('plan', {}).get('planned_duration_s')
        if planned is not None and result['plan'].get('valid'):
            tolerance = number(expected.get('duration_tolerance_s', .1), 'duration_tolerance_s', 0, 2)
            result['technical_checks'].append({'field': 'preview_vs_cuts_duration', 'expected': planned,
                'actual': meta['duration_s'], 'ok': abs(meta['duration_s'] - planned) <= tolerance})
        start = number(args.get('start_s', 0), 'start_s', 0)
        if start >= meta['duration_s']:
            raise ValueError('Scan starts after media ends')
        requested = number(args.get('duration_s', min(MAX_SCAN_SECONDS, meta['duration_s'] - start)), 'duration_s', .05, MAX_SCAN_SECONDS)
        duration = min(requested, meta['duration_s'] - start)
        argv = [shutil.which('ffmpeg') or 'ffmpeg', '-nostdin', '-hide_banner', '-v', 'info', '-xerror',
            '-err_detect', 'explode', '-threads', '1', '-filter_threads', '1', '-protocol_whitelist', 'file,pipe',
            '-ss', str(start), '-i', str(path), '-t', str(duration)]
        if video:
            argv += ['-map', '0:v:0', '-vf', 'blackdetect=d=0.04:pix_th=0.10:pic_th=0.98,freezedetect=n=-60dB:d=0.5',
                     '-fps_mode', 'passthrough']
        if audio:
            argv += ['-map', '0:a:0', '-af', 'silencedetect=n=-50dB:d=0.1']
        if not video and not audio:
            raise ValueError('No decodable audio/video stream')
        argv += ['-progress', 'pipe:1', '-f', 'null', '-']
        progress, log, code = self.run(argv, timeout=90, allow_error=True)
        result['scan'] = {'start_s': start, 'duration_s': duration,
                          'partial': start > 0 or start + duration < meta['duration_s'] - .001,
                          'decode_ok': code == 0, 'audio_stream_scanned': 0 if audio else None,
                          'video_stream_scanned': 0 if video else None}
        result['scan']['decoded_frame_count'] = int(re.findall(rb'frame=(\d+)', progress)[-1]) if re.findall(rb'frame=(\d+)', progress) else None
        if code:
            result['status'] = 'decode_error'
            result['scan']['error'] = log[-1200:]
        black = [[start+float(a), start+float(b)] for a, b in re.findall(r'black_start:([\d.eE+-]+) black_end:([\d.eE+-]+)', log)]
        def intervals(prefix):
            events = re.findall(prefix + r'_(start|end):\s*([\d.eE+-]+)', log)
            active, found = None, []
            for kind, value in events:
                t = max(0, min(duration, float(value)))
                if kind == 'start':
                    active = t
                elif active is not None:
                    found.append([start + active, start + t])
                    active = None
            if active is not None:
                found.append([start + active, start + duration])
            return found
        candidates = {'black_intervals_s': black, 'freeze_intervals_s': intervals('freeze'),
                      'silence_intervals_s': intervals('silence')}
        result['candidates'] = {key: items[:128] for key, items in candidates.items()}
        result['candidates_truncated'] = any(len(items) > 128 for items in candidates.values())
        result['note'] = 'Only supplied expectations and technical evidence are checked. Black, stillness and silence can be intentional; candidates are NOT failures or aesthetic scores. A partial scan does not certify the rest of the file. No reference render, grader, judge or network is used.'
        return result
