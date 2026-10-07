"""Root-owned, per-attempt image admission shared by native and MCP tools.

Collages cannot hide their actual pixel cost. Semantic panel counts of arbitrary
policy-created bitmaps cannot be recovered; official sheet tools also charge
their declared source-frame count. No prompt/image is rewritten after admission.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path
import threading
import time

from .contract import ID, ParityError, digest

UNIT_PIXELS = 448 * 448
MAX_EDGE = 1792
MAX_PIXELS = MAX_EDGE * MAX_EDGE
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_IMAGES = 999


def vllm_image_limits():
    """User-selected 999 images, matching vLLM 0.26's native default.

    Pin it explicitly so train/eval/provenance admission cannot drift after a
    dependency upgrade. Native video is disabled; tools return sampled images.
    """
    return {'image': MAX_CONTEXT_IMAGES, 'video': 0}


def image_bytes(url):
    if type(url) is not str or not url.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,',
                                                'data:image/webp;base64,', 'data:image/gif;base64,')):
        raise ParityError('Only inline image observations are supported; no URLs/files/PDF/audio')
    encoded = url.split(',', 1)[1]
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 8):
        raise ParityError('Image byte budget exceeded')
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ParityError('Invalid inline image') from exc


def normalize_image(body):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 24_000_000
    with Image.open(io.BytesIO(body)) as source:
        if source.width * source.height > 24_000_000 or getattr(source, 'n_frames', 1) != 1:
            raise ParityError('Oversized/animated image; extract explicit frames first')
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((MAX_EDGE, MAX_EDGE))
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=90)
    return output.getvalue(), image.size


def inline_jpeg(body):
    return 'data:image/jpeg;base64,' + base64.b64encode(body).decode()


class VisualBudget:
    def __init__(self, stage, *, preframes=()):
        if stage not in ('s_1', 's_2', 's_3', 's_4e'):
            raise ParityError('Invalid media stage')
        self.stage = stage
        # Usage telemetry is retained. Every stage is now free of an episode
        # visual quota. check_request verifies provenance and the user-selected
        # request image count; pixels are telemetry, not a separate total quota.
        self.limit = None
        self.spent = 0
        self.approved, self.calls = {}, {}
        self.preframes = set()
        self.lock = threading.Lock()
        # Prepared before starting policy code. Never trust a later /work/frames
        # pathname to exempt an image written by the policy.
        for body in preframes:
            self.preframes.add(hashlib.sha256(body).hexdigest())

    def register(self, request):
        if type(request) is not dict or set(request) != {'call_id', 'images', 'source_units'}:
            raise ParityError('Invalid image registration')
        call_id, urls, source_units = request['call_id'], request['images'], request['source_units']
        if type(call_id) is not str or not ID.fullmatch(call_id):
            raise ParityError('Invalid image call identity')
        if type(urls) is not list or not 1 <= len(urls) <= 32:
            raise ParityError('At most 32 images per tool result')
        if type(source_units) is not int or not 0 <= source_units <= 96:
            raise ParityError('Invalid visual source count')
        identity = digest(request)
        with self.lock:
            if call_id in self.calls:
                old, result = self.calls[call_id]
                if old != identity:
                    raise ParityError('Image call ID reused with different content')
                return result
            if len(self.calls) >= 2048:
                raise ParityError('Image registration count exceeded')
            normalized, cost, records = [], 0, []
            deadline = time.monotonic() + 20
            for url in urls:
                if time.monotonic() > deadline:
                    raise ParityError('Image normalization deadline exceeded; use fewer images')
                raw = image_bytes(url)
                body, (width, height) = normalize_image(raw)
                units = max(1, math.ceil(width * height / UNIT_PIXELS))
                is_preframe = hashlib.sha256(raw).hexdigest() in self.preframes
                cost += 0 if is_preframe else units
                sha = hashlib.sha256(body).hexdigest()
                normalized.append(inline_jpeg(body))
                records.append({'sha256': sha, 'width': width, 'height': height,
                                'pixel_units': units, 'preframe': is_preframe})
            cost = max(cost, source_units)
            if self.limit is not None and self.spent + cost > self.limit:
                raise ParityError('Visual budget exceeded; use fewer/smaller frames or crops')
            self.spent += cost
            self.approved.update({r['sha256']: r for r in records})
            result = {'images': normalized, 'charged_units': cost, 'spent_units': self.spent,
                      'limit_units': self.limit, 'records': records}
            self.calls[call_id] = (identity, result)
            return result

    def check_request(self, body):
        count, pixels = 0, 0
        with self.lock:
            for message in body.get('messages', []):
                content = message.get('content')
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get('type') == 'text':
                        continue
                    if part.get('type') != 'image_url':
                        raise ParityError('Unreviewed non-text/image input modality')
                    value = part.get('image_url', {})
                    url = value.get('url') if isinstance(value, dict) else value
                    sha = hashlib.sha256(image_bytes(url)).hexdigest()
                    record = self.approved.get(sha)
                    if record is None:
                        raise ParityError('Image bypassed the shared tool observation gate')
                    count += 1
                    pixels += record['width'] * record['height']
            if count > MAX_CONTEXT_IMAGES:
                raise ParityError('Actual context exceeds 999 images; compact history first')
        return {'images': count, 'pixels': pixels}


def prepared_frame_bytes(root=Path('/work/frames')):
    """Raw plus legacy 448px thumbnails, enumerated before policy starts."""
    from PIL import Image
    if not root.exists():
        return []
    result, total = [], 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
            continue
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp') or path.stat().st_size > MAX_IMAGE_BYTES:
            continue
        raw = path.read_bytes()
        total += len(raw)
        if len(result) >= 4096 or total > 64 * 1024 * 1024:
            raise ParityError('Prepared frame registry exceeds bounded initialization size')
        result.append(raw)
        with Image.open(io.BytesIO(raw)) as source:
            source.thumbnail((448, 448))
            stream = io.BytesIO()
            source.convert('RGB').save(stream, format='JPEG', quality=82)
            result.append(stream.getvalue())
    return result
