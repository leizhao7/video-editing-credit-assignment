"""Read-only local demonstration against explicitly supplied original task assets.

Writes a new evidence directory only. Never edits task inputs or renders a task
answer. Used to visually inspect the contact sheet/crop tool itself.
"""
import argparse
import base64
import json
from pathlib import Path

from .editing_tools import EditingTools
from .store import atomic_bytes
from .contract import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.task_root).resolve()
    target = Path(args.output)
    if target.exists():
        raise RuntimeError('Refusing to overwrite prior demonstration evidence')
    target.mkdir(parents=True)
    media = EditingTools(root)
    clips = sorted((root / 'initial_state/shots').glob('*.mp4'))[:8]
    if not clips:
        raise ValueError('No public shot videos')
    items = [{'path': str(path.relative_to(root)), 'frame_index': len(media.frame_index(path)['pts']) // 2} for path in clips]
    examples = {
        'contact_sheet': media.contact_sheet({'items': items, 'columns': 4, 'cell_width': 448}),
        'cut_neighborhood': media.inspect_frames({'path': str(clips[0].relative_to(root)),
            'center_s': media.frame_index(clips[0])['duration_s'] / 2, 'before': 3, 'after': 4}),
        'crop': media.inspect_frames({'path': str(clips[0].relative_to(root)), 'frame_indices': [items[0]['frame_index']],
                                      'crop': [.2, .15, .6, .7], 'layout': 'individual', 'edge': 896}),
    }
    for name, result in examples.items():
        for index, image in enumerate(result.pop('images')):
            atomic_bytes(target / f'{name}_{index}.jpg', base64.b64decode(image['jpeg_b64']))
        atomic_bytes(target / f'{name}.json', canonical(result))
    print(json.dumps({'status': 'generated', 'output': str(target), 'examples': list(examples), 'training_started': False}))


if __name__ == '__main__':
    main()
