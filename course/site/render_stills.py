"""Render the course stills from the film's engine, as listed in stills/captions.json.

    python3 course/site/render_stills.py                 # every still
    python3 course/site/render_stills.py 22.12 147.80    # just these times

Each entry is keyed by song time ("22.12"). Two optional fields take creative liberty, for stills only (the film is
never rendered this way):

  "fovk": 1.3        widen the lens 1.3×, when the phrase runs off the edge of the frame
  "sing": [288, 289] re-time lyric words 288..289 (global indices into data/lyrics.json) to have just been sung at
                     this time, when the phrase would otherwise end after a cut or on a blurred frame

Needs bun and the app's dependencies (cd app && bun install). Writes 1600-px JPEGs to stills/tSSS.ss.jpg.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SITE = Path(__file__).resolve().parent
APP = SITE.parents[1] / 'app'
CAPS = json.loads((SITE / 'stills' / 'captions.json').read_text())


def main(times: list[str]) -> None:
    keys = [f'{float(t):.2f}' for t in times] if times else sorted(CAPS, key=float)
    missing = [k for k in keys if k not in CAPS]
    if missing:
        sys.exit(f'not in captions.json: {missing}')
    sing = ','.join(f'{k}:{CAPS[k]["sing"][0]}-{CAPS[k]["sing"][1]}' for k in keys if 'sing' in CAPS[k])
    fovk = ','.join(f'{k}:{CAPS[k]["fovk"]}' for k in keys if 'fovk' in CAPS[k])
    with tempfile.TemporaryDirectory() as out:
        cmd = ['bun', 'scripts/render.ts', 'stills', '--samples', 'auto', '--max-samples', '64', '--out', out, '--t', ','.join(keys)]
        if sing:
            cmd += ['--sing', sing]
        if fovk:
            cmd += ['--fovk', fovk]
        subprocess.run(cmd, cwd=APP, check=True)
        for k in keys:
            src = Path(out) / f'f_{float(k):07.2f}.png'
            dst = SITE / 'stills' / f't{float(k):06.2f}.jpg'
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(src), '-vf', 'scale=1600:-2', '-q:v', '4', str(dst)], check=True)
            print('wrote', dst.relative_to(SITE))


if __name__ == '__main__':
    main(sys.argv[1:])
