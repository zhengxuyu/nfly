"""Assemble the Blender turntable into a GIF; FFmpeg must be on PATH.

uv run python docs/assets/assemble_mechanical.py
"""
from pathlib import Path
import shutil
import subprocess


FRAMES = Path('/tmp/nfly-mechanical-blender')
ASSETS = Path(__file__).parent.resolve()
FRAME_COUNT = 96


def main():
    if not shutil.which('ffmpeg'):
        raise SystemExit('Install FFmpeg and make ffmpeg available on PATH.')
    missing = [i for i in range(FRAME_COUNT) if not (FRAMES / f'frame-{i:03d}.png').is_file()]
    if missing:
        raise SystemExit(f'Render the Blender scene first; {len(missing)} frames are missing.')
    filters = ('[0:v]split[a][b];[a]palettegen=stats_mode=diff:reserve_transparent=0[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', '25/2', '-i', str(FRAMES / 'frame-%03d.png'),
                    '-frames:v', str(FRAME_COUNT), '-filter_complex', filters,
                    '-loop', '0', str(ASSETS / 'fly-mechanical-blender.gif')], check=True)
    shutil.copyfile(FRAMES / 'frame-000.png', ASSETS / 'fly-mechanical-blender.png')
    shutil.copyfile(FRAMES / 'mechanical-fly.blend', ASSETS / 'mechanical-fly.blend')
    print('Saved mechanical GIF, full-resolution still, and editable Blender scene.')


if __name__ == '__main__':
    main()
