"""Assemble Blender-rendered PNG frames with FFmpeg.

First run render_fly_blender.py with Blender, then:
uv run python docs/assets/render_fly.py
"""
from pathlib import Path
import shutil
import subprocess


FRAME_DIR = Path('/tmp/nfly-detailed')
OUTPUT = Path(__file__).parent
FRAME_COUNT = 96


def main():
    paths = [FRAME_DIR / f'frame-{i:03d}.png' for i in range(FRAME_COUNT)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f'Render the Blender frames first; missing {len(missing)} frames.')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('Install FFmpeg and make ffmpeg available on PATH.')
    # Weight changing regions so the fly retains detail; ordered dithering is stable.
    filters = ('[0:v]split[a][b];'
               '[a]palettegen=stats_mode=diff:reserve_transparent=0[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', '25/2', '-i', str(FRAME_DIR / 'frame-%03d.png'),
                    '-frames:v', str(FRAME_COUNT), '-filter_complex', filters,
                    '-loop', '0', str(OUTPUT / 'fly-3d.gif')], check=True)
    shutil.copyfile(paths[0], OUTPUT / 'fly-3d-preview.png')
    print(f'Saved {FRAME_COUNT} frames, {FRAME_COUNT * 80 / 1000:.2f}s loop.')


if __name__ == '__main__':
    main()
