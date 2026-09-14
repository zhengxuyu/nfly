"""Assemble the annotated Blender frames; requires FFmpeg on PATH."""
from pathlib import Path
import shutil
import subprocess

FRAMES = Path('/tmp/nfly-annotated')
ASSETS = Path(__file__).parent.resolve()
FRAME_COUNT = 72


def main():
    if not shutil.which('ffmpeg'):
        raise SystemExit('Install FFmpeg and make ffmpeg available on PATH.')
    if any(not (FRAMES/f'frame-{i:03d}.png').is_file() for i in range(FRAME_COUNT)):
        raise SystemExit('Render all 72 annotated Blender frames first.')
    filters = ('[0:v]scale=1120:700:flags=lanczos,split[a][b];'
               '[a]palettegen=stats_mode=diff:reserve_transparent=0[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', '25/2', '-i', str(FRAMES/'frame-%03d.png'),
                    '-frames:v', str(FRAME_COUNT), '-filter_complex', filters,
                    '-loop', '0', str(ASSETS/'fly-anatomy.gif')], check=True)
    shutil.copyfile(FRAMES/'frame-000.png', ASSETS/'fly-anatomy.png')
    shutil.copyfile(FRAMES/'mechanical-fly-annotated.blend', ASSETS/'mechanical-fly-annotated.blend')
    print('Saved annotated GIF, full-size diagram and editable Blender scene.')


if __name__ == '__main__':
    main()
