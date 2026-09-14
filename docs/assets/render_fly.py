"""Render the conceptual 3D turntable GIF with Pillow (no connectome data).

Run from the repository root: uv run --with pillow python docs/assets/render_fly.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw


SIZE = (800, 480)
SCALE = 83
BG = (5, 16, 27)


def rotate(point, angle):
    x, y, z = point
    x, y = x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)
    return x, y * 0.72 - z * 0.69, y * 0.69 + z * 0.72


def project(point):
    x, depth, z = point
    return (400 + x * SCALE, 225 - z * SCALE)


def ellipsoid(center, radii, color):
    faces = []
    for i in range(16):
        for j in range(32):
            corners = []
            for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                phi = math.pi * (i + di) / 16
                theta = 2 * math.pi * (j + dj) / 32
                unit = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))
                corners.append(tuple(c + r * u for c, r, u in zip(center, radii, unit)))
            shade = 0.5 + 0.5 * math.sin(math.pi * (i + 0.5) / 16)
            tint = tuple(int(c * shade) for c in color)
            faces.append((corners, tint, None))
    return faces


def fly_mesh():
    faces = []
    parts = [((0, -0.05, 0.25), (0.48, 0.63, 0.46), (57, 137, 143)),
             ((0, -0.85, 0.31), (0.43, 0.36, 0.37), (74, 157, 158)),
             ((0, 0.83, 0.13), (0.43, 0.91, 0.37), (139, 108, 58))]
    for side in (-1, 1):
        parts.append(((side * 0.34, -0.94, 0.39), (0.23, 0.26, 0.28), (216, 91, 64)))
    for center, radii, color in parts:
        faces.extend(ellipsoid(center, radii, color))
    for side in (-1, 1):
        for y, reach in ((-0.4, -1.25), (0.05, 0.3), (0.42, 1.48)):
            points = [(side * 0.3, y, 0.1), (side * 0.85, y + 0.1, -0.25),
                      (side * 1.18, reach, -0.7), (side * 1.46, reach + 0.2, -0.74)]
            faces.append((points, (126, 183, 176), 3))
        faces.append(([(side * 0.16, -1.1, 0.4), (side * 0.27, -1.43, 0.5),
                       (side * 0.4, -1.55, 0.64)], (146, 200, 193), 2))
        outline = [(side * (0.3 + 1.05 * math.sin(t)), 0.1 + 1.95 * (1 - math.cos(t)) / 2,
                    0.57 + 0.08 * math.sin(t)) for t in [math.pi * k / 24 for k in range(49)]]
        faces.append((outline, (28, 64, 78), None))
        faces.append((outline + [outline[0]], (102, 180, 192), 2))
        for k in (7, 12, 17, 23):
            faces.append(([(side * 0.3, 0.1, 0.57), outline[k]], (65, 123, 141), 1))
    return faces


def render_frame(mesh, angle):
    image = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.ellipse((184, 339, 616, 398), outline=(26, 63, 75), width=1)
    draw.ellipse((220, 348, 580, 389), outline=(17, 42, 56), width=1)
    transformed = [(list(map(lambda p: rotate(p, angle), points)), color, width)
                   for points, color, width in mesh]
    transformed.sort(key=lambda item: sum(p[1] for p in item[0]) / len(item[0]), reverse=True)
    for points, color, width in transformed:
        points = list(map(project, points))
        if width:
            draw.line(points, fill=color, width=width, joint='curve')
        else:
            draw.polygon(points, fill=color)
    draw.text((28, 26), 'N F L Y   /   3 D   S T U D Y', fill=(148, 218, 220))
    draw.text((28, 448), 'CONCEPTUAL GEOMETRY / NOT A BIOLOGICAL RECONSTRUCTION', fill=(96, 141, 155))
    return image


def main():
    mesh = fly_mesh()
    frames = [render_frame(mesh, 2 * math.pi * i / 72 + 0.5) for i in range(72)]
    target = Path(__file__).with_name('fly-3d.gif')
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print(f'{target}: {target.stat().st_size:,} bytes, 72 frames')


if __name__ == '__main__':
    main()
