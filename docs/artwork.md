# README artwork

The [hero](assets/hero-mechanical.png) and [neural background](assets/background.png) are
AI-generated illustrations, not MaleCNS visualisations.

The [mechanical fly animation](assets/fly-anatomy.gif) is rendered frame by frame in Blender
from a 3D model with titanium armor plates, hexagonal optical lenses, machined fasteners,
piston-driven legs and translucent photonic wings. It is conceptual artwork. Its camera-facing
labels map the eyes to `RetinaEncoder`, the body to the sparse rate-based `ConnectomeRNN`, and
the feet to `ActionDecoder`. The six feet illustrate output collectively; they do not imply six
fixed actions. The frame-to-hex diagram is illustrative, not a recorded rollout.

- [Full-resolution diagram](assets/fly-anatomy.png)
- [Annotated Blender scene](assets/mechanical-fly-annotated.blend)

Regenerate with Blender and FFmpeg:

```bash
blender --background docs/assets/mechanical-fly.blend \
  --python docs/assets/annotate_mechanical_blender.py -- --frames 72
uv run python docs/assets/assemble_anatomy.py
```
