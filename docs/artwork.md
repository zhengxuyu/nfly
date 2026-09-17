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

## Viewer capture

[viewer.gif](assets/viewer.gif) is a 24-second edit of the user's September 17, 2026 screen
recording. It shows a frontal view, layer controls, an oblique view and the zoomed-out anatomy
while the trained head plays Pong. Three excerpts (24-36 s, 40-46 s and 61-67 s) from the
70.54-second source retain their original playback speed; cuts omit pauses between them.
This is recorded activity, not a generated animation or a continuous evaluation episode.

The capture session uses viewer code from `82d227eed4a35cd261f81d202e0d30458c9e4e73` and
policy/retina code from `c68c5c6`, matching `bc-pong-mlp-oldeye.pt`. The historical retina's
triangular sampling layout is visible in the eye panel. The checkpoint's recorded greedy
mean return is +8.3333; the edited clip is not a benchmark of that mean.
Checkpoint SHA-256: `a583904fa299dd5c0bd54ada34e49fb3eab9d50b991f348d00dab08515e8ff51`.

The local capture launcher loads the historical `FlyAgent` with `readout_dim=0`,
`head_hidden=64`, `rnn_steps=4` and this checkpoint, then supplies it to the current viewer's
`Session` / `VizServer` API with `greedy=True`, `seed=500` and port 8002. The retina and model
weights are unchanged. No model, training or viewer source files were edited for the capture.

The source is a 1900 x 870, 60 fps HDR recording (PQ, BT.2020). It was converted to SDR BT.709
with Mobius tone mapping before GIF palette conversion, preventing the washed-out colors of
a direct HDR-to-GIF export. The selected GIF is 1280 x 586 at 12 fps, with a 256-color palette
and Bayer dithering. [viewer.png](assets/viewer.png) is a 1900 x 870 frontal still from the same
recording. The raw video, checkpoint and local capture launcher are not included here.
