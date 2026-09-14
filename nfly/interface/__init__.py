"""Interface layer: the fly's senses and muscles for a Gymnasium environment.

Encoders map observations onto input-neuron drive; decoders map readout-neuron activity
onto action distributions.  Both are chosen from the env's spaces, so the brain in between
never sees a game-specific type.
"""

from .encoders import ObservationEncoder, RetinaEncoder, ImageProjectionEncoder, VectorEncoder
from .decoders import ActionDecoder, DiscreteDecoder, BoxDecoder, RunningNorm, default_readout_nodes
from .retina import Retina, build_retina

__all__ = ["ObservationEncoder", "RetinaEncoder", "ImageProjectionEncoder", "VectorEncoder",
           "ActionDecoder", "DiscreteDecoder", "BoxDecoder", "RunningNorm", "default_readout_nodes", "Retina", "build_retina"]
