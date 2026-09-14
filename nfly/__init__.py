"""nfly - a fly connectome as a recurrent neural network, playing any Gymnasium game.

Layers (each depends only on the ones above it):
    nfly.connectome   data: Janelia MaleCNS release -> Connectome tensors
    nfly.brain        model: ConnectomeRNN (sparse, sign-constrained, learnable gains)
    nfly.interface    senses & muscles: ObservationEncoder / ActionDecoder chosen from gym spaces
    nfly.agent        FlyAgent = encoder -> brain -> decoder
    nfly.suite        GameSuite base class + registry (atari, classic, gym)
    nfly.rl           training algorithms
"""

from .connectome import Connectome, load_malecns, select_subset
from .brain import ConnectomeRNN, InputDrive, stimulate
from .agent import FlyAgent

__all__ = ["Connectome", "load_malecns", "select_subset", "ConnectomeRNN", "InputDrive",
           "stimulate", "FlyAgent"]
