"""Model layer: the connectome-constrained recurrent network.  Knows nothing about games."""

from .rnn import ConnectomeRNN, InputDrive, Weights
from .stimulate import StimulationResult, stimulate

__all__ = ["ConnectomeRNN", "InputDrive", "Weights", "StimulationResult", "stimulate"]
