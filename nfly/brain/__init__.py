"""Model layer: the connectome-constrained recurrent network.  Knows nothing about games."""

from .rnn import ConnectomeRNN, InputDrive
from .stimulate import stimulate, format_report

__all__ = ["ConnectomeRNN", "InputDrive", "stimulate", "format_report"]
