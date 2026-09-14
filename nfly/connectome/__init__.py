"""Data layer: turn a connectome release into index/weight/sign tensors plus a neuron table."""

from .base import Connectome, build_connectome
from .malecns import load_malecns
from .subsets import SUBSETS, select_subset
from .synthetic import write_synthetic

__all__ = ["Connectome", "build_connectome", "load_malecns", "SUBSETS", "select_subset", "write_synthetic"]
