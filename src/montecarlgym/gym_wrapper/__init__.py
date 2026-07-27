"""Optional-dependency-free Gymnasium transaction adapters."""

from .base import EnvSnapshot, MCTSEnvWrapper, SnapshotStrategy
from .deepcopy import DeepCopySnapshotStrategy
from .snapshot import NativeSnapshotStrategy, SnapshotError

__all__ = [
    "DeepCopySnapshotStrategy",
    "EnvSnapshot",
    "MCTSEnvWrapper",
    "NativeSnapshotStrategy",
    "SnapshotError",
    "SnapshotStrategy",
]
