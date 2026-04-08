from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import match, register_adapter

# Import adapters so they auto-register
from yt_universal.adapters import simple_hdf5  # noqa: F401

__all__ = ["BaseAdapter", "match", "register_adapter"]