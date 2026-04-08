from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import match, register_adapter

# Import adapters so they auto-register
from yt_universal.adapters import simple_hdf5  # noqa: F401
from yt_universal.adapters import gadget_family  # noqa: F401
from yt_universal.adapters import amr_hdf5  # noqa: F401
from yt_universal.adapters import ascii_table  # noqa: F401
from yt_universal.adapters import manifest  # noqa: F401
from yt_universal.adapters import gamer_hdf5  # noqa: F401
from yt_universal.adapters import tipsy  # noqa: F401
from yt_universal.adapters import enzo_dir  # noqa: F401
from yt_universal.adapters import ramses_dir  # noqa: F401

__all__ = ["BaseAdapter", "match", "register_adapter"]