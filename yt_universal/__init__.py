"""
yt_universal: A universal loader layer for yt.

Reuses existing yt frontends first, then falls back to universal parsing
for unsupported formats. Exposes a stable universal field layer on top.
"""

from yt_universal.loader import load_universal

__version__ = "0.1.0"
__all__ = ["load_universal"]
