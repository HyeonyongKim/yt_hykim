"""
Adapter registry: matches DatasetSignatures to the best adapter.
"""

import logging

from yt_universal.adapters.base import BaseAdapter
from yt_universal.inspectors.base import DatasetSignature

mylog = logging.getLogger(__name__)

_registered_adapters: list[type[BaseAdapter]] = []


def register_adapter(cls: type[BaseAdapter]) -> type[BaseAdapter]:
    """Class decorator to register an adapter."""
    _registered_adapters.append(cls)
    mylog.debug("Registered adapter: %s", cls.__name__)
    return cls


def match(sig: DatasetSignature) -> BaseAdapter | None:
    """Find the best adapter for the given signature.

    Returns an instance of the highest-priority adapter that can handle
    the signature, or None if no adapter matches.
    """
    candidates = [
        cls for cls in _registered_adapters if cls.can_handle(sig)
    ]
    if not candidates:
        return None

    # Sort by priority (highest first)
    candidates.sort(key=lambda cls: cls.priority(), reverse=True)
    best = candidates[0]
    mylog.info("Matched adapter: %s for %s", best.__name__, sig.path)
    return best()
