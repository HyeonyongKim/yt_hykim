"""
Binary file inspection.

Detects known binary formats by reading headers:
  - Tipsy: 32-byte header with (double time, int ntotal, int ndim=3, int ngas, int ndark, int nstar, int pad)
"""

import logging
import os
import struct
from pathlib import Path

from yt_universal.inspectors.base import (
    ContainerType,
    DatasetSignature,
    LayoutType,
)

mylog = logging.getLogger(__name__)


def inspect_binary(path: str, sig: DatasetSignature):
    """Populate a DatasetSignature from a binary file's header."""
    p = Path(path)
    if not p.is_file():
        return

    file_size = p.stat().st_size

    # Try Tipsy format
    if _detect_tipsy(path, file_size, sig):
        return

    mylog.debug("inspect_binary: no known binary format detected for %s", path)


def _detect_tipsy(path: str, file_size: int, sig: DatasetSignature) -> bool:
    """Detect Tipsy binary format (ChaNGa, Gasoline, PKDGRAV).

    Tipsy header (32 bytes, big-endian):
      double time (8 bytes)
      int    ntotal (4 bytes)
      int    ndim (4 bytes)  -- must be 3
      int    ngas (4 bytes)
      int    ndark (4 bytes)
      int    nstar (4 bytes)
      int    pad (4 bytes)
    """
    if file_size < 32:
        return False

    try:
        with open(path, "rb") as f:
            header = f.read(32)
    except OSError:
        return False

    # Try big-endian first (most common for Tipsy)
    for endian, prefix in [(">", "big"), ("<", "little")]:
        try:
            time_val = struct.unpack(f"{endian}d", header[:8])[0]
            ntotal, ndim, ngas, ndark, nstar, pad = struct.unpack(
                f"{endian}6i", header[8:32]
            )
        except struct.error:
            continue

        # Validate: ndim must be 3, counts must be non-negative, must sum correctly
        if ndim != 3:
            continue
        if ntotal < 0 or ngas < 0 or ndark < 0 or nstar < 0:
            continue
        if ngas + ndark + nstar != ntotal:
            continue

        # Validate file size: header + gas*48 + dark*36 + star*44
        expected = 32 + ngas * 48 + ndark * 36 + nstar * 44
        if expected != file_size:
            continue

        # Looks like a valid Tipsy file
        sig.container_type = ContainerType.BINARY
        sig.layout_type = LayoutType.PARTICLE_ONLY
        sig.candidate_family = "tipsy"
        sig.confidence = 0.90
        sig.yt_hint = "Tipsy"
        # Store header info for the adapter
        sig.hdf5_datasets = {
            "tipsy_header": (ntotal, None),
            "tipsy_endian": (prefix, None),
            "tipsy_ngas": (ngas, None),
            "tipsy_ndark": (ndark, None),
            "tipsy_nstar": (nstar, None),
            "tipsy_time": (time_val, None),
        }
        mylog.debug(
            "detect_tipsy: %s-endian, ntotal=%d, ngas=%d, ndark=%d, nstar=%d",
            prefix, ntotal, ngas, ndark, nstar,
        )
        return True

    return False
