"""
Directory-based dataset inspection.

Scans directories for known simulation output patterns:
  - Enzo: hierarchy file + *.cpu* HDF5 files
  - RAMSES: info_*.txt + amr_*.out* + hydro_*.out*
"""

import logging
import os
import re
from pathlib import Path

from yt_universal.inspectors.base import (
    ContainerType,
    DatasetSignature,
    HDF5_MAGIC,
    LayoutType,
)

mylog = logging.getLogger(__name__)


def inspect_directory(path: str, sig: DatasetSignature):
    """Populate a DatasetSignature from a directory's contents."""
    p = Path(path)
    if not p.is_dir():
        return

    files = list(p.iterdir())
    file_names = [f.name for f in files if f.is_file()]
    sig.sibling_files = file_names

    # Try Enzo pattern: *.hierarchy or hierarchy-like file + *.cpu* files
    if _detect_enzo(p, file_names, sig):
        return

    # Try RAMSES pattern: info_*.txt + amr_*.out* + hydro_*.out*
    if _detect_ramses(p, file_names, sig):
        return

    mylog.debug("inspect_directory: no known pattern in %s", path)


def _detect_enzo(p: Path, file_names: list, sig: DatasetSignature) -> bool:
    """Detect Enzo directory-based output."""
    # Look for *.hierarchy file OR a file with cpu* siblings
    hierarchy_file = None
    cpu_files = []

    for name in file_names:
        if name.endswith(".hierarchy"):
            hierarchy_file = name
        if ".cpu" in name and not name.endswith(".hierarchy"):
            cpu_files.append(name)

    # Also check: a parameter file + cpu files (Enzo sometimes uses different naming)
    # Look for a plain text file with matching cpu files
    if not hierarchy_file and cpu_files:
        base = cpu_files[0].split(".cpu")[0]
        if base in file_names:
            hierarchy_file = base  # The parameter file itself
        if base + ".hierarchy" in file_names:
            hierarchy_file = base + ".hierarchy"

    if not hierarchy_file and not cpu_files:
        # Check if any file looks like an Enzo parameter file
        for name in file_names:
            full_path = p / name
            if full_path.is_file() and full_path.stat().st_size < 100000:
                try:
                    with open(full_path, "r") as f:
                        content = f.read(1000)
                    if "NumberOfBaryonFields" in content or "TopGridDimension" in content:
                        hierarchy_file = name
                        break
                except (UnicodeDecodeError, OSError):
                    continue

    if hierarchy_file or len(cpu_files) > 3:
        sig.container_type = ContainerType.DIRECTORY
        sig.layout_type = LayoutType.AMR
        sig.candidate_family = "enzo_dir"
        sig.confidence = 0.80 if hierarchy_file else 0.50
        # Store the hierarchy file path for the adapter
        sig.hdf5_root_groups = [hierarchy_file] if hierarchy_file else []
        sig.hdf5_datasets = {"cpu_files": (len(cpu_files), None)}
        mylog.debug("detect_enzo: found hierarchy=%s, %d cpu files", hierarchy_file, len(cpu_files))
        return True

    return False


def _detect_ramses(p: Path, file_names: list, sig: DatasetSignature) -> bool:
    """Detect RAMSES directory-based output."""
    has_info = any(re.match(r"info_\d+\.txt", n) for n in file_names)
    has_amr = any(re.match(r"amr_\d+\.out\d+", n) for n in file_names)
    has_hydro = any(re.match(r"hydro_\d+\.out\d+", n) for n in file_names)
    has_part = any(re.match(r"part_\d+\.out\d+", n) for n in file_names)

    if has_info and (has_amr or has_hydro):
        sig.container_type = ContainerType.DIRECTORY
        sig.layout_type = LayoutType.AMR
        sig.candidate_family = "ramses_dir"
        sig.confidence = 0.85
        # Store info file for the adapter
        info_file = next(n for n in file_names if re.match(r"info_\d+\.txt", n))
        sig.hdf5_root_groups = [info_file]
        n_hydro = sum(1 for n in file_names if re.match(r"hydro_\d+\.out\d+", n))
        n_amr = sum(1 for n in file_names if re.match(r"amr_\d+\.out\d+", n))
        sig.hdf5_datasets = {"hydro_files": (n_hydro, None), "amr_files": (n_amr, None)}
        mylog.debug("detect_ramses: info=%s, %d hydro, %d amr files", info_file, n_hydro, n_amr)
        return True

    return False
