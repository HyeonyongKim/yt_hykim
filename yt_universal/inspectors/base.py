"""
Dataset inspection: fast fingerprinting of files without reading full data.

Produces a DatasetSignature that downstream adapters use to decide
whether they can handle the file.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

mylog = logging.getLogger(__name__)

# HDF5 magic bytes: \x89HDF\r\n\x1a\n
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ContainerType(Enum):
    HDF5 = auto()
    ASCII = auto()
    BINARY = auto()
    DIRECTORY = auto()
    UNKNOWN = auto()


class LayoutType(Enum):
    UNIGRID = auto()
    AMR = auto()
    PARTICLE_ONLY = auto()
    MIXED = auto()
    UNSTRUCTURED = auto()
    UNKNOWN = auto()


@dataclass
class DatasetSignature:
    """Lightweight fingerprint of a dataset produced by inspection."""

    path: str
    container_type: ContainerType = ContainerType.UNKNOWN
    layout_type: LayoutType = LayoutType.UNKNOWN
    candidate_family: str = "unknown"
    confidence: float = 0.0
    yt_hint: str | None = None
    needs_manifest: bool = False

    # HDF5-specific metadata (populated by HDF5 inspector)
    hdf5_root_groups: list[str] = field(default_factory=list)
    hdf5_datasets: dict[str, tuple] = field(default_factory=dict)
    # Maps dataset path -> (shape, dtype)

    # File metadata
    extension: str = ""
    file_size: int = 0
    is_directory: bool = False
    sibling_files: list[str] = field(default_factory=list)


def _detect_container_type(path: str) -> ContainerType:
    """Detect the container type from magic bytes and extension."""
    p = Path(path)

    if p.is_dir():
        return ContainerType.DIRECTORY

    if not p.is_file():
        return ContainerType.UNKNOWN

    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError:
        return ContainerType.UNKNOWN

    if header[:8] == HDF5_MAGIC:
        return ContainerType.HDF5

    # Extension-based fallback
    ext = p.suffix.lower()
    if ext in (".txt", ".csv", ".dat", ".tab", ".asc"):
        return ContainerType.ASCII

    # Check if the file looks like text
    try:
        with open(path, "r") as f:
            f.read(512)
        return ContainerType.ASCII
    except (UnicodeDecodeError, OSError):
        pass

    return ContainerType.BINARY


def inspect_path(path: str) -> DatasetSignature:
    """Inspect a file or directory and produce a DatasetSignature.

    This is the first step in the universal loading pipeline.
    It should be cheap and never read the full dataset.

    Parameters
    ----------
    path : str
        Path to the dataset file or directory.

    Returns
    -------
    DatasetSignature
        A lightweight fingerprint of the dataset.
    """
    path = os.path.expanduser(path)
    p = Path(path)
    sig = DatasetSignature(path=path)

    if not p.exists():
        mylog.warning("inspect_path: path does not exist: %s", path)
        return sig

    sig.is_directory = p.is_dir()
    sig.extension = p.suffix.lower()

    if not sig.is_directory:
        sig.file_size = p.stat().st_size

    # Gather sibling files
    parent = p.parent if not sig.is_directory else p
    try:
        sig.sibling_files = [f.name for f in parent.iterdir() if f.is_file()]
    except OSError:
        pass

    # Detect container type
    sig.container_type = _detect_container_type(path)

    # Delegate to type-specific inspectors
    if sig.container_type == ContainerType.HDF5:
        from yt_universal.inspectors.hdf5 import inspect_hdf5

        inspect_hdf5(path, sig)
    elif sig.container_type == ContainerType.ASCII:
        from yt_universal.inspectors.ascii import inspect_ascii

        inspect_ascii(path, sig)

    mylog.info(
        "inspect_path: %s -> container=%s, layout=%s, family=%s (%.0f%% confidence)",
        p.name,
        sig.container_type.name,
        sig.layout_type.name,
        sig.candidate_family,
        sig.confidence * 100,
    )
    return sig
