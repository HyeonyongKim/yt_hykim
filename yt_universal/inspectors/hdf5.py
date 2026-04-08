"""
HDF5-specific dataset inspection.

Reads root groups, dataset shapes, and attributes to classify the
HDF5 file as unigrid mesh, AMR, particle-only, or mixed.
"""

import logging

import h5py
import numpy as np

from yt_universal.inspectors.base import DatasetSignature, LayoutType

mylog = logging.getLogger(__name__)

# Known yt-supported HDF5 family signatures (group names / attributes)
GADGET_LIKE_GROUPS = {"Header", "PartType0", "PartType1"}
ENZO_LIKE_ATTRS = {"HydroMethod", "TopGridDimensions"}
FLASH_LIKE_VARS = {"dens", "temp", "pres", "velx"}
GAMER_LIKE_GROUPS = {"Info", "Tree"}
AREPO_LIKE_GROUPS = {"Header", "PartType0", "Config"}


def _collect_datasets(group, prefix="", result=None):
    """Recursively collect all HDF5 datasets with their shapes and dtypes."""
    if result is None:
        result = {}
    for key in group:
        item = group[key]
        full_path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            result[full_path] = (item.shape, item.dtype)
        elif isinstance(item, h5py.Group):
            _collect_datasets(item, full_path, result)
    return result


def _classify_layout(datasets):
    """Classify layout type based on dataset shapes.

    Parameters
    ----------
    datasets : dict
        Maps dataset path -> (shape, dtype).

    Returns
    -------
    LayoutType
        The detected layout type.
    """
    shapes = [shape for shape, _dtype in datasets.values()]

    if not shapes:
        return LayoutType.UNKNOWN

    # Check for 3D arrays (mesh data)
    has_3d = any(len(s) == 3 and all(d > 1 for d in s) for s in shapes)
    # Check for 1D arrays (particle data)
    has_1d = any(len(s) == 1 and s[0] > 1 for s in shapes)
    # Check for 2D arrays with shape (N, 3) — likely particle positions
    has_particle_2d = any(
        len(s) == 2 and s[1] in (2, 3) and s[0] > 1 for s in shapes
    )

    # Check for multiple 3D arrays with different shapes (AMR-like)
    shapes_3d = [s for s in shapes if len(s) == 3 and all(d > 1 for d in s)]
    unique_3d_shapes = set(shapes_3d)

    if has_3d and (has_1d or has_particle_2d):
        return LayoutType.MIXED
    elif has_3d:
        if len(unique_3d_shapes) > 1:
            return LayoutType.AMR
        return LayoutType.UNIGRID
    elif has_1d or has_particle_2d:
        return LayoutType.PARTICLE_ONLY
    else:
        return LayoutType.UNKNOWN


def _detect_family(root_groups, root_attrs, datasets):
    """Try to match the HDF5 structure to a known simulation family.

    Returns
    -------
    tuple of (family_name: str, yt_hint: str | None, confidence: float)
    """
    group_set = set(root_groups)

    # Gadget-like: has Header + PartType* groups
    if "Header" in group_set and any(g.startswith("PartType") for g in group_set):
        # Distinguish arepo vs gadget vs gizmo
        if "Config" in group_set or "Parameters" in group_set:
            return "arepo_like", "Arepo", 0.85
        return "gadget_like", "Gadget", 0.80

    # FLASH-like: root-level datasets with names like dens, temp, etc.
    root_dataset_names = {
        path.split("/")[0] for path in datasets if "/" not in path
    }
    if root_dataset_names & FLASH_LIKE_VARS:
        return "flash_like", "FLASH", 0.75

    # GAMER-like
    if "Info" in group_set and "Tree" in group_set:
        return "gamer_like", "GAMER", 0.75

    # Enzo-like: check root attributes
    attr_set = set(root_attrs)
    if attr_set & ENZO_LIKE_ATTRS:
        return "enzo_like", "Enzo", 0.75

    # Simple generic HDF5 — has 3D arrays but no recognized family
    has_3d = any(
        len(shape) == 3 and all(d > 1 for d in shape)
        for shape, _dtype in datasets.values()
    )
    if has_3d:
        return "simple_hdf5", None, 0.50

    # Generic HDF5 with particle-like data
    has_particles = any(
        len(shape) == 1 and shape[0] > 1
        for shape, _dtype in datasets.values()
    )
    if has_particles:
        return "generic_hdf5_particle", None, 0.40

    return "unknown_hdf5", None, 0.10


def inspect_hdf5(path, sig: DatasetSignature):
    """Populate an existing DatasetSignature with HDF5-specific metadata.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    sig : DatasetSignature
        The signature object to populate (modified in place).
    """
    try:
        with h5py.File(path, "r") as f:
            # Root groups
            sig.hdf5_root_groups = list(f.keys())

            # Collect all datasets (limit depth to avoid very deep trees)
            sig.hdf5_datasets = _collect_datasets(f)

            # Root attributes
            root_attrs = list(f.attrs.keys())

            # Classify layout
            sig.layout_type = _classify_layout(sig.hdf5_datasets)

            # Detect family
            family, yt_hint, confidence = _detect_family(
                sig.hdf5_root_groups, root_attrs, sig.hdf5_datasets
            )
            sig.candidate_family = family
            sig.yt_hint = yt_hint
            sig.confidence = confidence

    except Exception as exc:
        mylog.warning("inspect_hdf5: failed to read %s: %s", path, exc)
        sig.candidate_family = "corrupt_hdf5"
        sig.confidence = 0.0
