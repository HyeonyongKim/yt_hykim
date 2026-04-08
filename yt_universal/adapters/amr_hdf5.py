"""
Adapter for HDF5 files containing AMR (Adaptive Mesh Refinement) data.

Handles HDF5 files with multiple 3D arrays of different shapes,
interpreting them as grid patches at different refinement levels.
Converts them into yt AMR stream datasets via load_amr_grids.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature, LayoutType
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, GridPatch, UnitSpec

mylog = logging.getLogger(__name__)


def _infer_grid_patches(f, sig):
    """Infer AMR grid patches from HDF5 structure.

    Looks for groups that contain 3D arrays and tries to determine
    the grid hierarchy. Supports two common patterns:

    1. Named level groups: Level_0/, Level_1/, etc. each containing
       grid datasets with 3D arrays.
    2. Flat grid datasets: grid_0000, grid_0001, etc. with attributes
       specifying level, left_edge, right_edge.
    """
    patches = []
    field_names = set()

    # Pattern 1: Look for level-based groups
    level_groups = {}
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Group):
            # Check if group name suggests a level
            name_lower = key.lower()
            if "level" in name_lower or "lev" in name_lower:
                # Try to extract level number
                for part in name_lower.replace("_", " ").split():
                    if part.isdigit():
                        level_groups[int(part)] = item
                        break

    if level_groups:
        return _parse_level_groups(level_groups)

    # Pattern 2: Look for grid groups with metadata attributes
    grid_groups = {}
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Group):
            if "left_edge" in item.attrs or "LeftEdge" in item.attrs:
                grid_groups[key] = item

    if grid_groups:
        return _parse_grid_groups(grid_groups)

    # Pattern 3: Multiple 3D arrays of different shapes at root level
    # Treat each unique shape as a different refinement level
    return _parse_flat_amr(f, sig)


def _parse_level_groups(level_groups):
    """Parse AMR from level-organized groups."""
    patches = []
    field_names = set()

    for level in sorted(level_groups):
        grp = level_groups[level]
        for key in grp:
            item = grp[key]
            if isinstance(item, h5py.Group):
                # Sub-group = individual grid patch
                patch_data = {}
                shape = None
                for ds_name in item:
                    ds = item[ds_name]
                    if isinstance(ds, h5py.Dataset) and len(ds.shape) == 3:
                        patch_data[("gas", ds_name)] = ds[()]
                        field_names.add(ds_name)
                        shape = ds.shape
                if shape is not None:
                    le = np.array(item.attrs.get("left_edge", [0, 0, 0]), dtype="float64")
                    re = np.array(item.attrs.get("right_edge", [1, 1, 1]), dtype="float64")
                    patches.append(GridPatch(
                        left_edge=le,
                        right_edge=re,
                        dimensions=np.array(shape),
                        level=level,
                        data=patch_data,
                    ))
            elif isinstance(item, h5py.Dataset) and len(item.shape) == 3:
                # Direct dataset under level group
                patch_data = {("gas", key): item[()]}
                field_names.add(key)
                n = np.array(item.shape)
                # Infer edges from level and grid index
                le = np.zeros(3)
                re = np.ones(3)
                patches.append(GridPatch(
                    left_edge=le,
                    right_edge=re,
                    dimensions=n,
                    level=level,
                    data=patch_data,
                ))

    return patches, field_names


def _parse_grid_groups(grid_groups):
    """Parse AMR from grid groups with edge attributes."""
    patches = []
    field_names = set()

    for name, grp in grid_groups.items():
        le = np.array(grp.attrs.get("left_edge", grp.attrs.get("LeftEdge", [0, 0, 0])), dtype="float64")
        re = np.array(grp.attrs.get("right_edge", grp.attrs.get("RightEdge", [1, 1, 1])), dtype="float64")
        level = int(grp.attrs.get("level", grp.attrs.get("Level", 0)))

        patch_data = {}
        shape = None
        for ds_name in grp:
            ds = grp[ds_name]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) == 3:
                patch_data[("gas", ds_name)] = ds[()]
                field_names.add(ds_name)
                shape = ds.shape

        if shape is not None:
            patches.append(GridPatch(
                left_edge=le,
                right_edge=re,
                dimensions=np.array(shape),
                level=level,
                data=patch_data,
            ))

    return patches, field_names


def _parse_flat_amr(f, sig):
    """Parse AMR from flat datasets with different shapes.

    Groups datasets by shape, treating larger arrays as finer refinement.
    """
    patches = []
    field_names = set()

    # Group datasets by shape
    shape_groups = {}
    for ds_path, (shape, dtype) in sig.hdf5_datasets.items():
        if len(shape) == 3 and all(d > 1 for d in shape):
            shape_key = tuple(shape)
            if shape_key not in shape_groups:
                shape_groups[shape_key] = []
            shape_groups[shape_key].append(ds_path)

    if not shape_groups:
        return patches, field_names

    # Sort shapes by total cells (coarsest first)
    sorted_shapes = sorted(shape_groups.keys(), key=lambda s: np.prod(s))

    for level, shape in enumerate(sorted_shapes):
        ds_paths = shape_groups[shape]
        patch_data = {}
        for ds_path in ds_paths:
            field_name = ds_path.split("/")[-1]
            patch_data[("gas", field_name)] = f[ds_path][()]
            field_names.add(field_name)

        patches.append(GridPatch(
            left_edge=np.array([0.0, 0.0, 0.0]),
            right_edge=np.array([1.0, 1.0, 1.0]),
            dimensions=np.array(shape),
            level=level,
            data=patch_data,
        ))

    return patches, field_names


@register_adapter
class AMRHdf5Adapter(BaseAdapter):
    """Adapter for HDF5 files with AMR-like multi-resolution grid data."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.HDF5
            and sig.layout_type == LayoutType.AMR
            and sig.candidate_family in ("simple_hdf5", "unknown_hdf5")
        )

    @staticmethod
    def priority() -> int:
        return 60  # Higher than SimpleHDF5, lower than Gadget

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        ir = DatasetIR(
            data_kind=DataKind.AMR,
            source_path=path,
            dataset_name=Path(path).stem,
        )

        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
            velocity_unit=kwargs.get("velocity_unit"),
            magnetic_unit=kwargs.get("magnetic_unit"),
        )

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        else:
            ir.bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        with h5py.File(path, "r") as f:
            patches, field_names = _infer_grid_patches(f, sig)

        if not patches:
            raise ValueError(
                f"No AMR grid patches found in {path}. "
                "Could not determine grid hierarchy."
            )

        ir.grid_patches = patches

        # Infer domain dimensions from the coarsest level
        level_0 = [p for p in patches if p.level == 0]
        if level_0:
            ir.domain_dimensions = level_0[0].dimensions.copy()
        else:
            ir.domain_dimensions = patches[0].dimensions.copy()

        for name in field_names:
            ir.fields.append(FieldSpec(native_name=name, field_type="gas"))

        mylog.info(
            "AMRHdf5Adapter: built IR with %d patches, %d fields",
            len(patches),
            len(field_names),
        )
        return ir
