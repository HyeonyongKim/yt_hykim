"""
Build yt datasets from the canonical DatasetIR.

Converts IR into yt stream datasets using load_uniform_grid,
load_particles, or load_amr_grids.
"""

import logging
import numpy as np
import yt

from yt_universal.diagnostics import warn_no_units
from yt_universal.schema.ir import DataKind, DatasetIR

mylog = logging.getLogger(__name__)


def build_from_ir(ir: DatasetIR):
    """Convert a DatasetIR into a yt dataset.

    Parameters
    ----------
    ir : DatasetIR
        The canonical internal representation.

    Returns
    -------
    ds : yt Dataset
        A yt dataset (typically a stream dataset).
    """
    if ir.data_kind == DataKind.UNIGRID:
        return _build_uniform_grid(ir)
    elif ir.data_kind == DataKind.PARTICLE:
        return _build_particles(ir)
    elif ir.data_kind == DataKind.MIXED:
        # For mixed data, load the mesh part as a uniform grid
        # (particles can be added later or handled separately)
        if ir.mesh_data:
            return _build_uniform_grid(ir)
        elif ir.particle_data:
            return _build_particles(ir)
    elif ir.data_kind == DataKind.AMR:
        return _build_amr_grids(ir)

    raise ValueError(
        f"Cannot build dataset from IR with data_kind={ir.data_kind}. "
        "No suitable builder found."
    )


def _build_unit_kwargs(ir: DatasetIR) -> dict:
    """Extract unit keyword arguments from the IR's UnitSpec."""
    kwargs = {}
    if ir.units.length_unit is not None:
        kwargs["length_unit"] = ir.units.length_unit
    if ir.units.mass_unit is not None:
        kwargs["mass_unit"] = ir.units.mass_unit
    if ir.units.time_unit is not None:
        kwargs["time_unit"] = ir.units.time_unit
    if ir.units.velocity_unit is not None:
        kwargs["velocity_unit"] = ir.units.velocity_unit
    if ir.units.magnetic_unit is not None:
        kwargs["magnetic_unit"] = ir.units.magnetic_unit
    return kwargs


def _build_uniform_grid(ir: DatasetIR):
    """Build a uniform grid dataset from the IR."""
    if ir.domain_dimensions is None:
        raise ValueError("DatasetIR.domain_dimensions is required for uniform grid.")

    data = dict(ir.mesh_data)
    unit_kwargs = _build_unit_kwargs(ir)

    # Default bbox
    bbox = ir.bbox
    if bbox is None:
        bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # If no units specified, warn the user
    if not unit_kwargs:
        warn_no_units(ir.source_path)

    ds = yt.load_uniform_grid(
        data,
        ir.domain_dimensions,
        bbox=bbox,
        geometry=ir.geometry.value,
        periodicity=ir.periodicity,
        dataset_name=ir.dataset_name,
        **unit_kwargs,
    )

    mylog.info(
        "Built uniform grid dataset: %s, shape=%s, %d fields",
        ir.dataset_name,
        ir.domain_dimensions,
        len(data),
    )
    return ds


def _build_particles(ir: DatasetIR):
    """Build a particle dataset from the IR."""
    data = dict(ir.particle_data)
    unit_kwargs = _build_unit_kwargs(ir)

    bbox = ir.bbox
    if bbox is None:
        bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    if not unit_kwargs:
        warn_no_units(ir.source_path)

    ds = yt.load_particles(
        data,
        bbox=bbox,
        periodicity=ir.periodicity,
        dataset_name=ir.dataset_name,
        **unit_kwargs,
    )

    mylog.info(
        "Built particle dataset: %s, %d fields",
        ir.dataset_name,
        len(data),
    )
    return ds


def _build_amr_grids(ir: DatasetIR):
    """Build an AMR grid dataset from the IR."""
    if not ir.grid_patches:
        raise ValueError("DatasetIR.grid_patches is required for AMR datasets.")

    grid_data = []
    for patch in ir.grid_patches:
        entry = {
            "left_edge": patch.left_edge,
            "right_edge": patch.right_edge,
            "dimensions": patch.dimensions,
            "level": patch.level,
        }
        entry.update(patch.data)
        grid_data.append(entry)

    # Infer domain dimensions from level-0 patches
    if ir.domain_dimensions is not None:
        domain_dimensions = ir.domain_dimensions
    else:
        domain_dimensions = np.array([32, 32, 32])
        warnings.warn(
            "yt_universal: domain_dimensions not set for AMR data, defaulting to [32,32,32].",
            stacklevel=3,
        )

    unit_kwargs = _build_unit_kwargs(ir)
    bbox = ir.bbox
    if bbox is None:
        bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    ds = yt.load_amr_grids(
        grid_data,
        domain_dimensions,
        bbox=bbox,
        geometry=ir.geometry.value,
        **unit_kwargs,
    )

    mylog.info(
        "Built AMR dataset: %s, %d patches",
        ir.dataset_name,
        len(grid_data),
    )
    return ds
