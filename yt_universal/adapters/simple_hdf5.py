"""
Adapter for simple/generic HDF5 files containing regular 3D arrays.

Handles HDF5 files that are not recognized by any existing yt frontend
but contain uniform-grid mesh data (3D arrays of the same shape).
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature, LayoutType
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

# Common field name patterns that hint at physical meaning
_DENSITY_NAMES = {"density", "dens", "rho", "Density"}
_TEMPERATURE_NAMES = {"temperature", "temp", "Temperature"}
_PRESSURE_NAMES = {"pressure", "pres", "Pressure"}
_VELOCITY_PATTERNS = {"velocity", "vel", "Velocity"}


def _guess_universal_name(name):
    """Try to map a native field name to a universal alias."""
    lower = name.lower().replace(" ", "_")

    if lower in ("density", "dens", "rho"):
        return "density"
    if lower in ("temperature", "temp"):
        return "temperature"
    if lower in ("pressure", "pres"):
        return "pressure"
    for axis in ("x", "y", "z"):
        if lower in (f"velocity_{axis}", f"vel_{axis}", f"vel{axis}"):
            return f"velocity_{axis}"
        if lower in (f"magnetic_field_{axis}", f"mag_field_{axis}", f"b{axis}"):
            return f"magnetic_field_{axis}"

    return None


@register_adapter
class SimpleHDF5Adapter(BaseAdapter):
    """Adapter for generic HDF5 files with uniform 3D grid data."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.HDF5
            and sig.candidate_family == "simple_hdf5"
            and sig.layout_type in (LayoutType.UNIGRID, LayoutType.MIXED)
        )

    @staticmethod
    def priority() -> int:
        return 50

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        """Read the HDF5 file and construct a DatasetIR.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        sig : DatasetSignature
            Pre-computed signature.
        **kwargs
            Optional overrides: bbox, length_unit, mass_unit, etc.
        """
        ir = DatasetIR(
            data_kind=DataKind.UNIGRID,
            source_path=path,
            dataset_name=Path(path).stem,
        )

        # Apply user-provided unit overrides
        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
            velocity_unit=kwargs.get("velocity_unit"),
            magnetic_unit=kwargs.get("magnetic_unit"),
        )

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])

        with h5py.File(path, "r") as f:
            # Find all 3D datasets — these are mesh fields
            target_shape = None
            mesh_datasets = {}

            for ds_path, (shape, dtype) in sig.hdf5_datasets.items():
                if len(shape) == 3 and all(d > 1 for d in shape):
                    if target_shape is None:
                        target_shape = shape
                    # Only include arrays matching the dominant 3D shape
                    if shape == target_shape:
                        # Use the leaf name as the field name
                        field_name = ds_path.split("/")[-1]
                        mesh_datasets[ds_path] = field_name

            if not mesh_datasets:
                raise ValueError(
                    f"No 3D mesh arrays found in {path}. "
                    "SimpleHDF5Adapter requires at least one 3D array."
                )

            ir.domain_dimensions = np.array(target_shape)

            # Default bbox if not provided
            if ir.bbox is None:
                ir.bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

            # Read mesh data and build field specs
            for ds_path, field_name in mesh_datasets.items():
                data = f[ds_path][()]
                universal_name = _guess_universal_name(field_name)

                ir.fields.append(FieldSpec(
                    native_name=field_name,
                    universal_name=universal_name,
                    field_type="gas",
                ))
                ir.mesh_data[("gas", field_name)] = data

            # Also check for 1D particle-like arrays
            for ds_path, (shape, dtype) in sig.hdf5_datasets.items():
                if len(shape) == 1 and shape[0] > 1 and dtype.kind == "f":
                    field_name = ds_path.split("/")[-1]
                    ir.particle_data[("io", field_name)] = f[ds_path][()]
                    ir.fields.append(FieldSpec(
                        native_name=field_name,
                        field_type="io",
                    ))

            if ir.particle_data:
                ir.data_kind = DataKind.MIXED

        mylog.info(
            "SimpleHDF5Adapter: built IR with %d mesh fields, %d particle fields, "
            "grid shape %s",
            len(ir.mesh_data),
            len(ir.particle_data),
            target_shape,
        )
        return ir
