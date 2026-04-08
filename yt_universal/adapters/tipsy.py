"""
Adapter for Tipsy binary datasets (ChaNGa, Gasoline, PKDGRAV).

Tipsy format (big-endian by default):
  Header (32 bytes):
    double time, int ntotal, int ndim, int ngas, int ndark, int nstar, int pad

  Gas particles (ngas x 48 bytes = 12 floats each):
    mass, x, y, z, vx, vy, vz, rho, temp, hsmooth, metals, phi

  Dark matter particles (ndark x 36 bytes = 9 floats each):
    mass, x, y, z, vx, vy, vz, eps, phi

  Star particles (nstar x 44 bytes = 11 floats each):
    mass, x, y, z, vx, vy, vz, metals, tform, eps, phi
"""

import logging
import struct
from pathlib import Path

import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

# Tipsy gas particle fields (12 floats, 48 bytes)
GAS_FIELDS = [
    ("particle_mass", "particle_mass"),
    ("particle_position_x", "particle_position_x"),
    ("particle_position_y", "particle_position_y"),
    ("particle_position_z", "particle_position_z"),
    ("particle_velocity_x", "particle_velocity_x"),
    ("particle_velocity_y", "particle_velocity_y"),
    ("particle_velocity_z", "particle_velocity_z"),
    ("density", "density"),
    ("temperature", "temperature"),
    ("smoothing_length", "smoothing_length"),
    ("metallicity", "metallicity"),
    ("gravitational_potential", "gravitational_potential"),
]

# Dark matter particle fields (9 floats, 36 bytes)
DARK_FIELDS = [
    ("particle_mass", "particle_mass"),
    ("particle_position_x", "particle_position_x"),
    ("particle_position_y", "particle_position_y"),
    ("particle_position_z", "particle_position_z"),
    ("particle_velocity_x", "particle_velocity_x"),
    ("particle_velocity_y", "particle_velocity_y"),
    ("particle_velocity_z", "particle_velocity_z"),
    ("softening", "softening"),
    ("gravitational_potential", "gravitational_potential"),
]

# Star particle fields (11 floats, 44 bytes)
STAR_FIELDS = [
    ("particle_mass", "particle_mass"),
    ("particle_position_x", "particle_position_x"),
    ("particle_position_y", "particle_position_y"),
    ("particle_position_z", "particle_position_z"),
    ("particle_velocity_x", "particle_velocity_x"),
    ("particle_velocity_y", "particle_velocity_y"),
    ("particle_velocity_z", "particle_velocity_z"),
    ("metallicity", "metallicity"),
    ("formation_time", "formation_time"),
    ("softening", "softening"),
    ("gravitational_potential", "gravitational_potential"),
]


@register_adapter
class TipsyAdapter(BaseAdapter):
    """Adapter for Tipsy binary simulation outputs."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.BINARY
            and sig.candidate_family == "tipsy"
        )

    @staticmethod
    def priority() -> int:
        return 80

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,
            source_path=path,
            dataset_name=Path(path).stem,
        )

        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
            velocity_unit=kwargs.get("velocity_unit"),
        )

        # Get header info from signature
        endian_str = sig.hdf5_datasets.get("tipsy_endian", ("big", None))[0]
        endian = ">" if endian_str == "big" else "<"
        ngas = sig.hdf5_datasets.get("tipsy_ngas", (0, None))[0]
        ndark = sig.hdf5_datasets.get("tipsy_ndark", (0, None))[0]
        nstar = sig.hdf5_datasets.get("tipsy_nstar", (0, None))[0]

        with open(path, "rb") as f:
            # Skip header
            f.seek(32)

            # Read gas particles
            if ngas > 0:
                gas_data = np.frombuffer(
                    f.read(ngas * 48), dtype=np.dtype(f"{endian}f4")
                ).reshape(ngas, 12)
                self._store_particles(ir, "Gas", gas_data, GAS_FIELDS)

            # Read dark matter particles
            if ndark > 0:
                dark_data = np.frombuffer(
                    f.read(ndark * 36), dtype=np.dtype(f"{endian}f4")
                ).reshape(ndark, 9)
                self._store_particles(ir, "DarkMatter", dark_data, DARK_FIELDS)

            # Read star particles
            if nstar > 0:
                star_data = np.frombuffer(
                    f.read(nstar * 44), dtype=np.dtype(f"{endian}f4")
                ).reshape(nstar, 11)
                self._store_particles(ir, "Stars", star_data, STAR_FIELDS)

        # Compute bounding box from all particle positions
        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        else:
            all_x, all_y, all_z = [], [], []
            for ptype in ("Gas", "DarkMatter", "Stars"):
                xkey = (ptype, "particle_position_x")
                ykey = (ptype, "particle_position_y")
                zkey = (ptype, "particle_position_z")
                if xkey in ir.particle_data:
                    all_x.append(ir.particle_data[xkey])
                    all_y.append(ir.particle_data[ykey])
                    all_z.append(ir.particle_data[zkey])

            if all_x:
                x = np.concatenate(all_x)
                y = np.concatenate(all_y)
                z = np.concatenate(all_z)
                margin = 0.01 * max(
                    x.max() - x.min(),
                    y.max() - y.min(),
                    z.max() - z.min(),
                )
                ir.bbox = np.array([
                    [x.min() - margin, x.max() + margin],
                    [y.min() - margin, y.max() + margin],
                    [z.min() - margin, z.max() + margin],
                ])
            else:
                ir.bbox = np.array([[0, 1], [0, 1], [0, 1]])

        mylog.info(
            "TipsyAdapter: ngas=%d, ndark=%d, nstar=%d from %s",
            ngas, ndark, nstar, path,
        )
        return ir

    @staticmethod
    def _store_particles(ir, ptype, data, field_defs):
        """Store particle arrays into the IR."""
        for col_idx, (native_name, universal_name) in enumerate(field_defs):
            key = (ptype, universal_name)
            ir.particle_data[key] = data[:, col_idx].astype(np.float64)
            ir.fields.append(FieldSpec(
                native_name=native_name,
                universal_name=universal_name,
                field_type=ptype,
            ))
