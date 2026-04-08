"""
Adapter for GAMER HDF5 datasets.

GAMER stores AMR patch-based data in a specific layout:
  - GridData/: 4D arrays (n_patches, 8, 8, 8) for each fluid field
  - Particle/: 1D arrays for particle attributes (ParPosX, ParMass, etc.)
  - Tree/: AMR octree structure (Corner, Father, Son, Sibling)
  - Info/: Simulation metadata (KeyInfo, InputPara)

This adapter reads grid and particle data and builds a yt dataset
via the stream loaders.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, GridPatch, UnitSpec

mylog = logging.getLogger(__name__)

# GAMER field name -> universal name
GAMER_FIELD_MAP = {
    "Dens": "density",
    "MomX": "momentum_x",
    "MomY": "momentum_y",
    "MomZ": "momentum_z",
    "Engy": "total_energy",
    "Dual": "specific_thermal_energy",
    "Pote": "gravitational_potential",
    "Metal": "metallicity",
    "ParDens": "particle_density",
}

GAMER_PARTICLE_MAP = {
    "ParPosX": "particle_position_x",
    "ParPosY": "particle_position_y",
    "ParPosZ": "particle_position_z",
    "ParVelX": "particle_velocity_x",
    "ParVelY": "particle_velocity_y",
    "ParVelZ": "particle_velocity_z",
    "ParMass": "particle_mass",
    "ParPUid": "particle_index",
    "ParType": "particle_type",
    "ParCreTime": "particle_creation_time",
    "ParMetalFrac": "particle_metallicity",
}


@register_adapter
class GAMERHdf5Adapter(BaseAdapter):
    """Adapter for GAMER HDF5 simulation outputs."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.HDF5
            and sig.candidate_family == "gamer_like"
        )

    @staticmethod
    def priority() -> int:
        return 75  # High priority for recognized GAMER files

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,  # Use particle loader for simplicity
            source_path=path,
            dataset_name=Path(path).stem,
        )

        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
            velocity_unit=kwargs.get("velocity_unit"),
        )

        with h5py.File(path, "r") as f:
            # Read metadata from Info/KeyInfo if available
            boxsize = 1.0
            if "Info" in f and "KeyInfo" in f["Info"]:
                key_info = f["Info"]["KeyInfo"]
                if key_info.dtype.names:
                    ki = key_info[()]
                    if "BoxSize" in key_info.dtype.names:
                        bs = ki["BoxSize"]
                        # BoxSize can be scalar or array (3-element for 3D)
                        if hasattr(bs, '__len__') and len(bs) > 0:
                            boxsize = float(bs[0])
                        else:
                            boxsize = float(bs)

            if "bbox" in kwargs:
                ir.bbox = np.array(kwargs["bbox"])
            else:
                ir.bbox = np.array([[0, boxsize], [0, boxsize], [0, boxsize]])

            # Read particle data from Particle/ group
            if "Particle" in f:
                pgrp = f["Particle"]
                for field_name in pgrp:
                    ds = pgrp[field_name]
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    if len(ds.shape) != 1:
                        continue

                    universal = GAMER_PARTICLE_MAP.get(field_name, field_name)
                    ir.particle_data[("io", universal)] = ds[()].astype(np.float64)
                    ir.fields.append(FieldSpec(
                        native_name=field_name,
                        universal_name=universal,
                        field_type="io",
                    ))

            # Note: GridData contains 4D patch arrays (n_patches, 8, 8, 8).
            # Flattening all patches into particle-like data would use too much
            # memory (e.g. 509K patches × 512 cells × 9 fields ≈ 18 GB).
            # For now, we only load actual particles from the Particle/ group.
            # Full AMR grid support would require building an octree, which is
            # better handled by yt's native GAMER frontend.

        mylog.info(
            "GAMERHdf5Adapter: built IR with %d fields from %s",
            len(ir.particle_data),
            path,
        )
        return ir
