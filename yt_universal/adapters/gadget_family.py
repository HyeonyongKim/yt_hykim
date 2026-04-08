"""
Adapter for Gadget-family HDF5 files (Gadget, Gizmo, Arepo, SWIFT-like).

When yt's native frontend fails (e.g. missing dependencies, version mismatch),
this adapter reads the HDF5 directly and builds particle datasets via
yt's stream loaders.

When yt's native frontend succeeds (the common case), this adapter is not
used — load_universal() delegates to yt.load() first.
"""

import logging

import h5py
import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

# Standard Gadget HDF5 particle type mapping
GADGET_PTYPE_MAP = {
    "PartType0": "Gas",
    "PartType1": "Halo",
    "PartType2": "Disk",
    "PartType3": "Bulge",
    "PartType4": "Stars",
    "PartType5": "Boundary",
}

# Gadget vector fields that need to be split into components
VECTOR_FIELDS = {
    "Coordinates": ("particle_position_x", "particle_position_y", "particle_position_z"),
    "Velocities": ("particle_velocity_x", "particle_velocity_y", "particle_velocity_z"),
    "Velocity": ("particle_velocity_x", "particle_velocity_y", "particle_velocity_z"),
    "MagneticField": ("magnetic_field_x", "magnetic_field_y", "magnetic_field_z"),
}

# Scalar field name mapping: gadget native -> universal
SCALAR_FIELD_MAP = {
    "Masses": "particle_mass",
    "Mass": "particle_mass",
    "ParticleIDs": "particle_index",
    "InternalEnergy": "specific_thermal_energy",
    "Density": "density",
    "SmoothingLength": "smoothing_length",
    "ElectronAbundance": "electron_abundance",
    "NeutralHydrogenAbundance": "neutral_hydrogen_abundance",
    "StarFormationRate": "star_formation_rate",
    "Metallicity": "metallicity",
    "GFM_Metallicity": "metallicity",
    "Temperature": "temperature",
    "Potential": "potential",
}


def _parse_gadget_header(f):
    """Extract header info from a Gadget-like HDF5 file.

    Returns a dict with keys: time, redshift, boxsize, num_particles,
    omega_matter, omega_lambda, hubble_param, num_files.
    """
    header = {}
    if "Header" not in f:
        return header

    h = f["Header"].attrs

    header["time"] = float(h.get("Time", h.get("Time_GYR", 0.0)))
    header["redshift"] = float(h.get("Redshift", 0.0))
    header["boxsize"] = float(h.get("BoxSize", 1.0))
    header["num_particles"] = np.array(h.get("NumPart_ThisFile", h.get("NumPart_Total", [])))
    header["omega_matter"] = float(h.get("Omega0", 0.0))
    header["omega_lambda"] = float(h.get("OmegaLambda", 0.0))
    header["hubble_param"] = float(h.get("HubbleParam", 1.0))
    header["num_files"] = int(h.get("NumFilesPerSnapshot", 1))

    # Try to get unit info
    if "UnitLength_in_cm" in h:
        header["unit_length_cm"] = float(h["UnitLength_in_cm"])
    if "UnitMass_in_g" in h:
        header["unit_mass_g"] = float(h["UnitMass_in_g"])
    if "UnitVelocity_in_cm_per_s" in h:
        header["unit_velocity_cms"] = float(h["UnitVelocity_in_cm_per_s"])

    return header


@register_adapter
class GadgetFamilyAdapter(BaseAdapter):
    """Adapter for Gadget/Gizmo/Arepo/SWIFT-like HDF5 particle data."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.HDF5
            and sig.candidate_family in ("gadget_like", "arepo_like")
        )

    @staticmethod
    def priority() -> int:
        return 70  # Higher than SimpleHDF5Adapter

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,
            source_path=path,
            dataset_name=f"GadgetFamily_{sig.candidate_family}",
        )

        with h5py.File(path, "r") as f:
            header = _parse_gadget_header(f)

            # Set up units from header or kwargs
            ir.units = UnitSpec(
                length_unit=kwargs.get("length_unit"),
                mass_unit=kwargs.get("mass_unit"),
                velocity_unit=kwargs.get("velocity_unit"),
            )

            # If header has unit info and user didn't override, use header units
            # Pass as (value, "unit") tuples for yt/unyt compatibility
            if ir.units.length_unit is None and "unit_length_cm" in header:
                ir.units.length_unit = (header["unit_length_cm"], "cm")
            if ir.units.mass_unit is None and "unit_mass_g" in header:
                ir.units.mass_unit = (header["unit_mass_g"], "g")
            if ir.units.velocity_unit is None and "unit_velocity_cms" in header:
                ir.units.velocity_unit = (header["unit_velocity_cms"], "cm/s")

            # Set bbox from BoxSize
            boxsize = header.get("boxsize", 1.0)
            if "bbox" in kwargs:
                ir.bbox = np.array(kwargs["bbox"])
            else:
                ir.bbox = np.array([[0, boxsize], [0, boxsize], [0, boxsize]])

            # Read particle data keeping each PartType separate.
            # yt's load_particles needs each ptype to have its own position
            # fields, so we store as (PartType0, field), (PartType1, field)...
            # yt will automatically create an "all" union across types.
            ptype_groups = []
            for group_name in sorted(f.keys()):
                if not group_name.startswith("PartType"):
                    continue
                grp = f[group_name]
                if not isinstance(grp, h5py.Group):
                    continue
                # Skip empty particle groups
                if len(grp) == 0:
                    continue
                # Check if any dataset has non-zero size
                has_data = any(
                    isinstance(grp[k], h5py.Dataset) and grp[k].shape[0] > 0
                    for k in grp
                )
                if not has_data:
                    continue
                ptype_groups.append(group_name)

            # If only one PartType, use "io" as the type name (simpler)
            use_io = len(ptype_groups) == 1

            for group_name in ptype_groups:
                grp = f[group_name]
                ftype = "io" if use_io else group_name

                for field_name in grp:
                    ds = grp[field_name]
                    if not isinstance(ds, h5py.Dataset):
                        continue

                    data = ds[()]

                    if field_name in VECTOR_FIELDS and len(data.shape) == 2 and data.shape[1] >= 3:
                        component_names = VECTOR_FIELDS[field_name]
                        for i, comp_name in enumerate(component_names):
                            ir.particle_data[(ftype, comp_name)] = data[:, i]
                    elif len(data.shape) == 1:
                        universal = SCALAR_FIELD_MAP.get(field_name)
                        out_name = universal if universal else field_name
                        ir.particle_data[(ftype, out_name)] = data

                        ir.fields.append(FieldSpec(
                            native_name=f"{group_name}/{field_name}",
                            universal_name=universal,
                            field_type=ftype,
                        ))

        mylog.info(
            "GadgetFamilyAdapter: built IR with %d particle fields from %s",
            len(ir.particle_data),
            path,
        )
        return ir
