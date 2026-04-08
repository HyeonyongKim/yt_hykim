"""
Adapter for RAMSES directory-based datasets.

RAMSES stores output as a directory containing:
  - info_NNNNN.txt: Simulation metadata (units, cosmology, domain decomposition)
  - amr_NNNNN.outNNNNN: AMR tree structure (Fortran unformatted binary)
  - hydro_NNNNN.outNNNNN: Hydrodynamic field data (Fortran unformatted binary)
  - part_NNNNN.outNNNNN: Particle data (Fortran unformatted binary)

This adapter reads the info file for metadata and particle files for
particle data. Reading the full AMR grid + hydro data requires
reconstructing the octree, which is delegated to yt's native frontend.
"""

import logging
import os
import re
import struct
from pathlib import Path

import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)


def _read_fortran_record(f, endian="<"):
    """Read a single Fortran unformatted record.

    Fortran writes: [4-byte length][data][4-byte length]
    Returns the raw data bytes, or None if EOF.
    """
    rl_bytes = f.read(4)
    if len(rl_bytes) < 4:
        return None
    rl = struct.unpack(f"{endian}i", rl_bytes)[0]
    if rl < 0 or rl > 1_000_000_000:  # Sanity check
        return None
    data = f.read(rl)
    if len(data) < rl:
        return None
    rl2_bytes = f.read(4)
    if len(rl2_bytes) < 4:
        return None
    rl2 = struct.unpack(f"{endian}i", rl2_bytes)[0]
    if rl != rl2:
        mylog.debug("Fortran record mismatch: %d != %d", rl, rl2)
        return None
    return data


def _parse_ramses_info(info_path: str) -> dict:
    """Parse a RAMSES info file into a dict."""
    info = {}
    try:
        with open(info_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "=" in line and not line.startswith("DOMAIN"):
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    try:
                        # Try integer first, then float
                        if "." in val or "E" in val or "e" in val:
                            info[key] = float(val)
                        else:
                            info[key] = int(val)
                    except ValueError:
                        info[key] = val
                if line.startswith("ordering type"):
                    break  # Stop before domain decomposition table
    except (OSError, UnicodeDecodeError):
        pass
    return info


def _read_ramses_particles(part_path: str, endian="<") -> dict:
    """Read particles from a single RAMSES particle file.

    Returns dict of field_name -> numpy array, or empty dict on failure.
    """
    result = {}
    try:
        with open(part_path, "rb") as f:
            # Record 1: ncpu
            rec = _read_fortran_record(f, endian)
            if rec is None:
                return {}
            ncpu = struct.unpack(f"{endian}i", rec)[0]

            # Record 2: ndim
            rec = _read_fortran_record(f, endian)
            if rec is None:
                return {}
            ndim = struct.unpack(f"{endian}i", rec)[0]

            # Record 3: npart (local particle count)
            rec = _read_fortran_record(f, endian)
            if rec is None:
                return {}
            npart = struct.unpack(f"{endian}i", rec)[0]

            if npart <= 0:
                return {}

            # Records 4-8: metadata (localseed, nstar_tot, mstar_tot,
            # mstar_lost, nsink) — skip all of them.
            # We detect where positions start by looking for a record
            # of size npart * 8 bytes (npart doubles).
            expected_pos_size = npart * 8
            while True:
                rec = _read_fortran_record(f, endian)
                if rec is None:
                    return {}
                if len(rec) == expected_pos_size:
                    # This is the first position array (x)
                    # Put it back by storing it
                    first_pos = rec
                    break

            # Now read particle arrays
            # Positions: ndim records of npart doubles
            # (first_pos already contains x positions from the skip loop above)
            positions = [np.frombuffer(first_pos, dtype=f"{endian}f8").copy()]
            for _ in range(ndim - 1):
                rec = _read_fortran_record(f, endian)
                if rec is None:
                    return {}
                arr = np.frombuffer(rec, dtype=f"{endian}f8")
                if len(arr) != npart:
                    return {}
                positions.append(arr.copy())

            # Velocities: ndim records of npart doubles
            velocities = []
            for _ in range(ndim):
                rec = _read_fortran_record(f, endian)
                if rec is None:
                    return {}
                arr = np.frombuffer(rec, dtype=f"{endian}f8")
                if len(arr) != npart:
                    return {}
                velocities.append(arr.copy())

            # Mass: 1 record of npart doubles
            rec = _read_fortran_record(f, endian)
            if rec is None:
                return {}
            mass = np.frombuffer(rec, dtype=f"{endian}f8").copy()
            if len(mass) != npart:
                return {}

            # Identity: 1 record of npart integers
            rec = _read_fortran_record(f, endian)
            if rec is None:
                return {}
            # Could be int32 or int64 depending on RAMSES version
            if len(rec) == npart * 4:
                identity = np.frombuffer(rec, dtype=f"{endian}i4").copy()
            elif len(rec) == npart * 8:
                identity = np.frombuffer(rec, dtype=f"{endian}i8").copy()
            else:
                identity = None

            # Level: 1 record of npart integers
            rec = _read_fortran_record(f, endian)
            if rec is not None:
                if len(rec) == npart * 4:
                    level = np.frombuffer(rec, dtype=f"{endian}i4").copy()
                else:
                    level = None
            else:
                level = None

            # Store results
            field_names = ["particle_position_x", "particle_position_y", "particle_position_z"]
            for i in range(min(ndim, 3)):
                result[field_names[i]] = positions[i].astype(np.float64)

            vel_names = ["particle_velocity_x", "particle_velocity_y", "particle_velocity_z"]
            for i in range(min(ndim, 3)):
                result[vel_names[i]] = velocities[i].astype(np.float64)

            result["particle_mass"] = mass.astype(np.float64)

            if identity is not None:
                result["particle_index"] = identity.astype(np.float64)
            if level is not None:
                result["particle_level"] = level.astype(np.float64)

    except (OSError, struct.error) as e:
        mylog.debug("Error reading RAMSES particles from %s: %s", part_path, e)
        return {}

    return result


@register_adapter
class RamsesDirAdapter(BaseAdapter):
    """Adapter for RAMSES directory-based simulation outputs."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return sig.candidate_family == "ramses_dir"

    @staticmethod
    def priority() -> int:
        return 70

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        p = Path(path)
        if not p.is_dir():
            raise ValueError(f"Expected directory: {path}")

        # Find the info file
        info_files = sorted(p.glob("info_*.txt"))
        if not info_files:
            raise ValueError(f"No info_*.txt found in {path}")
        info_path = info_files[0]

        # Extract output number from info filename
        match = re.search(r"info_(\d+)\.txt", info_path.name)
        if not match:
            raise ValueError(f"Cannot parse output number from {info_path.name}")
        output_num = match.group(1)

        # Parse info file
        info = _parse_ramses_info(str(info_path))
        ncpu = info.get("ncpu", 1)
        boxlen = info.get("boxlen", 1.0)
        unit_l = info.get("unit_l", 1.0)
        unit_d = info.get("unit_d", 1.0)
        unit_t = info.get("unit_t", 1.0)

        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,
            source_path=path,
            dataset_name=p.name,
        )

        # Convert RAMSES units to yt units
        # unit_l is in cm, unit_d in g/cm^3, unit_t in seconds
        if kwargs.get("length_unit"):
            ir.units.length_unit = kwargs["length_unit"]
        else:
            ir.units.length_unit = (unit_l * boxlen, "cm")

        if kwargs.get("mass_unit"):
            ir.units.mass_unit = kwargs["mass_unit"]
        else:
            # mass_unit = unit_d * unit_l^3
            ir.units.mass_unit = (unit_d * unit_l**3, "g")

        if kwargs.get("time_unit"):
            ir.units.time_unit = kwargs["time_unit"]
        else:
            ir.units.time_unit = (unit_t, "s")

        if kwargs.get("velocity_unit"):
            ir.units.velocity_unit = kwargs["velocity_unit"]
        else:
            ir.units.velocity_unit = (unit_l / unit_t, "cm/s")

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        else:
            # RAMSES uses code units [0, 1] for positions
            ir.bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        # Find and read particle files
        part_files = sorted(p.glob(f"part_{output_num}.out*"))
        if not part_files:
            raise ValueError(
                f"No particle files (part_{output_num}.out*) found in {path}"
            )

        mylog.info(
            "RamsesDirAdapter: reading particles from %d files in %s",
            len(part_files), path,
        )

        # Read particles from all CPU files
        all_particles = {}
        n_total = 0

        for pf in part_files:
            cpu_particles = _read_ramses_particles(str(pf))
            if not cpu_particles:
                continue

            n_local = len(next(iter(cpu_particles.values())))
            n_total += n_local

            for field_name, arr in cpu_particles.items():
                if field_name not in all_particles:
                    all_particles[field_name] = []
                all_particles[field_name].append(arr)

        if n_total == 0:
            raise ValueError(f"No particles could be read from {path}")

        # Concatenate arrays
        for field_name in all_particles:
            combined = np.concatenate(all_particles[field_name])
            key = ("io", field_name)
            ir.particle_data[key] = combined
            ir.fields.append(FieldSpec(
                native_name=field_name,
                universal_name=field_name,
                field_type="io",
            ))

        mylog.info(
            "RamsesDirAdapter: built IR with %d particles, %d fields from %s",
            n_total, len(ir.particle_data), path,
        )
        return ir
