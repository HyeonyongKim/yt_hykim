"""
Adapter for Enzo directory-based datasets.

Enzo stores simulation output as:
  - A parameter file (e.g., FD0048) with simulation metadata
  - A hierarchy file (e.g., FD0048.hierarchy) listing all grids
  - Multiple HDF5 CPU files (e.g., FD0048.cpu0000..0143) with grid/particle data

This adapter:
  1. Parses the parameter file for domain info
  2. Parses the hierarchy file for grid metadata and particle counts
  3. Reads particle data from all CPU files
  4. Returns a particle dataset
"""

import logging
import os
import re
from pathlib import Path

import h5py
import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

# Enzo particle field name -> universal name
ENZO_PARTICLE_MAP = {
    "particle_position_x": "particle_position_x",
    "particle_position_y": "particle_position_y",
    "particle_position_z": "particle_position_z",
    "particle_velocity_x": "particle_velocity_x",
    "particle_velocity_y": "particle_velocity_y",
    "particle_velocity_z": "particle_velocity_z",
    "particle_mass": "particle_mass",
    "particle_index": "particle_index",
    "particle_type": "particle_type",
}


def _parse_enzo_params(path: str) -> dict:
    """Parse an Enzo parameter file into a dict of key-value pairs."""
    params = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    params[key.strip()] = val.strip()
    except (OSError, UnicodeDecodeError):
        pass
    return params


def _parse_hierarchy(hierarchy_path: str) -> list[dict]:
    """Parse an Enzo hierarchy file to extract grid metadata.

    Returns a list of dicts, one per grid, with keys:
      grid_id, dimensions, left_edge, right_edge, n_particles, cpu_file
    """
    grids = []
    current = {}

    try:
        with open(hierarchy_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Grid = "):
                    if current:
                        grids.append(current)
                    current = {"grid_id": int(line.split("=")[1].strip())}
                elif line.startswith("GridDimension"):
                    dims = [int(x) for x in line.split("=")[1].strip().split()]
                    current["dimensions"] = dims
                elif line.startswith("GridLeftEdge"):
                    edges = [float(x) for x in line.split("=")[1].strip().split()]
                    current["left_edge"] = edges
                elif line.startswith("GridRightEdge"):
                    edges = [float(x) for x in line.split("=")[1].strip().split()]
                    current["right_edge"] = edges
                elif line.startswith("NumberOfParticles"):
                    current["n_particles"] = int(line.split("=")[1].strip())
                elif line.startswith("ParticleFileName"):
                    current["cpu_file"] = line.split("=")[1].strip()
                elif line.startswith("BaryonFileName"):
                    current["baryon_file"] = line.split("=")[1].strip()

        if current:
            grids.append(current)
    except (OSError, UnicodeDecodeError):
        pass

    return grids


@register_adapter
class EnzoDirAdapter(BaseAdapter):
    """Adapter for Enzo directory-based simulation outputs."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return sig.candidate_family in ("enzo_file", "enzo_dir")

    @staticmethod
    def priority() -> int:
        return 75

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        p = Path(path)

        # Determine base name and directory
        if p.is_dir():
            # Path is the directory — find the parameter file
            base_dir = p
            param_file = self._find_param_file(base_dir)
            if param_file is None:
                raise ValueError(f"No Enzo parameter file found in {path}")
        else:
            # Path is the parameter file itself
            base_dir = p.parent
            param_file = p

        base_name = param_file.stem
        hierarchy_path = base_dir / f"{base_name}.hierarchy"

        # Parse parameter file for domain info
        params = _parse_enzo_params(str(param_file))
        domain_left = np.array(
            [float(x) for x in params.get("DomainLeftEdge", "0 0 0").split()]
        )
        domain_right = np.array(
            [float(x) for x in params.get("DomainRightEdge", "1 1 1").split()]
        )

        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,
            source_path=path,
            dataset_name=base_name,
        )

        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
            velocity_unit=kwargs.get("velocity_unit"),
        )

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        else:
            ir.bbox = np.array([
                [domain_left[0], domain_right[0]],
                [domain_left[1], domain_right[1]],
                [domain_left[2], domain_right[2]],
            ])

        # Parse hierarchy file
        if not hierarchy_path.exists():
            raise ValueError(f"Hierarchy file not found: {hierarchy_path}")

        grids = _parse_hierarchy(str(hierarchy_path))
        mylog.info("EnzoDirAdapter: parsed %d grids from hierarchy", len(grids))

        # Collect all particle data from CPU files
        # Group grids by CPU file to minimize file opens
        file_grids = {}
        total_particles = 0
        for g in grids:
            n_part = g.get("n_particles", 0)
            if n_part > 0:
                cpu_file = g.get("cpu_file", "")
                # Resolve relative path
                if cpu_file.startswith("./"):
                    cpu_file = cpu_file[2:]
                # The cpu_file path is relative to the parent of the output directory
                # e.g., "./RD0048/FD0048.cpu0000" relative to the enzo run dir
                # We need to find it relative to base_dir
                full_cpu = base_dir / os.path.basename(cpu_file)
                if not full_cpu.exists():
                    # Try resolving from parent
                    full_cpu = base_dir.parent / cpu_file
                if full_cpu.exists():
                    key = str(full_cpu)
                    if key not in file_grids:
                        file_grids[key] = []
                    file_grids[key].append(g)
                    total_particles += n_part

        if total_particles == 0:
            raise ValueError(f"No particles found in {path}")

        mylog.info(
            "EnzoDirAdapter: reading %d particles from %d CPU files",
            total_particles, len(file_grids),
        )

        # Pre-allocate arrays
        particle_arrays = {}
        offset = 0

        for cpu_path, cpu_grids in file_grids.items():
            try:
                with h5py.File(cpu_path, "r") as hf:
                    for g in cpu_grids:
                        grid_name = f"Grid{g['grid_id']:08d}"
                        if grid_name not in hf:
                            continue

                        grp = hf[grid_name]
                        n_part = g.get("n_particles", 0)

                        for field_name in grp:
                            ds = grp[field_name]
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            if len(ds.shape) != 1:
                                continue  # Skip 3D grid data
                            if ds.shape[0] != n_part:
                                continue

                            universal = ENZO_PARTICLE_MAP.get(field_name, field_name)
                            key = ("io", universal)

                            if key not in particle_arrays:
                                particle_arrays[key] = np.empty(
                                    total_particles, dtype=np.float64
                                )

                            data = ds[()].astype(np.float64)
                            particle_arrays[key][offset:offset + len(data)] = data

                        offset += n_part
            except (OSError, KeyError) as e:
                mylog.warning("EnzoDirAdapter: error reading %s: %s", cpu_path, e)
                continue

        # Trim arrays to actual size (some grids may have been skipped)
        for key in particle_arrays:
            if offset < total_particles:
                particle_arrays[key] = particle_arrays[key][:offset]

        ir.particle_data = particle_arrays
        for key in particle_arrays:
            ptype, universal = key
            ir.fields.append(FieldSpec(
                native_name=universal,
                universal_name=universal,
                field_type=ptype,
            ))

        mylog.info(
            "EnzoDirAdapter: built IR with %d particles, %d fields from %s",
            offset, len(particle_arrays), path,
        )
        return ir

    @staticmethod
    def _find_param_file(directory: Path):
        """Find the Enzo parameter file in a directory."""
        # Look for a file that has a matching .hierarchy sibling
        for f in directory.iterdir():
            if f.suffix == ".hierarchy":
                param = directory / f.stem
                if param.exists() and param.is_file():
                    return param
        return None
