"""
Adapter for ASCII columnar data files (particle tables and grid slices).

Handles space/comma/tab-delimited text files with optional headers.
Parses into numpy arrays and builds yt stream datasets.
"""

import logging
import re
from pathlib import Path

import numpy as np

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import ContainerType, DatasetSignature, LayoutType
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

# Common column name patterns → universal field names
COLUMN_NAME_MAP = {
    "x": "particle_position_x",
    "y": "particle_position_y",
    "z": "particle_position_z",
    "vx": "particle_velocity_x",
    "vy": "particle_velocity_y",
    "vz": "particle_velocity_z",
    "pos_x": "particle_position_x",
    "pos_y": "particle_position_y",
    "pos_z": "particle_position_z",
    "vel_x": "particle_velocity_x",
    "vel_y": "particle_velocity_y",
    "vel_z": "particle_velocity_z",
    "mass": "particle_mass",
    "m": "particle_mass",
    "id": "particle_index",
    "pid": "particle_index",
    "particle_id": "particle_index",
    "density": "density",
    "rho": "density",
    "temperature": "temperature",
    "temp": "temperature",
    "pressure": "pressure",
}


def _detect_delimiter(sample_line):
    """Detect the delimiter used in a data line."""
    if "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\t"
    if "|" in sample_line:
        return "|"
    return None  # whitespace


def _read_ascii_data(path):
    """Read an ASCII data file, returning column names and numpy arrays.

    Returns
    -------
    columns : list[str]
        Column names (from header or auto-generated).
    data : np.ndarray
        2D array of shape (n_rows, n_cols).
    """
    header_names = None
    data_lines = []
    delimiter = None

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Comment/header line
            if stripped.startswith("#") or stripped.startswith("!") or stripped.startswith("%"):
                content = stripped.lstrip("#!% ")
                # Check if it's a header with column names
                parts = re.split(r"[,\t|]+", content) if any(d in content for d in [",", "\t", "|"]) else content.split()
                if parts and not _is_numeric_line(parts):
                    header_names = [p.strip() for p in parts]
                continue

            # First data line — detect delimiter and check for text header
            if not data_lines:
                delimiter = _detect_delimiter(stripped)
                parts = stripped.split(delimiter) if delimiter else stripped.split()
                parts = [p.strip() for p in parts if p.strip()]
                if not _is_numeric_line(parts):
                    # This is a header row without comment prefix
                    header_names = parts
                    continue

            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in {path}")

    # Parse data
    rows = []
    for line in data_lines:
        parts = line.split(delimiter) if delimiter else line.split()
        parts = [p.strip() for p in parts if p.strip()]
        try:
            row = [float(p) for p in parts]
            rows.append(row)
        except ValueError:
            continue  # Skip unparseable lines

    data = np.array(rows)

    # Generate column names if none found
    n_cols = data.shape[1] if len(data.shape) == 2 else 1
    if header_names is None:
        header_names = [f"col{i}" for i in range(n_cols)]
    elif len(header_names) != n_cols:
        header_names = [f"col{i}" for i in range(n_cols)]

    return header_names, data


def _is_numeric_line(parts):
    """Check if all parts are numeric."""
    for p in parts:
        try:
            float(p)
        except ValueError:
            return False
    return True


@register_adapter
class ASCIIParticleAdapter(BaseAdapter):
    """Adapter for ASCII particle tables."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.ASCII
            and sig.candidate_family == "ascii_particle"
        )

    @staticmethod
    def priority() -> int:
        return 40

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        columns, data = _read_ascii_data(path)

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

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])

        # Map columns to particle fields
        has_position = False
        for i, col_name in enumerate(columns):
            col_lower = col_name.lower().strip()
            universal = COLUMN_NAME_MAP.get(col_lower)

            if universal is None:
                # Try partial matching
                universal = col_lower

            field_key = ("io", universal if universal else col_lower)
            ir.particle_data[field_key] = data[:, i]

            if "position" in (universal or ""):
                has_position = True

            ir.fields.append(FieldSpec(
                native_name=col_name,
                universal_name=universal,
                field_type="io",
            ))

        # If no bbox provided and we have position data, infer from data
        if ir.bbox is None and has_position:
            pos_fields = ["particle_position_x", "particle_position_y", "particle_position_z"]
            pos_data = []
            for pf in pos_fields:
                key = ("io", pf)
                if key in ir.particle_data:
                    pos_data.append(ir.particle_data[key])
            if len(pos_data) == 3:
                mins = [d.min() for d in pos_data]
                maxs = [d.max() for d in pos_data]
                margin = 0.05
                ir.bbox = np.array([
                    [m - margin * abs(m), M + margin * abs(M)]
                    for m, M in zip(mins, maxs)
                ])

        mylog.info(
            "ASCIIParticleAdapter: %d particles, %d columns from %s",
            data.shape[0],
            data.shape[1],
            path,
        )
        return ir


@register_adapter
class ASCIIGridAdapter(BaseAdapter):
    """Adapter for ASCII grid/mesh data (structured tables)."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return (
            sig.container_type == ContainerType.ASCII
            and sig.candidate_family == "ascii_table"
        )

    @staticmethod
    def priority() -> int:
        return 35

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        columns, data = _read_ascii_data(path)

        # For grid data, we need to know the dimensions
        # Try to infer from data shape or require user to specify
        grid_dims = kwargs.get("grid_dims")
        if grid_dims is not None:
            grid_dims = np.array(grid_dims)
        else:
            # Try to find a cube that fits the data
            n_rows = data.shape[0]
            cube_root = int(round(n_rows ** (1.0 / 3.0)))
            if cube_root ** 3 == n_rows:
                grid_dims = np.array([cube_root, cube_root, cube_root])
            else:
                raise ValueError(
                    f"Cannot infer grid dimensions from {n_rows} rows. "
                    "Pass grid_dims=(nx, ny, nz) to load_universal()."
                )

        ir = DatasetIR(
            data_kind=DataKind.UNIGRID,
            source_path=path,
            dataset_name=Path(path).stem,
            domain_dimensions=grid_dims,
        )

        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit"),
            mass_unit=kwargs.get("mass_unit"),
            time_unit=kwargs.get("time_unit"),
        )

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        else:
            ir.bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        # Reshape columns into 3D arrays
        for i, col_name in enumerate(columns):
            col_lower = col_name.lower().strip()
            reshaped = data[:, i].reshape(grid_dims)
            ir.mesh_data[("gas", col_lower)] = reshaped
            ir.fields.append(FieldSpec(
                native_name=col_name,
                field_type="gas",
            ))

        mylog.info(
            "ASCIIGridAdapter: grid %s, %d fields from %s",
            grid_dims,
            len(columns),
            path,
        )
        return ir
