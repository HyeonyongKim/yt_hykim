"""
Manifest-assisted adapter for exotic binary (or other) formats.

Requires a sidecar YAML config file (yt_universal.yaml) next to the
data file that describes the format, fields, units, and layout.

Example manifest (yt_universal.yaml):

    format: binary_particle
    endianness: little
    dtype:
      x: float32
      y: float32
      z: float32
      mass: float32
    units:
      length_unit: kpc
      mass_unit: Msun
    field_map:
      x: particle_position_x
      y: particle_position_y
      z: particle_position_z
      mass: particle_mass
    bbox:
      - [0, 1000]
      - [0, 1000]
      - [0, 1000]

Supported format values:
  - binary_particle: flat binary with consecutive fields
  - binary_grid: flat binary with 3D grid data
"""

import logging
from pathlib import Path

import numpy as np

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from yt_universal.adapters.base import BaseAdapter
from yt_universal.adapters.registry import register_adapter
from yt_universal.inspectors.base import DatasetSignature
from yt_universal.schema.ir import DataKind, DatasetIR, FieldSpec, UnitSpec

mylog = logging.getLogger(__name__)

MANIFEST_FILENAME = "yt_universal.yaml"

# numpy dtype mapping
DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def find_manifest(data_path: str) -> str | None:
    """Look for a yt_universal.yaml manifest next to the data file."""
    p = Path(data_path)
    candidates = [
        p.parent / MANIFEST_FILENAME,
        p.parent / f"{p.stem}.yaml",
        p.parent / f"{p.stem}.yml",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _parse_manifest(manifest_path: str) -> dict:
    """Parse a YAML manifest file."""
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required for manifest-assisted loading. "
            "Install it with: pip install pyyaml"
        )
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f)


@register_adapter
class ManifestAdapter(BaseAdapter):
    """Adapter that uses a sidecar YAML manifest to load exotic formats."""

    @staticmethod
    def can_handle(sig: DatasetSignature) -> bool:
        return find_manifest(sig.path) is not None

    @staticmethod
    def priority() -> int:
        return 30  # Low priority — only used as a fallback

    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        manifest_path = find_manifest(path)
        if manifest_path is None:
            raise FileNotFoundError(f"No manifest found for {path}")

        config = _parse_manifest(manifest_path)
        fmt = config.get("format", "binary_particle")

        if fmt == "binary_particle":
            return self._build_binary_particle(path, config, **kwargs)
        elif fmt == "binary_grid":
            return self._build_binary_grid(path, config, **kwargs)
        else:
            raise ValueError(f"Unknown format in manifest: {fmt}")

    def _build_binary_particle(self, path, config, **kwargs):
        """Load flat binary particle data using manifest-defined layout."""
        ir = DatasetIR(
            data_kind=DataKind.PARTICLE,
            source_path=path,
            dataset_name=Path(path).stem,
        )

        # Units from manifest, overridden by kwargs
        units_cfg = config.get("units", {})
        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit", units_cfg.get("length_unit")),
            mass_unit=kwargs.get("mass_unit", units_cfg.get("mass_unit")),
            time_unit=kwargs.get("time_unit", units_cfg.get("time_unit")),
            velocity_unit=kwargs.get("velocity_unit", units_cfg.get("velocity_unit")),
        )

        # Bbox
        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        elif "bbox" in config:
            ir.bbox = np.array(config["bbox"])

        # Parse binary data
        endian = config.get("endianness", "little")
        byte_order = "<" if endian == "little" else ">"
        dtype_spec = config.get("dtype", {})
        field_map = config.get("field_map", {})

        # Build a structured dtype
        dt_fields = []
        for name, dtype_str in dtype_spec.items():
            np_dtype = DTYPE_MAP.get(dtype_str, np.float64)
            dt_fields.append((name, f"{byte_order}{np_dtype().dtype.str[1:]}"))

        if dt_fields:
            dt = np.dtype(dt_fields)
            data = np.fromfile(path, dtype=dt)

            for name in dtype_spec:
                universal = field_map.get(name, name)
                ir.particle_data[("io", universal)] = data[name].astype(np.float64)
                ir.fields.append(FieldSpec(
                    native_name=name,
                    universal_name=universal,
                    field_type="io",
                ))

        mylog.info(
            "ManifestAdapter: loaded %d particles from binary %s",
            len(data) if dt_fields else 0,
            path,
        )
        return ir

    def _build_binary_grid(self, path, config, **kwargs):
        """Load flat binary grid data using manifest-defined layout."""
        ir = DatasetIR(
            data_kind=DataKind.UNIGRID,
            source_path=path,
            dataset_name=Path(path).stem,
        )

        grid_dims = config.get("grid_dims", kwargs.get("grid_dims"))
        if grid_dims is None:
            raise ValueError("grid_dims is required for binary_grid format")
        ir.domain_dimensions = np.array(grid_dims)

        units_cfg = config.get("units", {})
        ir.units = UnitSpec(
            length_unit=kwargs.get("length_unit", units_cfg.get("length_unit")),
            mass_unit=kwargs.get("mass_unit", units_cfg.get("mass_unit")),
            time_unit=kwargs.get("time_unit", units_cfg.get("time_unit")),
        )

        if "bbox" in kwargs:
            ir.bbox = np.array(kwargs["bbox"])
        elif "bbox" in config:
            ir.bbox = np.array(config["bbox"])
        else:
            ir.bbox = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

        endian = config.get("endianness", "little")
        byte_order = "<" if endian == "little" else ">"
        dtype_spec = config.get("dtype", {})
        field_map = config.get("field_map", {})

        n_cells = int(np.prod(ir.domain_dimensions))

        # Read fields sequentially from binary
        offset = 0
        with open(path, "rb") as f:
            raw = f.read()

        for name, dtype_str in dtype_spec.items():
            np_dtype = DTYPE_MAP.get(dtype_str, np.float64)
            dt = np.dtype(f"{byte_order}{np_dtype().dtype.str[1:]}")
            field_size = n_cells * dt.itemsize

            arr = np.frombuffer(raw, dtype=dt, count=n_cells, offset=offset)
            arr = arr.reshape(ir.domain_dimensions).astype(np.float64)
            offset += field_size

            universal = field_map.get(name, name)
            ir.mesh_data[("gas", universal)] = arr
            ir.fields.append(FieldSpec(
                native_name=name,
                universal_name=universal,
                field_type="gas",
            ))

        mylog.info(
            "ManifestAdapter: loaded grid %s with %d fields from %s",
            ir.domain_dimensions,
            len(dtype_spec),
            path,
        )
        return ir
