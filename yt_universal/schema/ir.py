"""
Canonical Internal Representation (IR) for datasets.

The IR is a format-neutral description of a dataset that can be
converted into yt stream datasets by the builders module.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class Geometry(Enum):
    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    POLAR = "polar"


class DataKind(Enum):
    """What kind of data this IR represents."""
    UNIGRID = auto()
    AMR = auto()
    PARTICLE = auto()
    MIXED = auto()


@dataclass
class FieldSpec:
    """Description of a single field in the dataset."""
    native_name: str
    universal_name: str | None = None
    units: str = ""
    field_type: str = "gas"  # "gas", "io", or a particle type name


@dataclass
class UnitSpec:
    """Unit system specification for the dataset."""
    length_unit: str | None = None
    mass_unit: str | None = None
    time_unit: str | None = None
    velocity_unit: str | None = None
    magnetic_unit: str | None = None


@dataclass
class GridPatch:
    """Description of a single grid patch (for AMR datasets)."""
    left_edge: np.ndarray
    right_edge: np.ndarray
    dimensions: np.ndarray
    level: int = 0
    data: dict = field(default_factory=dict)  # field_name -> numpy array


@dataclass
class DatasetIR:
    """Canonical internal representation of a dataset.

    This is the bridge between inspectors/adapters and yt stream builders.
    Adapters populate this from file data, and builders convert it into
    yt datasets.
    """
    # What kind of dataset
    data_kind: DataKind = DataKind.UNIGRID

    # Geometry
    geometry: Geometry = Geometry.CARTESIAN

    # Domain info
    domain_dimensions: np.ndarray | None = None
    bbox: np.ndarray | None = None  # shape (3, 2): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    periodicity: tuple[bool, bool, bool] = (True, True, True)

    # Units
    units: UnitSpec = field(default_factory=UnitSpec)

    # Field definitions
    fields: list[FieldSpec] = field(default_factory=list)

    # Mesh data: field_name -> numpy array (for unigrid)
    mesh_data: dict = field(default_factory=dict)

    # Particle data: field_name -> numpy array
    particle_data: dict = field(default_factory=dict)

    # AMR grid patches
    grid_patches: list[GridPatch] = field(default_factory=list)

    # Source path (for lazy reading)
    source_path: str = ""

    # Dataset name
    dataset_name: str = "UniversalData"
