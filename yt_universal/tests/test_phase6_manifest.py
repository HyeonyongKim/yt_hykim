"""
Phase 6 tests: Manifest-assisted binary loading.
"""

import struct

import numpy as np
import pytest

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from yt_universal import load_universal
from yt_universal.adapters.manifest import find_manifest


def _create_binary_particles(path, n=100):
    """Create a flat binary particle file (x, y, z, mass as float32)."""
    rng = np.random.default_rng(42)
    with open(path, "wb") as f:
        for _ in range(n):
            x, y, z = rng.uniform(0, 100, 3).astype(np.float32)
            m = np.float32(rng.uniform(1e8, 1e10))
            f.write(struct.pack("<ffff", x, y, z, m))


def _create_manifest_particle(path):
    """Create a yt_universal.yaml for binary particles."""
    manifest = {
        "format": "binary_particle",
        "endianness": "little",
        "dtype": {
            "x": "float32",
            "y": "float32",
            "z": "float32",
            "mass": "float32",
        },
        "units": {
            "length_unit": "kpc",
            "mass_unit": "Msun",
        },
        "field_map": {
            "x": "particle_position_x",
            "y": "particle_position_y",
            "z": "particle_position_z",
            "mass": "particle_mass",
        },
        "bbox": [[0, 100], [0, 100], [0, 100]],
    }
    with open(path, "w") as f:
        yaml.dump(manifest, f, sort_keys=False)


def _create_binary_grid(path, dims=(8, 8, 8)):
    """Create a flat binary grid file (density then temperature as float64)."""
    rng = np.random.default_rng(42)
    n = int(np.prod(dims))
    density = rng.uniform(1e-24, 1e-23, n).astype(np.float64)
    temperature = rng.uniform(1e4, 1e5, n).astype(np.float64)
    with open(path, "wb") as f:
        f.write(density.tobytes())
        f.write(temperature.tobytes())


def _create_manifest_grid(path, dims=(8, 8, 8)):
    """Create a yt_universal.yaml for binary grid data."""
    manifest = {
        "format": "binary_grid",
        "endianness": "little",
        "grid_dims": list(dims),
        "dtype": {
            "density": "float64",
            "temperature": "float64",
        },
        "units": {
            "length_unit": "Mpc",
        },
        "field_map": {
            "density": "density",
            "temperature": "temperature",
        },
        "bbox": [[0, 1], [0, 1], [0, 1]],
    }
    with open(path, "w") as f:
        yaml.dump(manifest, f, sort_keys=False)


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestManifestDiscovery:
    def test_find_manifest(self, tmp_path):
        manifest_path = tmp_path / "yt_universal.yaml"
        manifest_path.write_text("format: test\n")

        result = find_manifest(str(tmp_path / "data.bin"))
        assert result is not None
        assert "yt_universal.yaml" in result

    def test_no_manifest(self, tmp_path):
        result = find_manifest(str(tmp_path / "data.bin"))
        assert result is None


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestManifestParticleLoading:
    def test_load_binary_particles(self, tmp_path):
        """Binary particles with manifest should load correctly."""
        data_path = str(tmp_path / "particles.bin")
        manifest_path = str(tmp_path / "yt_universal.yaml")

        _create_binary_particles(data_path, n=50)
        _create_manifest_particle(manifest_path)

        ds = load_universal(data_path)
        ad = ds.all_data()

        pos = ad["all", "particle_position_x"]
        assert pos.size == 50

        mass = ad["all", "particle_mass"]
        assert mass.size == 50


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestManifestGridLoading:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_binary_grid(self, tmp_path):
        """Binary grid with manifest should load correctly."""
        dims = (8, 8, 8)
        data_path = str(tmp_path / "grid.bin")
        manifest_path = str(tmp_path / "yt_universal.yaml")

        _create_binary_grid(data_path, dims=dims)
        _create_manifest_grid(manifest_path, dims=dims)

        ds = load_universal(data_path)
        ad = ds.all_data()

        rho = ad["gas", "density"]
        assert rho.size > 0

        temp = ad["gas", "temperature"]
        assert temp.size > 0
