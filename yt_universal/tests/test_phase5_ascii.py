"""
Phase 5 tests: ASCII particle and grid file loading.
"""

import numpy as np
import pytest

from yt_universal import load_universal
from yt_universal.inspectors.base import ContainerType, inspect_path


def _create_particle_csv(path, n=200):
    """Create a CSV particle file with header."""
    rng = np.random.default_rng(42)
    with open(path, "w") as f:
        f.write("# x y z vx vy vz mass\n")
        for i in range(n):
            x, y, z = rng.uniform(0, 100, 3)
            vx, vy, vz = rng.uniform(-200, 200, 3)
            m = rng.uniform(1e8, 1e10)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {vx:.3f} {vy:.3f} {vz:.3f} {m:.4e}\n")


def _create_particle_csv_comma(path, n=100):
    """Create a comma-delimited particle file."""
    rng = np.random.default_rng(42)
    with open(path, "w") as f:
        f.write("x,y,z,mass\n")
        for i in range(n):
            x, y, z = rng.uniform(0, 50, 3)
            m = rng.uniform(1e9, 1e11)
            f.write(f"{x:.6f},{y:.6f},{z:.6f},{m:.4e}\n")


def _create_grid_ascii(path, n=8):
    """Create an ASCII file with grid data (n^3 rows)."""
    rng = np.random.default_rng(42)
    with open(path, "w") as f:
        f.write("# density temperature\n")
        for i in range(n ** 3):
            rho = rng.uniform(1e-24, 1e-23)
            temp = rng.uniform(1e4, 1e5)
            f.write(f"{rho:.6e} {temp:.6e}\n")


class TestASCIIInspection:
    def test_ascii_particle_detected(self, tmp_path):
        path = str(tmp_path / "particles.dat")
        _create_particle_csv(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.ASCII
        assert sig.candidate_family == "ascii_particle"

    def test_ascii_column_names_parsed(self, tmp_path):
        path = str(tmp_path / "particles.dat")
        _create_particle_csv(path)

        sig = inspect_path(path)
        # Should detect column names from the header comment
        assert "x" in sig.hdf5_root_groups
        assert "mass" in sig.hdf5_root_groups


class TestASCIIParticleLoading:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_particle_csv(self, tmp_path):
        """Space-delimited particle file should load."""
        path = str(tmp_path / "particles.dat")
        _create_particle_csv(path, n=100)

        ds = load_universal(path, length_unit="kpc")
        ad = ds.all_data()

        pos = ad["all", "particle_position_x"]
        assert pos.size == 100

        mass = ad["all", "particle_mass"]
        assert mass.size == 100

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_comma_csv(self, tmp_path):
        """Comma-delimited particle file should load."""
        path = str(tmp_path / "catalog.csv")
        _create_particle_csv_comma(path, n=50)

        ds = load_universal(path, length_unit="Mpc")
        ad = ds.all_data()

        pos = ad["all", "particle_position_x"]
        assert pos.size == 50


class TestASCIIGridLoading:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_grid_ascii(self, tmp_path):
        """ASCII grid data with explicit grid_dims should load."""
        path = str(tmp_path / "grid.dat")
        _create_grid_ascii(path, n=8)

        ds = load_universal(path, grid_dims=(8, 8, 8), length_unit="kpc")
        ad = ds.all_data()

        rho = ad["gas", "density"]
        assert rho.size > 0
