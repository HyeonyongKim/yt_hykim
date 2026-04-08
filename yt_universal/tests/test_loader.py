"""
Phase 1 tests for yt_universal.

Tests that load_universal works with yt's built-in stream loaders
and that universal field aliases are correctly attached.
"""

import numpy as np
import pytest
import yt

from yt_universal import load_universal
from yt_universal.fields.aliases import attach_universal_aliases


class TestAttachUniversalAliases:
    """Test alias attachment on datasets created via yt stream loaders."""

    def _make_uniform_grid(self):
        """Create a simple uniform grid dataset with known fields."""
        data = {
            ("gas", "density"): np.random.uniform(1e-24, 1e-23, (16, 16, 16)),
            ("gas", "temperature"): np.random.uniform(1e4, 1e5, (16, 16, 16)),
            ("gas", "velocity_x"): np.random.uniform(-1e5, 1e5, (16, 16, 16)),
            ("gas", "velocity_y"): np.random.uniform(-1e5, 1e5, (16, 16, 16)),
            ("gas", "velocity_z"): np.random.uniform(-1e5, 1e5, (16, 16, 16)),
        }
        bbox = np.array([[0, 1], [0, 1], [0, 1]])
        ds = yt.load_uniform_grid(
            data,
            [16, 16, 16],
            length_unit="Mpc",
            bbox=bbox,
        )
        return ds

    def _make_particle_dataset(self):
        """Create a simple particle dataset."""
        n_particles = 1000
        data = {
            ("io", "particle_position_x"): np.random.uniform(0, 1, n_particles),
            ("io", "particle_position_y"): np.random.uniform(0, 1, n_particles),
            ("io", "particle_position_z"): np.random.uniform(0, 1, n_particles),
            ("io", "particle_mass"): np.ones(n_particles) * 1e10,
            ("io", "particle_velocity_x"): np.random.uniform(-1e5, 1e5, n_particles),
            ("io", "particle_velocity_y"): np.random.uniform(-1e5, 1e5, n_particles),
            ("io", "particle_velocity_z"): np.random.uniform(-1e5, 1e5, n_particles),
        }
        bbox = np.array([[0, 1], [0, 1], [0, 1]])
        ds = yt.load_particles(data, length_unit="Mpc", bbox=bbox)
        return ds

    def test_gas_aliases_on_uniform_grid(self):
        """Universal gas aliases should be accessible after attachment."""
        ds = self._make_uniform_grid()
        ds = attach_universal_aliases(ds)

        ad = ds.all_data()
        # These should all be accessible via universal names
        assert ("gas", "density") in ds.derived_field_list
        assert ("gas", "temperature") in ds.derived_field_list
        assert ("gas", "velocity_x") in ds.derived_field_list

        # Verify data is actually retrievable
        rho = ad["gas", "density"]
        assert rho.size > 0

    def test_particle_aliases(self):
        """Universal particle aliases should be accessible after attachment."""
        ds = self._make_particle_dataset()
        ds = attach_universal_aliases(ds)

        ad = ds.all_data()
        # Particle fields under "all" type
        mass = ad["all", "particle_mass"]
        assert mass.size > 0

        pos_x = ad["all", "particle_position_x"]
        assert pos_x.size > 0

    def test_no_duplicate_aliases(self):
        """Calling attach_universal_aliases twice should not raise errors."""
        ds = self._make_uniform_grid()
        ds = attach_universal_aliases(ds)
        ds = attach_universal_aliases(ds)  # second call should be a no-op

        ad = ds.all_data()
        rho = ad["gas", "density"]
        assert rho.size > 0

    def test_returns_same_dataset(self):
        """attach_universal_aliases should return the same dataset object."""
        ds = self._make_uniform_grid()
        ds2 = attach_universal_aliases(ds)
        assert ds is ds2


class TestLoadUniversal:
    """Test the load_universal entry point with stream-created temp files."""

    def test_load_nonexistent_file_raises(self):
        """Loading a nonexistent file should raise an error."""
        with pytest.raises(Exception):
            load_universal("/nonexistent/path/to/data.hdf5")
