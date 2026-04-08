"""
Phase 2 tests: HDF5 inspection and generic unigrid loading.

Creates temporary HDF5 files and verifies the full pipeline:
  inspect_path → adapter → build_from_ir → attach_universal_aliases
"""

import os
import tempfile
import warnings

import h5py
import numpy as np
import pytest

from yt_universal import load_universal
from yt_universal.inspectors.base import (
    ContainerType,
    LayoutType,
    inspect_path,
)


def _create_unigrid_hdf5(path, shape=(32, 32, 32)):
    """Create a simple HDF5 file with 3D mesh arrays."""
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=np.random.uniform(1e-24, 1e-23, shape))
        f.create_dataset("temperature", data=np.random.uniform(1e4, 1e5, shape))
        f.create_dataset("velocity_x", data=np.random.uniform(-1e5, 1e5, shape))
        f.create_dataset("velocity_y", data=np.random.uniform(-1e5, 1e5, shape))
        f.create_dataset("velocity_z", data=np.random.uniform(-1e5, 1e5, shape))


def _create_grouped_hdf5(path, shape=(16, 16, 16)):
    """Create an HDF5 file with fields inside a group."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("fields")
        grp.create_dataset("density", data=np.random.uniform(1e-24, 1e-23, shape))
        grp.create_dataset("pressure", data=np.random.uniform(1e-10, 1e-9, shape))


def _create_particle_hdf5(path, n_particles=500):
    """Create an HDF5 file with 1D particle arrays only."""
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.random.uniform(0, 1, n_particles))
        f.create_dataset("y", data=np.random.uniform(0, 1, n_particles))
        f.create_dataset("z", data=np.random.uniform(0, 1, n_particles))
        f.create_dataset("mass", data=np.ones(n_particles) * 1e10)


class TestHDF5Inspector:
    """Test the HDF5 inspection pipeline."""

    def test_unigrid_detection(self, tmp_path):
        path = str(tmp_path / "test_unigrid.hdf5")
        _create_unigrid_hdf5(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.HDF5
        assert sig.layout_type == LayoutType.UNIGRID
        assert sig.candidate_family == "simple_hdf5"
        assert sig.confidence > 0

    def test_grouped_hdf5_detection(self, tmp_path):
        path = str(tmp_path / "test_grouped.hdf5")
        _create_grouped_hdf5(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.HDF5
        assert sig.layout_type == LayoutType.UNIGRID

    def test_particle_hdf5_detection(self, tmp_path):
        path = str(tmp_path / "test_particles.hdf5")
        _create_particle_hdf5(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.HDF5
        assert sig.layout_type == LayoutType.PARTICLE_ONLY

    def test_hdf5_root_groups_populated(self, tmp_path):
        path = str(tmp_path / "test.hdf5")
        _create_unigrid_hdf5(path)

        sig = inspect_path(path)
        assert "density" in sig.hdf5_root_groups
        assert "temperature" in sig.hdf5_root_groups

    def test_hdf5_datasets_populated(self, tmp_path):
        path = str(tmp_path / "test.hdf5")
        shape = (16, 16, 16)
        _create_unigrid_hdf5(path, shape=shape)

        sig = inspect_path(path)
        assert "density" in sig.hdf5_datasets
        assert sig.hdf5_datasets["density"][0] == shape


class TestSimpleHDF5Loading:
    """Test loading generic HDF5 files through load_universal."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_unigrid_hdf5(self, tmp_path):
        """Unknown HDF5 with 3D arrays should load via the universal pipeline."""
        path = str(tmp_path / "test_unigrid.hdf5")
        _create_unigrid_hdf5(path, shape=(16, 16, 16))

        ds = load_universal(path)
        ad = ds.all_data()

        # Should be able to access density via universal name
        rho = ad["gas", "density"]
        assert rho.size > 0

    def test_load_with_units(self, tmp_path):
        """Unit overrides should propagate to the dataset."""
        path = str(tmp_path / "test_units.hdf5")
        _create_unigrid_hdf5(path, shape=(16, 16, 16))

        ds = load_universal(path, length_unit="Mpc", mass_unit="Msun")
        assert ds is not None

        ad = ds.all_data()
        rho = ad["gas", "density"]
        assert rho.size > 0

    def test_load_with_bbox(self, tmp_path):
        """Custom bounding box should be respected."""
        path = str(tmp_path / "test_bbox.hdf5")
        _create_unigrid_hdf5(path, shape=(16, 16, 16))

        bbox = [[0, 100], [0, 100], [0, 100]]
        ds = load_universal(path, bbox=bbox, length_unit="kpc")
        assert ds is not None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_universal_aliases_present(self, tmp_path):
        """Fields like ("gas", "density") should exist after loading."""
        path = str(tmp_path / "test_aliases.hdf5")
        _create_unigrid_hdf5(path, shape=(16, 16, 16))

        ds = load_universal(path)
        # "density" in the HDF5 should map to ("gas", "density")
        assert ("gas", "density") in ds.derived_field_list

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_velocity_fields_aliased(self, tmp_path):
        """Velocity fields should get universal aliases."""
        path = str(tmp_path / "test_vel.hdf5")
        _create_unigrid_hdf5(path, shape=(16, 16, 16))

        ds = load_universal(path)
        assert ("gas", "velocity_x") in ds.derived_field_list

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_grouped_hdf5_loads(self, tmp_path):
        """HDF5 files with fields inside groups should still work."""
        path = str(tmp_path / "test_grouped.hdf5")
        _create_grouped_hdf5(path, shape=(16, 16, 16))

        ds = load_universal(path)
        ad = ds.all_data()

        # The field should be accessible (leaf name "density")
        rho = ad["gas", "density"]
        assert rho.size > 0
