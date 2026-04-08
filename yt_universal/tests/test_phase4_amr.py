"""
Phase 4 tests: AMR fallback loading from HDF5.
"""

import h5py
import numpy as np
import pytest

from yt_universal import load_universal
from yt_universal.inspectors.base import inspect_path, LayoutType


def _create_amr_hdf5_flat(path):
    """Create an HDF5 file with multiple 3D arrays of different shapes (AMR-like)."""
    with h5py.File(path, "w") as f:
        # Coarse grid (level 0)
        f.create_dataset("density_coarse", data=np.random.uniform(1e-24, 1e-23, (8, 8, 8)))
        f.create_dataset("temperature_coarse", data=np.random.uniform(1e4, 1e5, (8, 8, 8)))
        # Fine grid (level 1)
        f.create_dataset("density_fine", data=np.random.uniform(1e-24, 1e-23, (16, 16, 16)))
        f.create_dataset("temperature_fine", data=np.random.uniform(1e4, 1e5, (16, 16, 16)))


def _create_amr_hdf5_with_attrs(path):
    """Create an HDF5 file with grid groups that have edge/level attributes."""
    with h5py.File(path, "w") as f:
        # Grid 0 (level 0, covers full domain)
        g0 = f.create_group("grid_0000")
        g0.attrs["left_edge"] = [0.0, 0.0, 0.0]
        g0.attrs["right_edge"] = [1.0, 1.0, 1.0]
        g0.attrs["level"] = 0
        g0.create_dataset("density", data=np.random.uniform(1e-24, 1e-23, (8, 8, 8)))

        # Grid 1 (level 1, sub-region)
        g1 = f.create_group("grid_0001")
        g1.attrs["left_edge"] = [0.0, 0.0, 0.0]
        g1.attrs["right_edge"] = [0.5, 0.5, 0.5]
        g1.attrs["level"] = 1
        g1.create_dataset("density", data=np.random.uniform(1e-24, 1e-23, (8, 8, 8)))

        # Grid 2 (level 1, another sub-region)
        g2 = f.create_group("grid_0002")
        g2.attrs["left_edge"] = [0.5, 0.5, 0.5]
        g2.attrs["right_edge"] = [1.0, 1.0, 1.0]
        g2.attrs["level"] = 1
        g2.create_dataset("density", data=np.random.uniform(1e-24, 1e-23, (8, 8, 8)))


class TestAMRInspection:
    """Test that AMR-like HDF5 files are correctly classified."""

    def test_flat_amr_detected(self, tmp_path):
        path = str(tmp_path / "amr_flat.hdf5")
        _create_amr_hdf5_flat(path)

        sig = inspect_path(path)
        assert sig.layout_type == LayoutType.AMR


class TestAMRLoading:
    """Test loading AMR HDF5 files through the universal pipeline."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_amr_with_attrs(self, tmp_path):
        """HDF5 with grid groups and edge attributes should load as AMR."""
        path = str(tmp_path / "amr_grids.hdf5")
        _create_amr_hdf5_with_attrs(path)

        ds = load_universal(path)
        ad = ds.all_data()

        rho = ad["gas", "density"]
        assert rho.size > 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_amr_data_accessible(self, tmp_path):
        """All grid data should be accessible through the dataset."""
        path = str(tmp_path / "amr_levels.hdf5")
        _create_amr_hdf5_with_attrs(path)

        ds = load_universal(path)
        ad = ds.all_data()
        # All grids should have density field accessible
        rho = ad["gas", "density"]
        assert rho.size > 0
        # Dataset should have grids
        assert len(ds.index.grids) >= 1
