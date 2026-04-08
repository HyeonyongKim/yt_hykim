"""
Phase 3 tests: Gadget-family HDF5 normalization.

Creates mock Gadget-like HDF5 files and verifies they load through
the universal pipeline with correct particle field mapping.
"""

import h5py
import numpy as np
import pytest

from yt_universal import load_universal
from yt_universal.inspectors.base import inspect_path, ContainerType


def _create_gadget_hdf5(path, n_gas=500, n_dm=1000):
    """Create a mock Gadget-like HDF5 snapshot."""
    with h5py.File(path, "w") as f:
        # Header
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = np.array([n_gas, n_dm, 0, 0, 0, 0])
        header.attrs["NumPart_Total"] = np.array([n_gas, n_dm, 0, 0, 0, 0])
        header.attrs["BoxSize"] = 100.0
        header.attrs["Time"] = 0.5
        header.attrs["Redshift"] = 1.0
        header.attrs["Omega0"] = 0.3
        header.attrs["OmegaLambda"] = 0.7
        header.attrs["HubbleParam"] = 0.7
        header.attrs["UnitLength_in_cm"] = 3.085678e21  # kpc
        header.attrs["UnitMass_in_g"] = 1.989e43  # 1e10 Msun
        header.attrs["UnitVelocity_in_cm_per_s"] = 1e5  # km/s

        # PartType0 (Gas)
        if n_gas > 0:
            pt0 = f.create_group("PartType0")
            pt0.create_dataset("Coordinates", data=np.random.uniform(0, 100, (n_gas, 3)))
            pt0.create_dataset("Velocities", data=np.random.uniform(-200, 200, (n_gas, 3)))
            pt0.create_dataset("Masses", data=np.ones(n_gas) * 0.01)
            pt0.create_dataset("ParticleIDs", data=np.arange(1, n_gas + 1))
            pt0.create_dataset("InternalEnergy", data=np.random.uniform(1e3, 1e4, n_gas))
            pt0.create_dataset("Density", data=np.random.uniform(1e-4, 1e-2, n_gas))

        # PartType1 (Dark Matter)
        if n_dm > 0:
            pt1 = f.create_group("PartType1")
            pt1.create_dataset("Coordinates", data=np.random.uniform(0, 100, (n_dm, 3)))
            pt1.create_dataset("Velocities", data=np.random.uniform(-300, 300, (n_dm, 3)))
            pt1.create_dataset("Masses", data=np.ones(n_dm) * 0.1)
            pt1.create_dataset("ParticleIDs", data=np.arange(n_gas + 1, n_gas + n_dm + 1))


def _create_arepo_hdf5(path, n_gas=300):
    """Create a mock Arepo-like HDF5 snapshot."""
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = np.array([n_gas, 0, 0, 0, 0, 0])
        header.attrs["BoxSize"] = 50.0
        header.attrs["Time"] = 1.0

        # Config group (Arepo signature)
        config = f.create_group("Config")
        config.attrs["VORONOI"] = 1

        pt0 = f.create_group("PartType0")
        pt0.create_dataset("Coordinates", data=np.random.uniform(0, 50, (n_gas, 3)))
        pt0.create_dataset("Velocities", data=np.random.uniform(-100, 100, (n_gas, 3)))
        pt0.create_dataset("Masses", data=np.ones(n_gas) * 0.005)
        pt0.create_dataset("GFM_Metallicity", data=np.random.uniform(0, 0.03, n_gas))


class TestGadgetInspection:
    """Test that Gadget-like files are correctly classified."""

    def test_gadget_family_detected(self, tmp_path):
        path = str(tmp_path / "snap_000.hdf5")
        _create_gadget_hdf5(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.HDF5
        assert sig.candidate_family == "gadget_like"
        assert sig.yt_hint == "Gadget"
        assert sig.confidence >= 0.7

    def test_arepo_family_detected(self, tmp_path):
        path = str(tmp_path / "arepo_snap.hdf5")
        _create_arepo_hdf5(path)

        sig = inspect_path(path)
        assert sig.container_type == ContainerType.HDF5
        assert sig.candidate_family == "arepo_like"
        assert sig.yt_hint == "Arepo"


class TestGadgetLoading:
    """Test loading Gadget-family files through the universal pipeline."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_gadget_particles(self, tmp_path):
        """Gadget-like HDF5 should load with particle fields mapped."""
        path = str(tmp_path / "snap_000.hdf5")
        _create_gadget_hdf5(path, n_gas=100, n_dm=200)

        ds = load_universal(path)
        ad = ds.all_data()

        # Multi-ptype: each PartType has its own position fields
        # "all" union should combine them
        pos_x = ad["all", "particle_position_x"]
        assert pos_x.size == 300  # 100 gas + 200 DM

        mass = ad["all", "particle_mass"]
        assert mass.size == 300

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_gadget_velocities_split(self, tmp_path):
        """Vector fields like Velocities should be split into components."""
        path = str(tmp_path / "snap_vel.hdf5")
        # Single PartType → uses "io"
        _create_gadget_hdf5(path, n_gas=50, n_dm=0)

        ds = load_universal(path)
        ad = ds.all_data()

        vel_x = ad["all", "particle_velocity_x"]
        vel_y = ad["all", "particle_velocity_y"]
        vel_z = ad["all", "particle_velocity_z"]
        assert vel_x.size == 50
        assert vel_y.size == 50
        assert vel_z.size == 50

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_gadget_scalar_fields(self, tmp_path):
        """Scalar fields like Density, InternalEnergy should be mapped."""
        path = str(tmp_path / "snap_scalar.hdf5")
        # Single PartType → uses "io"
        _create_gadget_hdf5(path, n_gas=80, n_dm=0)

        ds = load_universal(path)
        ad = ds.all_data()

        density = ad["io", "density"]
        assert density.size == 80

        energy = ad["io", "specific_thermal_energy"]
        assert energy.size == 80

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_arepo_loading(self, tmp_path):
        """Arepo-like HDF5 should load with correct field mapping."""
        path = str(tmp_path / "arepo_snap.hdf5")
        _create_arepo_hdf5(path, n_gas=150)

        ds = load_universal(path)
        ad = ds.all_data()

        pos = ad["all", "particle_position_x"]
        assert pos.size == 150

        metal = ad["io", "metallicity"]
        assert metal.size == 150
