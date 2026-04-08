"""
Universal field alias definitions and attachment logic.

Defines a minimum guaranteed field contract so that users can always
access fields like ("gas", "density") and ("all", "particle_mass")
regardless of the underlying simulation format.
"""

import logging

mylog = logging.getLogger(__name__)

# Universal mesh/gas field aliases.
# Each entry: (universal_alias, [possible_native_names])
# The first matching native name found on the dataset will be aliased.
UNIVERSAL_GAS_FIELDS = {
    "density": [
        "density",
        "dens",
        "Density",
        "rho",
        "gas_density",
    ],
    "temperature": [
        "temperature",
        "temp",
        "Temperature",
        "gas_temperature",
    ],
    "pressure": [
        "pressure",
        "pres",
        "Pressure",
        "gas_pressure",
    ],
    "velocity_x": [
        "velocity_x",
        "velx",
        "vel_x",
        "Velocityx",
        "gas_velocity_x",
    ],
    "velocity_y": [
        "velocity_y",
        "vely",
        "vel_y",
        "Velocityy",
        "gas_velocity_y",
    ],
    "velocity_z": [
        "velocity_z",
        "velz",
        "vel_z",
        "Velocityz",
        "gas_velocity_z",
    ],
    "metallicity": [
        "metallicity",
        "metal",
        "Metallicity",
        "Z_gas",
    ],
    "specific_thermal_energy": [
        "specific_thermal_energy",
        "specific_energy",
        "thermal_energy",
        "eint",
    ],
    "magnetic_field_x": [
        "magnetic_field_x",
        "mag_field_x",
        "Bx",
        "Bfield_x",
        "magx",
    ],
    "magnetic_field_y": [
        "magnetic_field_y",
        "mag_field_y",
        "By",
        "Bfield_y",
        "magy",
    ],
    "magnetic_field_z": [
        "magnetic_field_z",
        "mag_field_z",
        "Bz",
        "Bfield_z",
        "magz",
    ],
}

# Universal particle field aliases.
UNIVERSAL_PARTICLE_FIELDS = {
    "particle_position_x": [
        "particle_position_x",
        "Coordinates_x",
        "x",
        "pos_x",
    ],
    "particle_position_y": [
        "particle_position_y",
        "Coordinates_y",
        "y",
        "pos_y",
    ],
    "particle_position_z": [
        "particle_position_z",
        "Coordinates_z",
        "z",
        "pos_z",
    ],
    "particle_velocity_x": [
        "particle_velocity_x",
        "Velocities_x",
        "vx",
        "vel_x",
    ],
    "particle_velocity_y": [
        "particle_velocity_y",
        "Velocities_y",
        "vy",
        "vel_y",
    ],
    "particle_velocity_z": [
        "particle_velocity_z",
        "Velocities_z",
        "vz",
        "vel_z",
    ],
    "particle_mass": [
        "particle_mass",
        "Masses",
        "mass",
        "particle_masses",
    ],
    "particle_index": [
        "particle_index",
        "ParticleIDs",
        "pid",
        "particle_id",
    ],
    "particle_type": [
        "particle_type",
        "ParticleType",
        "ptype",
    ],
}


def _find_field_on_dataset(ds, field_type, candidate_names):
    """Search the dataset's field list for the first matching candidate name.

    Returns the full field tuple (field_type, field_name) if found, else None.
    """
    for name in candidate_names:
        field_key = (field_type, name)
        if field_key in ds.field_list or field_key in ds.derived_field_list:
            return field_key
    return None


def attach_universal_aliases(ds):
    """Attach universal field aliases to an existing yt dataset.

    For each universal field name, search the dataset for a matching native
    field. If found and the universal alias doesn't already exist, register
    it via the dataset's field_info.alias() mechanism.

    Parameters
    ----------
    ds : yt Dataset
        A loaded yt dataset.

    Returns
    -------
    ds : yt Dataset
        The same dataset, with universal aliases attached.
    """
    fi = ds.field_info
    aliases_added = []

    # Determine gas/fluid field types present on the dataset
    fluid_types = set()
    for ftype, _fname in ds.field_list:
        if ftype not in ds.particle_types:
            fluid_types.add(ftype)
    # Always try "gas" as a target alias type
    gas_ftype = "gas"

    # Attach gas/mesh aliases
    for universal_name, candidates in UNIVERSAL_GAS_FIELDS.items():
        alias_key = (gas_ftype, universal_name)
        # Skip if the alias already exists
        if alias_key in fi:
            continue
        # Search across all fluid field types for a match
        for ftype in fluid_types:
            source = _find_field_on_dataset(ds, ftype, candidates)
            if source is not None:
                try:
                    fi.alias(alias_key, source)
                    aliases_added.append(alias_key)
                except Exception:
                    mylog.debug(
                        "Failed to alias %s -> %s", alias_key, source
                    )
                break

    # Determine particle types present
    particle_types = set()
    for ftype, _fname in ds.field_list:
        if ftype in ds.particle_types:
            particle_types.add(ftype)

    # Attach particle aliases under "all" type
    all_ptype = "all"
    for universal_name, candidates in UNIVERSAL_PARTICLE_FIELDS.items():
        alias_key = (all_ptype, universal_name)
        if alias_key in fi:
            continue
        for ptype in particle_types:
            source = _find_field_on_dataset(ds, ptype, candidates)
            if source is not None:
                try:
                    fi.alias(alias_key, source)
                    aliases_added.append(alias_key)
                except Exception:
                    mylog.debug(
                        "Failed to alias %s -> %s", alias_key, source
                    )
                break

    if aliases_added:
        mylog.info(
            "yt_universal: attached %d universal aliases", len(aliases_added)
        )
    else:
        mylog.debug("yt_universal: no new aliases needed")

    return ds
