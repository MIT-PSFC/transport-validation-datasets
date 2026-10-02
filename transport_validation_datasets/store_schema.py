"""The signals every device store shares, with their SI units and IMAS data dictionary paths.

This package writes them for C-Mod and MAST,
and POPSIM-Transport-Predictor for DIII-D and TCV.
Each device adds its own description of how a signal was measured.
"""

import xarray as xr

# Auxiliary heating, the summary/heating_current_drive powers.
# Every device records them as launched into the vessel, net of reflection, not as absorbed.
HEATING_POWERS = ("power_nbi", "power_ic", "power_lh", "power_ec")
# The power put into the plasma, ohmic and auxiliary
INPUT_POWERS = ("power_ohm", *HEATING_POWERS)
POWER_SIGNALS = ("power_ohm", "power_radiated", *HEATING_POWERS)

_HEATING_REF = "/summary/heating_current_drive"
_PROFILES_REF = "/core_profiles/profiles_1d(itime)/electrons"

# Every store signal, in store order: its SI unit, and its IMAS path where IMAS has a leaf for it
STORE_SIGNAL_ATTRS = {
    "ip": {"units": "A", "ref": "/summary/global_quantities/ip/value"},
    "b0": {"units": "T", "ref": "/summary/global_quantities/b0/value"},
    # Per shot, on (shot,)
    "r0": {"units": "m", "ref": "/summary/global_quantities/r0/value"},
    "energy_mhd": {
        "units": "J",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/energy_mhd",
    },
    "beta_tor_norm": {
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/beta_tor_norm",
    },
    "n_e_line_average": {"units": "m^-3", "ref": "/summary/line_average/n_e/value"},
    "minor_radius": {
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/minor_radius",
    },
    "geometric_axis_r": {
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/geometric_axis/r",
    },
    "elongation": {
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/elongation",
    },
    "triangularity_upper": {
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_upper",
    },
    "triangularity_lower": {
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_lower",
    },
    "power_ohm": {"units": "W", "ref": "/summary/global_quantities/power_ohm/value"},
    "power_radiated": {
        "units": "W",
        "ref": "/summary/global_quantities/power_radiated/value",
    },
    "power_nbi": {"units": "W", "ref": f"{_HEATING_REF}/power_launched_nbi/value"},
    "power_ic": {"units": "W", "ref": f"{_HEATING_REF}/power_launched_ic/value"},
    "power_lh": {"units": "W", "ref": f"{_HEATING_REF}/power_launched_lh/value"},
    "power_ec": {"units": "W", "ref": f"{_HEATING_REF}/power_launched_ec/value"},
    "t_e": {"units": "eV", "ref": f"{_PROFILES_REF}/temperature"},
    "t_e_error": {"units": "eV", "ref": f"{_PROFILES_REF}/temperature_error_upper"},
    "t_e_gradient": {"units": "eV per unit rho_tor_norm"},
    "t_e_gradient_error": {"units": "eV per unit rho_tor_norm"},
    "n_e": {"units": "m^-3", "ref": f"{_PROFILES_REF}/density"},
    "n_e_error": {"units": "m^-3", "ref": f"{_PROFILES_REF}/density_error_upper"},
    "n_e_gradient": {"units": "m^-3 per unit rho_tor_norm"},
    "n_e_gradient_error": {"units": "m^-3 per unit rho_tor_norm"},
    "fresh_profile": {"units": "dimensionless"},
    "fresh_equilibrium": {"units": "dimensionless"},
}
STORE_SIGNALS = tuple(STORE_SIGNAL_ATTRS)

# Every 0D signal of a store, finite at every stored time (the finite filter of every device)
DATASET_0D_SIGNALS = (
    "ip",
    "b0",
    "energy_mhd",
    "beta_tor_norm",
    "n_e_line_average",
    "minor_radius",
    "geometric_axis_r",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    *POWER_SIGNALS,
)


def apply_signal_attrs(ds: xr.Dataset, device_attrs: dict[str, dict]) -> xr.Dataset:
    """Attach a device's attributes, then the shared units and IMAS paths, which win over the device's own.

    Args:
        ds: Dataset whose variables are updated in place.
        device_attrs: {name: attrs} of the device, its descriptions above all.

    Returns:
        The same dataset.
    """
    for name, attrs in device_attrs.items():
        if name in ds.variables:
            ds[name].attrs.update(attrs)
    for name, attrs in STORE_SIGNAL_ATTRS.items():
        if name in ds.variables:
            ds[name].attrs.update(attrs)
    return ds
