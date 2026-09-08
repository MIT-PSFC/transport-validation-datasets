"""MAST data workflow, built from the open-access Zarr stores on STFC ECHO S3.

Most of it comes out of the level 2 store: https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr
the 0D summary and equilibrium signals and the full GEQDSK reconstruction, whose flux map also places
the Thomson channels in rho.
Two things come from the level 1 store instead: the GEQDSK safety factor, which only level 1
publishes as a flux function (see _equilibrium_qpsi), and the Thomson profiles, which level 2
only carries interpolated onto a uniform (R, t) grid and without uncertainties
(see _thomson_dataset). No MDSplus is involved,
so this workflow runs anywhere with internet access.

Reads are slow, so staging runs in a thread pool of prepare_workers threads.
One shot costs ~60-90 s of round trips, which puts the
packaged 1101-shot list at a few hours on the default 8 threads.
"""

import numpy as np
import xarray as xr
from disruption_py.core.utils.math import causal_boxcar_smooth, interp1
from loguru import logger

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.generic import (
    efit_cocos_from_signs,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho,
    snap_to_grid,
    ts_channel_fit_rows,
)
from transport_validation_datasets.workflow import DataWorkflow

# Public MAST open data, no credentials needed
S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
LEVEL2_PATH = "mast/level2/shots"

# The raw diagnostic output. Two groups are read: EFM for the one GEQDSK field
# the level 2 store does not carry as a flux function (qpsi, see
# _equilibrium_qpsi), and AYC for the Thomson profiles (see _thomson_dataset).
LEVEL1_PATH = "mast/level1/shots"
LEVEL1_EFM_GROUP = "efm"
LEVEL1_TS_GROUP = "ayc"

# Variables the ayc group must carry to be usable. Everything else it holds
# (raw spectra, laser diagnostics, the pe product) is not needed here.
REQUIRED_TS_VARIABLES = ("radius", "te", "te_error", "ne", "ne_error")

# Two equilibrium samples closer than this are the same reconstruction slice
# published on two timebases (level 2 keeps the level 1 values, on a longer
# grid), so level 1 rows are matched to level 2 times, never interpolated [s].
EQUILIBRIUM_TIME_TOL = 1e-4

# Shotlist for M8 and M9 campaigns
DEFAULT_SHOTLIST_FILE = PACKAGE_ROOT / "machine" / "mast" / "mast_shotlist_M8_M9"

# The core Thomson system (AYC) views along a horizontal chord at the midplane,
# so every channel is at the same height and only its major radius varies.
TS_CHANNEL_Z = 0.0

# Plasma current magnitude that marks the end of the shot window [A].
# Only used to find the last time worth putting on the timebase,
# the real current cut is valid_filter["ip"].
SHOT_WINDOW_MIN_IP = 100e3

# Samples of the causal boxcar smoothing dIp/dt in the ohmic power calculation
OHMIC_SMOOTHING_SAMPLES = 10

# Channels this far outside the separatrix sit in the far SOL, where mapping
# through a magnetics-only reconstruction is not trustworthy.
MAX_FIT_RHO = 1.05

# Equilibrium fields the TS channel mapping reads, see _equilibrium_at_ts_times.
TS_MAPPING_EQUILIBRIUM_FIELDS = ("psirz", "simagx", "sibdry", "zmagx", "rmagx")

# How far a TS slice may reach for the reconstruction it is mapped through [s].
# EFIT runs on a 5 ms grid and the Thomson laser fires every ~4.2 ms, so the two
# almost never land on the same grid time: the offset is up to half an EFIT step
# plus the half millisecond the 1 kHz snap can add. MAST equilibria move slowly
# enough over 3 ms that selecting the nearest reconstruction is good enough.
TS_EQUILIBRIUM_TIME_TOL = 3e-3

# level 2 equilibrium signal -> standardized name.
# All 0D, interpolated onto the 1 kHz timebase.
EQUILIBRIUM_SIGNALS = {
    "wmhd": "energy_mhd",
    "beta_tor_normal": "beta_tor_norm",
    "minor_radius": "minor_radius",
    "elongation": "elongation",
    "triangularity_upper": "triangularity_upper",
    "triangularity_lower": "triangularity_lower",
    "geometric_axis_r": "geometric_axis_r",
}

# level 2 summary signal -> standardized name, same treatment
SUMMARY_SIGNALS = {
    "line_average_n_e": "n_e_line_average",
    "power_radiated": "power_radiated",
}

# GEQDSK 1D flux-function profiles: freeqdsk name -> level 2 equilibrium name.
# All are published on the uniform psi_norm grid the GEQDSK block wants.
# qpsi is absent here on purpose, level 2 only has q along the midplane
# (see _equilibrium_qpsi).
GEQDSK_PROFILES = {
    "fpol": "f",
    "pres": "pressure",
    "ffprime": "f_df_dpsi",
    "pprime": "dpressure_dpsi",
}

# Store paths a shot must carry to be worth staging. Everything else is derived.
# summary/power_nbi is required rather than zero filled: nothing in the archive
# can tell a shot whose beams were off from one whose beam record is missing.
REQUIRED_LEVEL2_PATHS = (
    "summary/ip",
    "summary/power_nbi",
    "equilibrium/psi",
    "equilibrium/psi_axis",
    "equilibrium/psi_boundary",
    "equilibrium/magnetic_axis_r",
    "equilibrium/magnetic_axis_z",
    "equilibrium/ip",
    "equilibrium/bvac_rmag",
    "equilibrium/li",
    "equilibrium/vloop_dynamic",
    *(f"summary/{name}" for name in SUMMARY_SIGNALS),
    *(f"equilibrium/{name}" for name in EQUILIBRIUM_SIGNALS),
)

# Per-variable attributes, IMAS data dictionary path under "ref"
SIGNAL_ATTRS = {
    "ip": {
        "description": "Measured plasma current magnitude",
        "units": "A",
        "ref": "/summary/global_quantities/ip/value",
    },
    "b0": {
        "description": "Vacuum toroidal field magnitude at geometric_axis_r",
        "units": "T",
        "ref": "/summary/global_quantities/b0/value",
    },
    "energy_mhd": {
        "description": "Total stored energy from the equilibrium reconstruction",
        "units": "J",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/energy_mhd",
    },
    "beta_tor_norm": {
        "description": "Normalized toroidal beta",
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/beta_tor_norm",
    },
    "n_e_line_average": {
        "description": "Line averaged electron density",
        "units": "m^-3",
        "ref": "/summary/line_average/n_e/value",
    },
    "power_ohm": {
        "description": "Ohmic heating power, Ip * (V_loop - L dIp/dt), clipped at 0",
        "units": "W",
        "ref": "/summary/global_quantities/power_ohm/value",
    },
    "power_radiated": {
        "description": "Bulk radiated power",
        "units": "W",
        "ref": "/summary/global_quantities/power_radiated_inside_lcfs/value",
    },
    "power_nbi": {
        "description": "Neutral beam heating power",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_nbi/value",
    },
    "power_ic": {
        "description": "Ion cyclotron heating power (none on MAST)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_ic/value",
    },
    "power_lh": {
        "description": "Lower hybrid heating power (none on MAST)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_lh/value",
    },
    "minor_radius": {
        "description": "Plasma minor radius",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/minor_radius",
    },
    "elongation": {
        "description": "Plasma elongation",
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/elongation",
    },
    "triangularity_upper": {
        "description": "Upper triangularity",
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_upper",
    },
    "triangularity_lower": {
        "description": "Lower triangularity",
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_lower",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/geometric_axis/r",
    },
    "ts_channel_r": {
        "description": (
            "Major radius of TS channel measurement locations, per slice: the "
            "ayc radial basis is re-derived for every laser pulse and moves by "
            "up to ~2 cm over a shot"
        ),
        "units": "m",
        "ref": "/thomson_scattering/channel(i1)/position/r",
    },
    "ts_channel_z": {
        "description": "Height of TS channel measurement locations",
        "units": "m",
        "ref": "/thomson_scattering/channel(i1)/position/z",
    },
    "ts_channel_n_e": {
        "description": "Electron density measured by TS channels",
        "units": "m^-3",
        "ref": "/thomson_scattering/channel(i1)/n_e/data",
    },
    "ts_channel_n_e_error": {
        "description": (
            "Electron density uncertainty from the ayc spectral fit, typically "
            "3-10% of the value"
        ),
        "units": "m^-3",
        "ref": "/thomson_scattering/channel(i1)/n_e/data_error_upper",
    },
    "ts_channel_t_e": {
        "description": "Electron temperature measured by TS channels",
        "units": "eV",
        "ref": "/thomson_scattering/channel(i1)/t_e/data",
    },
    "ts_channel_t_e_error": {
        "description": (
            "Electron temperature uncertainty from the ayc spectral fit, "
            "typically 5-20% of the value"
        ),
        "units": "eV",
        "ref": "/thomson_scattering/channel(i1)/t_e/data_error_upper",
    },
    "psirz": {
        "description": "Poloidal flux per radian on the reconstruction grid",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/profiles_2d(i1)/psi",
    },
    "simagx": {
        "description": "Poloidal flux at the magnetic axis",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/psi_axis",
    },
    "sibdry": {
        "description": "Poloidal flux at the plasma boundary",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/psi_boundary",
    },
    "rmagx": {
        "description": "Major radius of the magnetic axis",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/magnetic_axis/r",
    },
    "zmagx": {
        "description": "Height of the magnetic axis",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/magnetic_axis/z",
    },
    "current": {
        "description": "Plasma current from the equilibrium reconstruction, signed",
        "units": "A",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/ip",
    },
    "bcentr": {
        "description": (
            "Vacuum toroidal field at the GEQDSK reference radius rcentr, signed. "
            "The standardized b0 is the same field referenced at geometric_axis_r."
        ),
        "units": "T",
        "ref": "/equilibrium/vacuum_toroidal_field/b0",
    },
    "fpol": {
        "description": "Poloidal current function f = R*B_phi on the psi grid",
        "units": "m T",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/f",
    },
    "pres": {
        "description": "Total pressure on the psi grid",
        "units": "Pa",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/pressure",
    },
    "ffprime": {
        "description": "f df/dpsi on the psi grid",
        "units": "m^2 T^2 rad / Wb",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/f_df_dpsi",
    },
    "pprime": {
        "description": "dp/dpsi on the psi grid",
        "units": "Pa rad / Wb",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/dpressure_dpsi",
    },
    "qpsi": {
        "description": (
            "Safety factor on the psi grid, from the level 1 EFM reconstruction"
        ),
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/q",
    },
    "rbdry": {
        "description": "Major radius of the plasma boundary contour",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/outline/r",
    },
    "zbdry": {
        "description": "Height of the plasma boundary contour",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/outline/z",
    },
    "rlim": {
        "description": "Major radius of the limiter contour",
        "units": "m",
        "ref": "/wall/description_2d(i1)/limiter/unit(i2)/outline/r",
    },
    "zlim": {
        "description": "Height of the limiter contour",
        "units": "m",
        "ref": "/wall/description_2d(i1)/limiter/unit(i2)/outline/z",
    },
}


class MASTDataWorkflow(DataWorkflow):
    """MAST specific data workflow for creating and processing datasets."""

    signal_attrs = SIGNAL_ATTRS

    min_pulse_length = 0.2
    min_usable_time = 0.1
    min_segment_length = 0.1
    valid_filter = {
        # Only care about the magnitude of ip. These shots run 0.4-0.9 MA, so a
        # 1.5 MA reading is a broken record rather than a real current.
        "ip": {"min_abs": 210e3, "max_abs": 1.5e6},
        # Upper bound is the MAST density limit, 10.1088/1361-6587/ace476
        "n_e_line_average": {"min": 1e19, "max": 1.2e20},
        "energy_mhd": {"min": 10e3, "max": 2e6},
        "beta_tor_norm": {"min": 0.01, "max": 10.0},
        # Sample validity, distinct from the transient gate below: MAST total
        # input power tops out near 5 MW, so a recorded radiated power above
        # 4 MW is not a valid measurement.
        "power_radiated": {"min": 0.0, "max": 4e6},
    }
    # Both thresholds are empirical, and both gate the radiative or ohmic
    # collapse at the end of a shot rather than normal operation: the closest
    # ordinary approach found while porting these was shot 29153, whose ohmic
    # power peaks at 3.9 MW right before the end of the shot.
    transient_filter = {
        "power_radiated": 3.0e6,
        "power_ohm": 5.0e6,
    }
    # MAST's ip record runs through the current quench, so a small ip cutoff
    # still lets disruption transients in. Has to be longer than C-Mod's 20 ms.
    end_margin = 0.04
    # Shots the validity filtering rejects outright (checked 2026-08-20 by
    # running them through the pipeline), skipped here to save the processing.
    # 29430 is the worst of them: its bolometer is broken, the raw radiated
    # power spans -2.2 to +40.5 MW while MAST input power stays below 5 MW.
    # The rest of the "early campaign shots with known problems" list this
    # inherited from the older MAST workflow had no recorded reasons, passed
    # both the store probe and the filtering, and was dropped the same day.
    shot_blacklist = [
        28938,
        28976,
        28988,
        28995,
        29008,
        29430,
        30317,
        30318,
    ]

    # GP fit staging knobs. rho_max extends past the separatrix so the grid
    # covers the fit's edge value boundary conditions.
    fit_rho = np.linspace(0.0, 1.1, 64)
    fit_min_points = 10
    fit_scale_per_slice = True
    # Both variables share the same bounds on MAST:
    # - l1 floor 0.4: the chord's tangency point leaves many slices with no
    #   data inside rho ~0.4, and an l1 of 0.2 lets the fit collapse onto the
    #   zero prior there (core dives below the innermost channels, amplitude
    #   rails, fit_ignores_data culls the slice).
    # - x0 down to 0.85: edge-peaked ne (ears) has its structure at rho 0.85-0.95,
    #   out of reach of the short edge scale with the default 0.95 bound.
    # - var ceiling 5: on slices with an empty core the marginal likelihood
    #   rails the amplitude at the default ceiling of 20, which invents core
    #   values several times the slice max with a band to match.
    #   5 allows a prior amplitude of ~2x the slice max and
    #   leaves every data-covered region untouched.
    fit_bounds = {
        "te": FitBounds(l1_min=0.4, x0_min=0.85, var_max=5.0),
        "ne": FitBounds(l1_min=0.4, x0_min=0.85, var_max=5.0),
    }

    # The public S3 store tolerates concurrent reads, and every read is a
    # round trip, so staging scales almost linearly until the store throttles
    default_prepare_workers = 8

    def get_shotlist_from_source(self) -> list[int]:
        """Read the shotlist shipped with the package.

        MAST has no SQL database reachable outside Culham, so the shotlist has
        to be supplied explicitly.

        Returns:
            Shot numbers to process.

        Raises:
            FileNotFoundError: If the packaged shotlist file is missing.
        """
        if not DEFAULT_SHOTLIST_FILE.exists():
            raise FileNotFoundError(
                f"No MAST shotlist file at {DEFAULT_SHOTLIST_FILE}. Create a text "
                "file with one shot number per line, or pass shotlist_file."
            )
        with open(DEFAULT_SHOTLIST_FILE) as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from the MAST stores into standardized signals.

        The three sources are put on the shot's 1 kHz timebase differently:
        the 0D signals are interpolated (they are smooth), while the
        equilibrium reconstruction and the Thomson slices are snapped to the
        nearest grid time without interpolation, since neither is meaningful
        interpolated. The equilibrium is written as a full GEQDSK, so only the
        ~1 grid time in 5 that carries an EFIT slice has one (the rest are NaN,
        as are the trailing slots of the NaN-padded boundary contour).
        Thomson slices land on their own ~4.2 ms laser cadence, which is why
        the fit staging has to reach for a nearby reconstruction
        (see _equilibrium_at_ts_times).

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        data_tree = _open_level2(shot)
        if data_tree is None:
            if not _store_path_exists(f"{LEVEL2_PATH}/{shot}.zarr"):
                self.record_failed_shot(shot, "No level 2 store for this shot.")
            return None

        missing = [p for p in REQUIRED_LEVEL2_PATHS if not _has_path(data_tree, p)]
        if missing:
            reason = f"Missing level 2 signals: {', '.join(missing)}."
            logger.warning(f"Shot {shot}: {reason} Skipping.")
            self.record_failed_shot(shot, reason)
            return None

        summary = data_tree["summary"].ds
        equilibrium = data_tree["equilibrium"].ds

        summary_time = summary["time"].values
        ip = np.asarray(summary["ip"].values, dtype=float)
        in_shot = np.abs(ip) > SHOT_WINDOW_MIN_IP
        if in_shot.sum() < 2:
            reason = f"No plasma current above {SHOT_WINDOW_MIN_IP:.0f} A."
            logger.warning(f"Shot {shot}: {reason} Skipping.")
            self.record_failed_shot(shot, reason)
            return None
        timebase = make_uniform_1kHz_timebase(float(summary_time[in_shot][-1]))

        thomson = _open_level1_group(shot, LEVEL1_TS_GROUP)
        if thomson is None:
            if not _store_path_exists(f"{LEVEL1_PATH}/{shot}.zarr/{LEVEL1_TS_GROUP}"):
                self.record_failed_shot(
                    shot, f"No level 1 {LEVEL1_TS_GROUP} Thomson group for this shot."
                )
            return None
        missing_ts = [v for v in REQUIRED_TS_VARIABLES if v not in thomson]
        if missing_ts:
            reason = (
                f"Missing level 1 {LEVEL1_TS_GROUP} signals: {', '.join(missing_ts)}."
            )
            logger.warning(f"Shot {shot}: {reason} Skipping.")
            self.record_failed_shot(shot, reason)
            return None

        ds_thomson = _thomson_dataset(shot, thomson, timebase)
        if ds_thomson.sizes["idx"] == 0:
            reason = "No Thomson slices with usable data within the shot window."
            logger.warning(f"Shot {shot}: {reason} Skipping.")
            self.record_failed_shot(shot, reason)
            return None

        limiter = data_tree["wall"].ds if "wall" in data_tree else None
        ds_0d = _zero_d_dataset(shot, summary, equilibrium, timebase)
        ds_equilibrium = snap_to_grid(
            _equilibrium_dataset(shot, equilibrium, limiter), timebase
        )
        ds_thomson = snap_to_grid(ds_thomson, timebase)

        ds = xr.merge(
            [ds_0d, ds_equilibrium, ds_thomson], compat="no_conflicts", join="outer"
        )
        ds = ds.set_index(idx=["shot", "time"]).unstack("idx")
        ds.attrs = dict(ds_equilibrium.attrs)
        for name, attrs in SIGNAL_ATTRS.items():
            if name in ds:
                ds[name].attrs.update(attrs)
        return ds

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        1: Give every TS slice the nearest reconstruction to map through
        2: Map the TS channels onto rho through that flux map
        3: Convert to the fit units (Te [keV], ne [1e20 m^-3])
        4: Drop the channels outside the fittable rho range

        BOTH sides of the chord are fit. The inboard side maps onto the same
        rho through the reconstruction's interior flux, which a magnetics-only
        reconstruction does not pin precisely, and on a spherical tokamak Te
        is not strictly a flux function (poloidal asymmetries can be real).

        NOTE: The Thomson chord runs along z = TS_CHANNEL_Z while the MAST
        equilibria may put the magnetic axis 0.15-0.25 m lower, so the chord
        passes above the axis and never crosses the innermost flux surfaces.
        This may lead to extrapolation in the core.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ds_shot = _equilibrium_at_ts_times(ds.squeeze("shot", drop=True))
        ts_times, rho = map_ts_channels_to_rho(ds_shot)
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(ds_shot, ts_times)

        with np.errstate(invalid="ignore"):
            rho = np.where((rho >= 0.0) & (rho <= MAX_FIT_RHO), rho, np.nan)

        fit_input = ShotFitInput(
            x=rho, te_y=te_y, te_err=te_err, ne_y=ne_y, ne_err=ne_err, time=ts_times
        )
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no finite (rho, te, ne) channel data to fit")
            return None
        return fit_input


def _equilibrium_at_ts_times(ds_shot: xr.Dataset) -> xr.Dataset:
    """Put the nearest EFIT reconstruction on every grid time, for the mapping.

    The unprocessed dataset carries each reconstruction only at the grid time it
    was reconstructed on, which is the honest way to store it, but it leaves the
    Thomson slices with nothing to map through: EFIT is on a 5 ms grid and the
    laser fires every ~4.2 ms (see TS_EQUILIBRIUM_TIME_TOL). This copies the
    fields the mapping reads onto every grid time within that window of a real
    reconstruction, and only for the mapping, the stored dataset is untouched.

    Args:
        ds_shot: One shot's unprocessed dataset, with the shot dim squeezed out.

    Returns:
        The same dataset with the mapping's equilibrium fields nearest-filled,
        still NaN at grid times with no reconstruction inside the window.
    """
    has_equilibrium = np.isfinite(np.asarray(ds_shot["simagx"].values, dtype=float))
    if not has_equilibrium.any():
        return ds_shot

    grid_times = ds_shot["time"].values
    nearest = (
        ds_shot[list(TS_MAPPING_EQUILIBRIUM_FIELDS)]
        .isel(time=has_equilibrium)
        .reindex(time=grid_times, method="nearest", tolerance=TS_EQUILIBRIUM_TIME_TOL)
    )
    return ds_shot.assign(
        {name: nearest[name] for name in TS_MAPPING_EQUILIBRIUM_FIELDS}
    )


def _s3():
    """Open the anonymous filesystem the MAST stores are published on.

    Returns:
        The s3fs filesystem.
    """
    # Imported lazily: s3fs pulls in aiobotocore, which is slow to import and
    # useless to a run that only fits already staged shots.
    import s3fs

    return s3fs.S3FileSystem(anon=True, endpoint_url=S3_ENDPOINT)


def _store_path_exists(path: str) -> bool:
    """Check whether a path exists in the public MAST buckets.

    Separates a shot or group that was never published (a permanent failure
    worth recording) from an S3 hiccup (worth retrying on the next run).

    Args:
        path: Bucket-qualified path, e.g. "mast/level2/shots/30097.zarr".

    Returns:
        True if the path exists, False if it does not or cannot be listed.
    """
    try:
        return bool(_s3().ls(path, detail=False))
    except Exception:
        return False


def _open_level2(shot: int) -> xr.DataTree | None:
    """Open one shot's level 2 Zarr store.

    Args:
        shot: Shot number to open.

    Returns:
        The store's data tree, or None if it could not be opened.
    """
    import s3fs

    try:
        return xr.open_datatree(
            s3fs.S3Map(f"{LEVEL2_PATH}/{shot}.zarr", s3=_s3()),
            engine="zarr",
            chunks=None,
            consolidated=True,
        )
    except Exception as e:
        logger.warning(f"Shot {shot}: failed to open the level 2 store: {e}")
        logger.opt(exception=True).debug(e)
        return None


def _open_level1_group(shot: int, group: str) -> xr.Dataset | None:
    """Open one group of one shot's level 1 Zarr store.

    Args:
        shot: Shot number to open.
        group: Level 1 group name, e.g. "efm" or "ayc".

    Returns:
        The group, or None if it could not be opened. A group the shot never
        had and an S3 hiccup both land here, _store_path_exists tells them
        apart. Roughly 3% of the packaged shotlist has no ayc group, all of
        them shots the level 2 store also publishes no Thomson for.
    """
    import s3fs

    try:
        return xr.open_zarr(
            s3fs.S3Map(f"{LEVEL1_PATH}/{shot}.zarr", s3=_s3()),
            group=group,
            chunks=None,
            consolidated=True,
        )
    except Exception as e:
        logger.warning(f"Shot {shot}: failed to open level 1 {group}: {e}")
        logger.opt(exception=True).debug(e)
        return None


def _has_path(data_tree: xr.DataTree, path: str) -> bool:
    """Check whether a variable path exists in the store.

    Args:
        data_tree: Data tree of one shot's store.
        path: Group-qualified variable path, e.g. "summary/ip".

    Returns:
        True if the path resolves to a variable.
    """
    try:
        data_tree[path]
    except KeyError:
        return False
    return True


def _ohmic_power(
    summary_time: np.ndarray,
    ip: np.ndarray,
    eq_time: np.ndarray,
    li: np.ndarray,
    r_axis: np.ndarray,
    v_loop: np.ndarray,
    timebase: np.ndarray,
) -> np.ndarray:
    """Compute the ohmic heating power on the timebase.

    P_oh = Ip * (V_loop - L dIp/dt), with the internal inductance
    L = mu0 * R_axis * li / 2 and the dynamic LCFS loop voltage from the
    equilibrium. dIp/dt is smoothed with a causal boxcar first, since
    differentiating the raw current is noisy. Negative results are clipped to
    zero: they mean the inductive term overshot, not that the plasma is
    generating power.

    Args:
        summary_time: Timebase of the summary signals [s].
        ip: Plasma current on summary_time [A].
        eq_time: Timebase of the equilibrium signals [s].
        li: Internal inductance on eq_time.
        r_axis: Magnetic axis major radius on eq_time [m].
        v_loop: Dynamic LCFS loop voltage on eq_time [V].
        timebase: Uniform 1 kHz timebase to return the power on [s].

    Returns:
        Ohmic heating power on timebase [W].
    """
    dip_dt = np.gradient(ip, summary_time)
    if dip_dt.size >= OHMIC_SMOOTHING_SAMPLES:
        dip_dt = causal_boxcar_smooth(dip_dt, OHMIC_SMOOTHING_SAMPLES)
    inductance = 2.0e-7 * np.pi * r_axis * li  # mu0 * R * li / 2

    v_resistive = interp1(eq_time, v_loop, timebase) - interp1(
        eq_time, inductance, timebase
    ) * interp1(summary_time, dip_dt, timebase)
    return np.clip(interp1(summary_time, ip, timebase) * v_resistive, 0.0, None)


def _zero_d_dataset(
    shot: int,
    summary: xr.Dataset,
    equilibrium: xr.Dataset,
    timebase: np.ndarray,
) -> xr.Dataset:
    """Interpolate the 0D signals onto the timebase under standardized names.

    Plasma current and toroidal field are stored as magnitudes, the signed
    versions live in the equilibrium signals (see _equilibrium_dataset).

    Args:
        shot: Shot number being read.
        summary: The store's summary group.
        equilibrium: The store's equilibrium group.
        timebase: Uniform 1 kHz timebase [s].

    Returns:
        Dataset of 0D signals on dim "idx", with "time" and "shot" coords.
    """
    summary_time = summary["time"].values
    eq_time = equilibrium["time"].values

    data = {
        name: interp1(eq_time, equilibrium[source].values, timebase)
        for source, name in EQUILIBRIUM_SIGNALS.items()
    }
    data.update(
        {
            name: interp1(summary_time, summary[source].values, timebase)
            for source, name in SUMMARY_SIGNALS.items()
        }
    )

    ip = np.asarray(summary["ip"].values, dtype=float)
    data["ip"] = np.abs(interp1(summary_time, ip, timebase))
    # bvac_rmag is the vacuum field at the magnetic axis. Rescale it by 1/R to
    # the geometric axis, so b0 is referenced the same way as on the other
    # devices (C-Mod rout, D3D rsurf, TCV R_geom).
    data["b0"] = np.abs(
        interp1(eq_time, equilibrium["bvac_rmag"].values, timebase)
        * interp1(eq_time, equilibrium["magnetic_axis_r"].values, timebase)
        / data["geometric_axis_r"]
    )
    data["power_ohm"] = _ohmic_power(
        summary_time,
        ip,
        eq_time,
        equilibrium["li"].values,
        equilibrium["magnetic_axis_r"].values,
        equilibrium["vloop_dynamic"].values,
        timebase,
    )
    data["power_nbi"] = interp1(summary_time, summary["power_nbi"].values, timebase)
    # MAST has no ICRF or lower hybrid, zero where ip is valid
    data["power_ic"] = data["ip"] * 0.0
    data["power_lh"] = data["ip"] * 0.0

    return xr.Dataset(
        data_vars={name: ("idx", values) for name, values in data.items()},
        coords={
            "time": ("idx", timebase),
            "shot": ("idx", np.repeat(shot, timebase.size)),
        },
    )


def _time_first(data: xr.DataArray) -> np.ndarray:
    """Read a (dim, time) store variable as a (time, dim) float array.

    Args:
        data: Store variable with a "time" dimension.

    Returns:
        The values with time on axis 0.
    """
    other = [dim for dim in data.dims if dim != "time"]
    return np.asarray(data.transpose("time", *other).values, dtype=float)


def _optional_rows(
    shot: int, source: xr.Dataset, name: str, shape: tuple[int, int]
) -> np.ndarray:
    """Read a per-slice profile from the store, NaN-filled when it is absent.

    The GEQDSK extras are best-effort: a shot missing one of them is still
    worth staging for its Thomson profiles, it just cannot be exported as a
    complete GEQDSK.

    Args:
        shot: Shot number being read.
        source: Store group holding the variable.
        name: Variable name in that group.
        shape: (n_time, n_points) to fall back to.

    Returns:
        The (n_time, n_points) values, or an all-NaN array of that shape.
    """
    if name not in source:
        logger.warning(f"Shot {shot}: no {name} in the store, staging it as NaN")
        return np.full(shape, np.nan)
    return _time_first(source[name])


def _equilibrium_qpsi(shot: int, eq_time: np.ndarray, n_psi: int) -> np.ndarray:
    """Read the safety factor profile from the level 1 EFM reconstruction.

    The level 2 store's own "q" is q(R) along the midplane (its own label says
    "q(r) at z=0"), which is not the GEQDSK 1D block's q(psi), so this is the
    one GEQDSK field that has to come from level 1. Both stores publish the
    same EFM reconstruction (their psi_axis agrees bit for bit at shared
    times), so level 1 rows are matched to the level 2 slice times within
    EQUILIBRIUM_TIME_TOL rather than interpolated. Level 2 covers a longer
    window than level 1, and those extra slices come back NaN.

    Args:
        shot: Shot number being read.
        eq_time: Level 2 equilibrium slice times [s].
        n_psi: Points on the psi grid, to shape the fallback.

    Returns:
        The (n_time, n_psi) safety factor, NaN where level 1 has no slice.
    """
    qpsi = np.full((eq_time.size, n_psi), np.nan)
    efm = _open_level1_group(shot, LEVEL1_EFM_GROUP)
    if efm is None or "qpsi_c" not in efm:
        logger.warning(f"Shot {shot}: no level 1 qpsi_c, staging qpsi as NaN")
        return qpsi

    source_time = np.asarray(efm["time"].values, dtype=float)
    source_qpsi = _time_first(efm["qpsi_c"])
    if source_qpsi.shape[1] != n_psi:
        logger.warning(
            f"Shot {shot}: level 1 qpsi_c has {source_qpsi.shape[1]} psi points "
            f"against level 2's {n_psi}, staging qpsi as NaN"
        )
        return qpsi

    nearest = np.abs(source_time[:, None] - eq_time[None, :]).argmin(axis=0)
    matched = np.abs(source_time[nearest] - eq_time) <= EQUILIBRIUM_TIME_TOL
    qpsi[matched] = source_qpsi[nearest[matched]]
    return qpsi


def _equilibrium_dataset(
    shot: int, equilibrium: xr.Dataset, limiter: xr.Dataset | None
) -> xr.Dataset:
    """Build the full GEQDSK reconstruction, on the EFIT timebase.

    Everything comes from the level 2 equilibrium group except qpsi (see
    _equilibrium_qpsi) and the limiter contour (the store's wall group).
    Profiles the store is missing are staged as NaN rather than dropping the
    shot.

    Args:
        shot: Shot number being read.
        equilibrium: The store's equilibrium group.
        limiter: The store's wall group, or None when it has none.

    Returns:
        Dataset on dim "idx" with "time"/"shot" coords, in the freeqdsk
        canonical names (see machine.generic.make_geqdsk_dataset), carrying the
        COCOS number as an attribute.
    """
    eq_time = np.asarray(equilibrium["time"].values, dtype=float)
    n_psi = equilibrium.sizes["psi_norm"]
    # Named dimension transpose so the array layout matches the labels below
    psirz = equilibrium["psi"].transpose("time", "major_radius", "z").values
    current = np.asarray(equilibrium["ip"].values, dtype=float)
    r_grid = np.asarray(equilibrium["major_radius"].values, dtype=float)
    # GEQDSK pairs bcentr with rcentr: a reader reconstructing the vacuum field
    # as bcentr*rcentr/R has to land on fpol at the boundary. So scale the
    # published vacuum field (given at the magnetic axis) by 1/R onto the same
    # rcentr make_geqdsk_dataset writes into the file.
    bcentr = (
        np.asarray(equilibrium["bvac_rmag"].values, dtype=float)
        * np.asarray(equilibrium["magnetic_axis_r"].values, dtype=float)
        / r_grid[len(r_grid) // 2]
    )
    profiles = {
        name: _optional_rows(shot, equilibrium, source, (eq_time.size, n_psi))
        for name, source in GEQDSK_PROFILES.items()
    }
    n_boundary = equilibrium.sizes.get("n_boundary_coords", 1)
    boundary = {
        name: _optional_rows(shot, equilibrium, source, (eq_time.size, n_boundary))
        for name, source in (("rbdry", "lcfs_r"), ("zbdry", "lcfs_z"))
    }
    if limiter is None or "limiter_r" not in limiter:
        logger.warning(f"Shot {shot}: no limiter contour in the store")
        rlim = zlim = None
    else:
        rlim = np.asarray(limiter["limiter_r"].values, dtype=float)
        zlim = np.asarray(limiter["limiter_z"].values, dtype=float)

    return make_geqdsk_dataset(
        shot_id=shot,
        times=eq_time,
        r_grid=r_grid,
        z_grid=equilibrium["z"].values,
        rmagx=np.asarray(equilibrium["magnetic_axis_r"].values, dtype=float),
        zmagx=np.asarray(equilibrium["magnetic_axis_z"].values, dtype=float),
        simagx=np.asarray(equilibrium["psi_axis"].values, dtype=float),
        sibdry=np.asarray(equilibrium["psi_boundary"].values, dtype=float),
        bcentr=bcentr,
        current=current,
        qpsi=_equilibrium_qpsi(shot, eq_time, n_psi),
        psirz=psirz,
        cocos_input=efit_cocos_from_signs(current, bcentr),
        rlim=rlim,
        zlim=zlim,
        **profiles,
        **boundary,
    )


def _thomson_dataset(
    shot: int, thomson: xr.Dataset, timebase: np.ndarray
) -> xr.Dataset:
    """Collect the Thomson channel data on its native timebase.

    The source is the level 1 ayc group, not the level 2 thomson_scattering
    group. Level 2 is the same measurement bilinearly interpolated onto a
    uniform 1 cm major radius grid and a uniform 5 ms timebase, and published
    without uncertainties. That regrid is what makes it the wrong input here:
    the laser fires every ~4.2 ms, so a level 2 sample blends two pulses, and
    the interpolated radial points are not independent measurements even though
    the fit would weight them as if they were. Level 1 instead gives the 131
    real channels at the real laser times, with the spectral fit's own te and
    ne uncertainties and the per-slice radial basis (ayc re-derives it every
    pulse, and it moves by up to ~2 cm over a shot).

    A channel is kept only where the value and its error are both finite and
    positive, since the fit needs both. Slices left with no usable channel at
    all are dropped: some shots publish every other slice empty, radial basis
    included (shot 30097, for one).

    Args:
        shot: Shot number being read.
        thomson: The shot's level 1 ayc group.
        timebase: Uniform 1 kHz timebase, used to drop TS slices outside the
            shot window [s].

    Returns:
        Dataset on dims ("idx", "ts_channel") with "time"/"shot" coords.
    """
    ts_time = np.asarray(thomson["time"].values, dtype=float)
    in_shot = (ts_time >= timebase[0]) & (ts_time <= timebase[-1])
    ts_time = ts_time[in_shot]

    channel_data = {}
    for source, name in (("te", "ts_channel_t_e"), ("ne", "ts_channel_n_e")):
        values = _time_first(thomson[source])[in_shot]
        errors = _time_first(thomson[f"{source}_error"])[in_shot]
        with np.errstate(invalid="ignore"):
            usable = (values > 0) & (errors > 0)
        channel_data[name] = np.where(usable, values, np.nan)
        channel_data[f"{name}_error"] = np.where(usable, errors, np.nan)

    r_channel = _time_first(thomson["radius"])[in_shot]
    has_data = np.isfinite(channel_data["ts_channel_t_e"]) | np.isfinite(
        channel_data["ts_channel_n_e"]
    )
    keep = has_data.any(axis=1)
    n_empty = int((~keep).sum())
    if n_empty:
        logger.debug(f"Shot {shot}: dropping {n_empty} empty Thomson slices")

    return xr.Dataset(
        data_vars={
            "ts_channel_r": (("idx", "ts_channel"), r_channel[keep]),
            "ts_channel_z": (
                ("idx", "ts_channel"),
                np.full(r_channel[keep].shape, TS_CHANNEL_Z),
            ),
            **{
                name: (("idx", "ts_channel"), values[keep])
                for name, values in channel_data.items()
            },
        },
        coords={
            "time": ("idx", ts_time[keep]),
            "shot": ("idx", np.repeat(shot, int(keep.sum()))),
            "ts_channel": np.arange(r_channel.shape[1]),
        },
    )
