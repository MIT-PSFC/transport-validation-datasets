"""MAST data workflow, built from the open-access Zarr stores on STFC ECHO S3.

Most of it comes out of the level 2 store: https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr
the 0D summary and equilibrium signals and the full GEQDSK reconstruction, whose flux map also places
the Thomson channels in rho_tor_norm.
Two things come from the level 1 store instead:
the GEQDSK safety factor, which only level 1 publishes as a flux function (see equilibrium_qpsi),
and the Thomson profiles, which level 2 only carries interpolated onto a uniform (R, t) grid and without uncertainties (see _thomson_dataset).

No MDSplus is involved, so this workflow runs anywhere with internet access.
Reads are slow, so staging runs in a thread pool of prepare_workers threads.
One shot costs ~30-40 s of round trips.
The packaged shotlist is built by machine/mast/shotlist.py
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from disruption_py.core.utils.math import causal_boxcar_smooth, interp1
from loguru import logger

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.cleaning import drop_in_both
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.generic import (
    channel_rows_at_times,
    efit_cocos_from_signs,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho_tor_norm,
    snap_to_grid,
    ts_channel_fit_rows,
)
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# Public MAST open data, no credentials needed
S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"
LEVEL2_PATH = "mast/level2/shots"

# The raw diagnostic output. Two groups are read:
# EFM for the one GEQDSK field the level 2 store does not carry as a flux function (qpsi, see equilibrium_qpsi)
# AYC for the Thomson profiles (see _thomson_dataset)
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

# Shotlist for the M7-M9 campaigns, built by shotlist.py
DEFAULT_SHOTLIST_FILE = PACKAGE_ROOT / "machine" / "mast" / "mast_shotlist_M7_M9"

# The core Thomson system (AYC) views along a horizontal chord at the midplane,
# so every channel is at the same height and only its major radius varies.
TS_CHANNEL_Z = 0.0

# Plasma current magnitude that marks the end of the shot window [A].
# Only used to find the last time worth putting on the timebase,
# the real current cut is valid_filter["ip"].
SHOT_WINDOW_MIN_IP = 100e3

# Samples of the causal boxcar smoothing dIp/dt in the ohmic power calculation
OHMIC_SMOOTHING_SAMPLES = 10

# Channels this far outside the separatrix sit in the far SOL,
# where mapping through a magnetics-only reconstruction is not trustworthy.
MAX_FIT_RHO_TOR_NORM = 1.1

# Sometimes MAST measurements have huge errors, way larger than their values.
# If this ever happens, drop that point.
MAX_RELATIVE_ERROR = 1.0

# Kinetic profiles in spherical tokamaks are not necessarily flux functions,
# and as such the inboard and outboard side can disagree.
# Where they do, each channel's error gets half the local disagreement added,
# so both branches are consistent with a profile between them.
# The disagreement at a channel is its value minus the other branch interpolated to its rho,
# only between two channels of the other branch at most BRANCH_MAX_GAP apart.
# Each channel takes the median |disagreement| of the channels of both branches within BRANCH_SMOOTH_HALFWIDTH
# (at least BRANCH_MIN_CHANNELS of them), which keeps one spike from inflating its neighbours.
# Pooling both branches inflates channels at the same rho alike.
# Per branch, the sparse outboard edge (channels ~0.07 apart) had too few estimates and kept its raw errors,
# while the dense inboard pedestal was inflated, which steered the fit onto the sparse branch (24403 t=0.342).
# A channel past the overlap, with too few estimates in its window,
# takes the disagreement of the nearest channel that has one, up to BRANCH_CARRY_DISTANCE away,
# as a fraction of the value, scaled to its own value.
# Otherwise the last channel of the longer branch keeps its raw error and pins the fit
# (24403 t=0.342: outboard ne 0.19 +- 0.006 at rho 1.055, 0.087 past the last inboard channel,
# held the fit 0.2 above the inboard pedestal).
# Carried as an absolute value, a disagreement from the steep pedestal swamped the channels of its foot
# (29632 t=0.212: 0.07 keV added to inboard Te of 2-55 eV, and the fit floated to 40 eV at the separatrix, against 3 eV).
BRANCH_MAX_GAP = 0.08
BRANCH_SMOOTH_HALFWIDTH = 0.05
BRANCH_MIN_CHANNELS = 3
BRANCH_CARRY_DISTANCE = 0.15

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
GEQDSK_PROFILES = {
    "fpol": "f",
    "pres": "pressure",
    "ffprime": "f_df_dpsi",
    "pprime": "dpressure_dpsi",
}

# Level 2 groups and the signals in them a shot must carry to be worth staging.
# Everything else is derived.
# summary/power_nbi is required rather than zero filled: nothing in the archive
# can tell a shot whose beams were off from one whose beam record is missing.
REQUIRED_LEVEL2_SIGNALS = {
    "summary": ("ip", "power_nbi", *SUMMARY_SIGNALS),
    "equilibrium": (
        "psi",
        "psi_axis",
        "psi_boundary",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "ip",
        "bvac_rmag",
        "li",
        "vloop_dynamic",
        *EQUILIBRIUM_SIGNALS,
    ),
}

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


@dataclass(frozen=True)
class MASTSettings(DeviceSettings):
    """MAST settings, the [mast] table of the config file."""


class MissingSourceError(Exception):
    """A shot lacks data the workflow reads, so a retry will not help.

    The message says what is missing.
    """


@dataclass(frozen=True)
class ShotSources:
    """One shot's stores, checked for everything get_source_dataset reads.

    Attributes:
        summary: The level 2 summary group.
        equilibrium: The level 2 equilibrium group.
        ds_thomson: The usable Thomson slices inside the shot window,
            on their own timebase (see _thomson_dataset).
        timebase: The shot's uniform 1 kHz timebase [s].
    """

    summary: xr.Dataset
    equilibrium: xr.Dataset
    ds_thomson: xr.Dataset
    timebase: np.ndarray


class MASTDataWorkflow(DataWorkflow):
    """MAST specific data workflow for creating and processing datasets."""

    settings_cls = MASTSettings
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

    # GP fit staging knobs.
    # A 1/60 step puts 1.0 and 1.1 on the grid.
    fit_rho_tor_norm = np.linspace(0.0, 1.6, 97)
    fit_min_points = 10
    fit_scale_per_slice = True
    # Both variables share the same bounds on MAST:
    # - l1 floor 0.2: MAST cores carry structure a longer scale smooths away,
    #   e.g. a hollow Te in the current ramp or a flat core with a knee at rho ~0.45
    # - var ceiling 5: on slices with an empty core the marginal likelihood
    #   rails the amplitude at the default ceiling of 20, which invents core
    #   values several times the slice max with a band to match.
    #   5 allows a prior amplitude of ~2x the slice max and
    #   leaves every data-covered region untouched.
    fit_bounds = {
        "te": FitBounds(l1_min=0.2, var_max=5.0),
        "ne": FitBounds(l1_min=0.2, var_max=5.0),
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
        Thomson slices land on their own ~4.2 ms laser cadence,
        so the fit staging maps each through the nearest reconstruction.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        try:
            sources = open_shot_sources(shot)
        except MissingSourceError as e:
            logger.warning(f"Shot {shot}: {e} Skipping.")
            self.record_failed_shot(shot, str(e))
            return None
        if sources is None:
            return None

        limiter = _open_store_group(LEVEL2_PATH, shot, "wall")
        if limiter is None and _store_path_exists(f"{LEVEL2_PATH}/{shot}.zarr/wall"):
            return None

        timebase = sources.timebase
        ds_0d = _zero_d_dataset(shot, sources.summary, sources.equilibrium, timebase)
        ds_equilibrium = snap_to_grid(
            _equilibrium_dataset(shot, sources.equilibrium, limiter), timebase
        )
        ds_thomson = snap_to_grid(sources.ds_thomson, timebase)

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

        1: Map the TS channels onto rho_tor_norm through the nearest reconstruction
        2: Convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: Drop every channel whose Te or ne error exceeds MAX_RELATIVE_ERROR times its value
        4: Inflate the errors where the inboard and outboard branches disagree (_branch_disagreement_errors)
        5: Drop the channels outside the fittable range

        BOTH sides of the chord are fit. The inboard side maps onto the same
        rho_tor_norm through the reconstruction's interior flux, which a magnetics-only
        reconstruction does not pin precisely, and on a spherical tokamak Te
        is not strictly a flux function (poloidal asymmetries can be real).

        NOTE: The Thomson chord runs along z = TS_CHANNEL_Z while the MAST
        equilibria may put the magnetic axis 0.15-0.25 m lower, so the chord
        passes above the axis and never crosses the innermost flux surfaces.
        This will lead to extrapolation and extremely poor fits in the core.
        The packaged shotlist keeps only shots whose chord passes near the axis.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ds_shot = ds.squeeze("shot", drop=True)
        ts_times, rho_tor_norm = map_ts_channels_to_rho_tor_norm(
            ds_shot, self.settings.sol_extension
        )
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(ds_shot, ts_times)

        # The read keeps only positive values with positive errors, see _thomson_dataset
        with np.errstate(invalid="ignore"):
            te_huge = te_err > MAX_RELATIVE_ERROR * te_y
            ne_huge = ne_err > MAX_RELATIVE_ERROR * ne_y
        te_y, ne_y, _, _ = drop_in_both(te_y, ne_y, te_huge, ne_huge)
        n_huge = int((te_huge | ne_huge).sum())
        if n_huge:
            logger.info(
                f"Shot {shot}: dropped {n_huge} channel readings with an error over "
                f"{MAX_RELATIVE_ERROR:g}x the value"
            )

        r_channel = channel_rows_at_times(ds_shot["ts_channel_r"], ts_times)
        inboard = _inboard_channels(rho_tor_norm, r_channel)
        te_err = _branch_disagreement_errors(rho_tor_norm, te_y, te_err, inboard)
        ne_err = _branch_disagreement_errors(rho_tor_norm, ne_y, ne_err, inboard)

        with np.errstate(invalid="ignore"):
            rho_tor_norm = np.where(
                rho_tor_norm <= MAX_FIT_RHO_TOR_NORM, rho_tor_norm, np.nan
            )

        fit_input = ShotFitInput(
            x=rho_tor_norm,
            te_y=te_y,
            te_err=te_err,
            ne_y=ne_y,
            ne_err=ne_err,
            time=ts_times,
        )
        if not fit_input.has_fittable_points():
            logger.warning(
                f"Shot {shot}: no finite (rho_tor_norm, te, ne) channel data to fit"
            )
            return None
        return fit_input

    def fit_plot_channel_groups(
        self, shot: int, fit_input: ShotFitInput
    ) -> list | None:
        """Split the fit-plot channels into the inboard and outboard branches of the chord, row by row.

        The rows split at their lowest-rho channel, as in the branch error inflation (_inboard_channels).
        Each channel sits at its median major radius over the shot.
        That keeps the channels' order along the chord, and serves pooled window rows as well,
        whose columns repeat the channels once per pooled sample.

        Args:
            shot: Shot number being plotted.
            fit_input: The shot's staged fit input.

        Returns:
            (mask, color, label) triples, each mask (n_rows, n_columns).
        """
        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds:
            r_rows = (
                ds["ts_channel_r"]
                .squeeze("shot", drop=True)
                .transpose("time", "ts_channel")
                .values
            )
        has_r = np.isfinite(r_rows).any(axis=0)
        r_channel = np.full(r_rows.shape[1], np.nan)
        r_channel[has_r] = np.nanmedian(r_rows[:, has_r], axis=0)
        n_rows, n_columns = fit_input.x.shape
        r_tiled = np.tile(r_channel, (n_rows, n_columns // r_channel.size))
        inboard = _inboard_channels(fit_input.x, r_tiled)
        outboard = np.isfinite(fit_input.x) & ~inboard
        return [
            (inboard, "tab:blue", "inboard TS"),
            (outboard, "tab:orange", "outboard TS"),
        ]


def _inboard_channels(rho_tor_norm: np.ndarray, r_channel: np.ndarray) -> np.ndarray:
    """Mark the channels on the inboard branch of the chord, slice by slice.

    The branches split at the channel of lowest rho_tor_norm,
    where the chord passes closest to the magnetic axis of the same reconstruction the channels were mapped through.

    Args:
        rho_tor_norm: (n_t, n_ch) channel positions, NaN where unmapped.
        r_channel: (n_t, n_ch) channel major radii [m].

    Returns:
        (n_t, n_ch) mask of the inboard channels, False in slices with no mapped channel.
    """
    mapped = np.isfinite(rho_tor_norm).any(axis=1)
    rho_filled = np.where(np.isfinite(rho_tor_norm), rho_tor_norm, np.inf)
    i_axis = np.argmin(rho_filled, axis=1)
    r_axis = np.take_along_axis(r_channel, i_axis[:, None], axis=1)
    return mapped[:, None] & (r_channel < r_axis)


def _branch_disagreement_errors(
    rho_tor_norm: np.ndarray, y: np.ndarray, err: np.ndarray, inboard: np.ndarray
) -> np.ndarray:
    """Inflate the errors by half the local disagreement between the inboard and outboard branches.

    See the BRANCH_* constants for the calibration.
    A channel further than BRANCH_CARRY_DISTANCE from any disagreement estimate keeps its error.

    Args:
        rho_tor_norm: (n_t, n_ch) channel positions, NaN where unmapped.
        y: (n_t, n_ch) channel values, NaN where invalid.
        err: (n_t, n_ch) channel errors.
        inboard: (n_t, n_ch) inboard branch mask (_inboard_channels).

    Returns:
        The (n_t, n_ch) inflated errors.
    """
    err_out = np.array(err, dtype=float)
    for i_time in range(y.shape[0]):
        rho = rho_tor_norm[i_time]
        valid = np.isfinite(rho) & np.isfinite(y[i_time])
        is_inboard = inboard[i_time]
        delta = np.full(rho.shape, np.nan)
        for side in (True, False):
            this = np.flatnonzero(valid & (is_inboard == side))
            other = valid & (is_inboard != side)
            if other.sum() < 2 or this.size == 0:
                continue
            order = np.argsort(rho[other])
            rho_other = rho[other][order]
            y_other = y[i_time][other][order]
            right = np.searchsorted(rho_other, rho[this])
            right_clipped = np.clip(right, 1, rho_other.size - 1)
            gap = rho_other[right_clipped] - rho_other[right_clipped - 1]
            bracketed = (right > 0) & (right < rho_other.size) & (gap <= BRANCH_MAX_GAP)
            y_other_at_this = np.interp(rho[this], rho_other, y_other)
            delta_this = y[i_time][this] - y_other_at_this
            delta[this[bracketed]] = delta_this[bracketed]
        abs_delta = np.abs(delta)
        has_delta = np.isfinite(abs_delta)
        disagreement = np.full(rho.shape, np.nan)
        for k in np.flatnonzero(valid):
            near = np.abs(rho - rho[k]) <= BRANCH_SMOOTH_HALFWIDTH
            window = has_delta & near
            if window.sum() >= BRANCH_MIN_CHANNELS:
                disagreement[k] = np.median(abs_delta[window])
        # Channels past the overlap take the nearest estimate relative to the value,
        # a carried value is never carried on
        has_estimate = np.flatnonzero(np.isfinite(disagreement))
        if has_estimate.size:
            relative_disagreement = disagreement[has_estimate] / np.abs(
                y[i_time][has_estimate]
            )
            for k in np.flatnonzero(valid & ~np.isfinite(disagreement)):
                distance = np.abs(rho[has_estimate] - rho[k])
                i_nearest = int(np.argmin(distance))
                if distance[i_nearest] <= BRANCH_CARRY_DISTANCE:
                    y_k = np.abs(y[i_time, k])
                    disagreement[k] = relative_disagreement[i_nearest] * y_k
        inflate = np.isfinite(disagreement)
        err_out[i_time, inflate] = np.hypot(
            err[i_time, inflate], 0.5 * disagreement[inflate]
        )
    return err_out


def open_shot_sources(shot: int) -> ShotSources | None:
    """Open one shot's stores and check they carry everything get_source_dataset reads.

    The shot window ends at the last plasma current above SHOT_WINDOW_MIN_IP.

    The Thomson group is checked first, it is the cheapest to open
    and shots from before AYC was installed have none.

    Args:
        shot: Shot number to open.

    Returns:
        The opened sources, or None when a store could not be reached,
        which is worth retrying on the next run.

    Raises:
        MissingSourceError: If the shot has no store, a required signal,
            plasma current, or usable Thomson slice.
    """
    thomson = _open_store_group(LEVEL1_PATH, shot, LEVEL1_TS_GROUP)
    if thomson is None:
        if _store_path_exists(f"{LEVEL1_PATH}/{shot}.zarr/{LEVEL1_TS_GROUP}"):
            return None
        raise MissingSourceError(
            f"No level 1 {LEVEL1_TS_GROUP} Thomson group for this shot."
        )
    missing_ts = [v for v in REQUIRED_TS_VARIABLES if v not in thomson]
    if missing_ts:
        raise MissingSourceError(
            f"Missing level 1 {LEVEL1_TS_GROUP} signals: {', '.join(missing_ts)}."
        )

    level2 = {}
    missing = []
    for group, names in REQUIRED_LEVEL2_SIGNALS.items():
        ds_group = _open_store_group(LEVEL2_PATH, shot, group)
        if ds_group is None:
            if _store_path_exists(f"{LEVEL2_PATH}/{shot}.zarr/{group}"):
                return None
            raise MissingSourceError(f"No level 2 {group} group for this shot.")
        missing += [f"{group}/{name}" for name in names if name not in ds_group]
        level2[group] = ds_group
    if missing:
        raise MissingSourceError(f"Missing level 2 signals: {', '.join(missing)}.")

    summary = level2["summary"]
    summary_time = summary["time"].values
    ip = np.asarray(summary["ip"].values, dtype=float)
    in_shot = np.abs(ip) > SHOT_WINDOW_MIN_IP
    if in_shot.sum() < 2:
        raise MissingSourceError(f"No plasma current above {SHOT_WINDOW_MIN_IP:.0f} A.")
    timebase = make_uniform_1kHz_timebase(float(summary_time[in_shot][-1]))

    ds_thomson = _thomson_dataset(shot, thomson, timebase)
    if ds_thomson.sizes["idx"] == 0:
        raise MissingSourceError(
            "No Thomson slices with usable data within the shot window."
        )
    return ShotSources(
        summary=summary,
        equilibrium=level2["equilibrium"],
        ds_thomson=ds_thomson,
        timebase=timebase,
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


def _open_store_group(store_path: str, shot: int, group: str) -> xr.Dataset | None:
    """Open one group of one shot's level 1 or level 2 Zarr store.

    Groups are opened one at a time because opening a store loads the coordinates of every group in it.
    A whole level 2 store takes ~65 s that way, its summary and equilibrium groups ~8 s.
    Level 2 is Zarr v3 with its consolidated metadata only at the root, so its groups open through the root.
    Level 1 is Zarr v2 with consolidated metadata in every group, so its groups open from their own path.
    Through the root some level 1 groups do not open at all (every group of shot 24448, for one).

    Args:
        store_path: LEVEL1_PATH or LEVEL2_PATH.
        shot: Shot number to open.
        group: Group name, e.g. "equilibrium" or "ayc".

    Returns:
        The group, or None if it could not be opened.
        A group the shot never had and an S3 hiccup both land here,
        _store_path_exists tells them apart.
    """
    import s3fs

    if store_path == LEVEL1_PATH:
        mapper = s3fs.S3Map(f"{store_path}/{shot}.zarr/{group}", s3=_s3())
        group_in_mapper = None
    else:
        mapper = s3fs.S3Map(f"{store_path}/{shot}.zarr", s3=_s3())
        group_in_mapper = group
    try:
        return xr.open_zarr(
            mapper, group=group_in_mapper, chunks=None, consolidated=True
        )
    except Exception as e:
        logger.warning(f"Shot {shot}: failed to open {store_path} {group}: {e}")
        logger.opt(exception=True).debug(e)
        return None


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


def equilibrium_qpsi(shot: int, eq_time: np.ndarray, n_psi: int) -> np.ndarray:
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
    efm = _open_store_group(LEVEL1_PATH, shot, LEVEL1_EFM_GROUP)
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
    equilibrium_qpsi) and the limiter contour (the store's wall group).
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
    # MAST publishes no RCENTR, so the grid midpoint serves as the reference radius.
    # A reader rebuilds the vacuum field as bcentr*rcentr/R, which has to land on fpol at the boundary.
    # So the published vacuum field, given at the magnetic axis, is rescaled by 1/R onto rcentr.
    rcentr = r_grid[len(r_grid) // 2]
    bcentr = (
        np.asarray(equilibrium["bvac_rmag"].values, dtype=float)
        * np.asarray(equilibrium["magnetic_axis_r"].values, dtype=float)
        / rcentr
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
        qpsi=equilibrium_qpsi(shot, eq_time, n_psi),
        psirz=psirz,
        cocos_input=efit_cocos_from_signs(current, bcentr),
        rcentr=rcentr,
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
