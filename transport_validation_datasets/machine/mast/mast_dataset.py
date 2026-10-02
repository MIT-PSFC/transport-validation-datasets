"""MAST data workflow, built from the open-access Zarr stores on STFC ECHO S3.

Most of it comes out of the level 1 store: https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr
- efm: the EFIT reconstruction, as the full GEQDSK block and the 0D equilibrium signals.
  Its flux map also places the Thomson channels in rho_tor_norm.
- esm: the ohmic power.
- ayc: the Thomson profiles, see _thomson_dataset.
The level 2 store (https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr)
only supplies the summary signals: ip, power_nbi, n_e_line_average and power_radiated.

No MDSplus is involved, so this workflow runs anywhere with internet access.
Reads are slow, so staging runs in a thread pool of prepare_workers threads.
One shot costs ~30-40 s of round trips.
The packaged shotlist is built by machine/mast/shotlist.py
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from loguru import logger

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.cleaning import drop_in_both
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.generic import (
    channel_rows_at_times,
    cocos_from_signs,
    held_signal_on_grid,
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

# The raw diagnostic output, see the module docstring for the groups read
LEVEL1_PATH = "mast/level1/shots"

# Shotlist for the M7-M9 campaigns, built by shotlist.py
DEFAULT_SHOTLIST_FILE = PACKAGE_ROOT / "machine" / "mast" / "mast_shotlist_M7_M9"

# The core Thomson system (AYC) views along a horizontal chord at the midplane,
# so every channel is at the same height and only its major radius varies.
TS_CHANNEL_Z = 0.0

# Plasma current magnitude that marks the end of the shot window [A].
# Only used to find the last time worth putting on the timebase,
# the real current cut is min_filter["ip"].
SHOT_WINDOW_MIN_IP = 100e3

# Channels this far outside the separatrix sit in the far SOL,
# where mapping through a magnetics-only reconstruction is not trustworthy.
MAX_FIT_RHO_TOR_NORM = 1.1

# Sometimes MAST measurements have huge errors, way larger than their values.
# If this ever happens, drop that point.
MAX_RELATIVE_ERROR = 1.0

# Past this rho the inboard branch departs from the outboard one systematically, not as scatter.
# Over the 100 shots of tuning iteration 5 the inboard read high by a median of
# 20 percent in Te and 7 percent in ne at rho 0.8-0.85, and 42 and 18 percent at 0.85-0.9,
# while its own point-to-point scatter stayed below the outboard's.
# It has 3-5 times the outboard's channels there and steered the edge fit, so it is dropped.
MAX_INBOARD_RHO_TOR_NORM = 0.8

# Kinetic profiles in spherical tokamaks are not necessarily flux functions,
# and as such the inboard and outboard side can disagree.
# Where they do, each channel's error gets half the local disagreement added,
# so both branches are consistent with a profile between them.
# The disagreement at a channel is its value minus the other branch interpolated to its rho,
# only between two channels of the other branch at most BRANCH_MAX_GAP apart.
# Each channel takes the median |disagreement| of the channels of both branches within BRANCH_SMOOTH_HALFWIDTH
# (at least BRANCH_MIN_CHANNELS of them), which keeps one spike from inflating its neighbours.
# Pooling both branches inflates channels at the same rho alike,
# where per branch the sparser one could keep its raw errors and steer the fit (24403 t=0.342).
# Channels outside the overlap keep their raw errors.
# Past MAX_INBOARD_RHO_TOR_NORM only the outboard branch is left, with nothing to disagree with.
BRANCH_MAX_GAP = 0.08
BRANCH_SMOOTH_HALFWIDTH = 0.05
BRANCH_MIN_CHANNELS = 3

# efm signal -> standardized name.
# All 0D, interpolated onto the 1 kHz timebase.
EQUILIBRIUM_SIGNALS = {
    # plasma_energy (EFM_PLASMA_ENERGY) is 3/2 the volume integral of the reconstructed pressure.
    # Not wplasmd (EFM_WPLASMD), the diamagnetic energy, built on a measured diamagnetic flux that is 0 in level 1
    "plasma_energy": "energy_mhd",
    "betan": "beta_tor_norm",
    "minor_radius": "minor_radius",
    "elongation": "elongation",
    "triang_upper": "triangularity_upper",
    "triang_lower": "triangularity_lower",
    "geom_axis_rc": "geometric_axis_r",
}

# level 2 summary signal -> standardized name, same treatment
SUMMARY_SIGNALS = {
    "line_average_n_e": "n_e_line_average",
    "power_radiated": "power_radiated",
}

# GEQDSK 1D flux-function profiles: freeqdsk name -> efm name.
# All are published on the uniform psi_norm grid the GEQDSK block wants.
GEQDSK_PROFILES = {
    "fpol": "fpsi_c",
    "pres": "ppsi_c",
    "ffprime": "ffprime",
    "pprime": "pprime",
    "qpsi": "qpsi_c",
}

# Groups and the signals in them a shot must carry to be worth staging, in the order they are opened.
# ayc goes first, it is the cheapest to open and shots from before AYC was installed have none.
# Everything else is derived, or optional like the GEQDSK profiles and the limiter.
REQUIRED_LEVEL1_SIGNALS = {
    "ayc": ("radius", "te", "te_error", "ne", "ne_error"),
    "efm": (
        "psirz",
        "gridr",
        "gridz",
        "psi_axis",
        "psi_boundary",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "plasma_current_c",
        "bvac_rmag",
        "bvac_r",
        "bvac_val",
        *EQUILIBRIUM_SIGNALS,
    ),
    "esm": ("pphix",),
}
# summary/power_nbi is required rather than zero filled: nothing in the archive
# can tell a shot whose beams were off from one whose beam record is missing.
REQUIRED_LEVEL2_SIGNALS = {
    "summary": ("ip", "power_nbi", *SUMMARY_SIGNALS),
}

# Per-variable attributes
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
        "description": "Stored energy from the equilibrium reconstruction, 3/2 the volume integral of its pressure",
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
        "description": (
            "Ohmic heating power, Ip * V_loop at the LCFS minus the rate of change of the stored poloidal magnetic energy "
            "(ESM_PPHIX), clipped at 0"
        ),
        "units": "W",
        "ref": "/summary/global_quantities/power_ohm/value",
    },
    "power_radiated": {
        "description": "Total radiated power from the poloidal bolometer array (ABM_PRAD_POL)",
        "units": "W",
        "ref": "/summary/global_quantities/power_radiated/value",
    },
    "power_nbi": {
        "description": "Neutral beam power injected into the vessel (ANB_TOT_SUM_POWER)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_launched_nbi/value",
    },
    "power_ic": {
        "description": "Ion cyclotron heating power (none on MAST)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_launched_ic/value",
    },
    "power_lh": {
        "description": "Lower hybrid heating power (none on MAST)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_launched_lh/value",
    },
    "power_ec": {
        "description": "Electron cyclotron heating power (none on MAST)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_launched_ec/value",
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
    "bcentr": {
        "description": (
            "Vacuum toroidal field at the GEQDSK reference radius rcentr, signed. "
            "The standardized b0 is the same field referenced at geometric_axis_r."
        ),
        "units": "T",
        "ref": "/equilibrium/vacuum_toroidal_field/b0",
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
        efm: The level 1 efm group, the equilibrium reconstruction.
        esm: The level 1 esm group, for the ohmic power.
        ds_thomson: The usable Thomson slices inside the shot window,
            on their own timebase (see _thomson_dataset).
        timebase: The shot's uniform 1 kHz timebase [s].
    """

    summary: xr.Dataset
    efm: xr.Dataset
    esm: xr.Dataset
    ds_thomson: xr.Dataset
    timebase: np.ndarray


class MASTDataWorkflow(DataWorkflow):
    """MAST specific data workflow for creating and processing datasets."""

    settings_cls = MASTSettings
    signal_attrs = SIGNAL_ATTRS

    min_pulse_length = 0.2
    min_filter = {
        "ip": 210e3,
        # 5 kJ of plasma_energy cuts about what 10 kJ of the old wmhd did (median ratio 2.1),
        # 1.9 percent of the times iteration 12 kept, mostly ramp phases
        "energy_mhd": 5e3,
    }
    max_filter = {
        "greenwald_fraction": 2.0,
        # Sample validity, distinct from the transient gate below:
        # MAST total input power tops out near 5 MW, so a recorded radiated power above 4 MW is not a valid measurement.
        # No minimum, the bolometer noise dips below 0 for 1-5 ms (median -0.25 MW) and _clip_powers writes them as 0.
        "power_radiated": 4e6,
    }
    # Both thresholds are empirical, and both gate the radiative or ohmic collapse
    # rather than normal operation: the closest ordinary approach found while porting these was shot 29153,
    # whose ohmic power peaks at 3.9 MW right before the end of the shot.
    # On 100 random shots read from source, 3 MW radiated fires in 5 and changes no kept time.
    transient_filter = {
        "power_ohm": 5.0e6,
        "power_radiated": 3.0e6,
    }
    # MAST's ip record runs through the current quench, so a small ip cutoff
    # still lets disruption transients in. Has to be longer than C-Mod's 20 ms.
    end_margin = 0.04
    # Shots radiate a median 10 percent of their heating power (it11 store, from 23809 on).
    # 12 shots sit below 1 percent with MW of input, a dead bolometer, and the next lowest is at 3.75 percent.
    min_radiated_fraction = 0.025
    # There appears to be a systematic change in TS calibrations after this early-campaign shot.
    first_shot = 23809
    # From 23809 on, 23990 sits at 0.13 and the next lowest shot at 0.75, the highest at 1.10
    density_ratio_bounds = (0.7, 1.3)
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

        timebase = sources.timebase
        ds_0d = _zero_d_dataset(
            shot, sources.summary, sources.efm, sources.esm, timebase
        )
        ds_equilibrium_efit = _equilibrium_dataset(shot, sources.efm)
        ds_equilibrium = snap_to_grid(ds_equilibrium_efit, timebase)
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
        4: Drop the inboard channels past MAX_INBOARD_RHO_TOR_NORM
        5: Inflate the errors where the inboard and outboard branches disagree (_branch_disagreement_errors)
        6: Drop the channels outside the fittable range

        BOTH sides of the chord are fit, the inboard side only inside MAX_INBOARD_RHO_TOR_NORM.
        The inboard side maps onto the same rho_tor_norm through the reconstruction's interior flux,
        which a magnetics-only reconstruction does not pin precisely,
        and on a spherical tokamak Te is not strictly a flux function (poloidal asymmetries can be real).

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
        with np.errstate(invalid="ignore"):
            inboard_edge = inboard & (rho_tor_norm > MAX_INBOARD_RHO_TOR_NORM)
        n_inboard_edge = int((inboard_edge & np.isfinite(te_y)).sum())
        te_y = np.where(inboard_edge, np.nan, te_y)
        ne_y = np.where(inboard_edge, np.nan, ne_y)
        logger.info(
            f"Shot {shot}: dropped {n_inboard_edge} inboard channel readings past "
            f"rho_tor_norm {MAX_INBOARD_RHO_TOR_NORM:g}"
        )
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

        The rows split at their lowest-rho channel, as in prepare_fit_input (_inboard_channels).
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
    A channel with fewer than BRANCH_MIN_CHANNELS estimates within BRANCH_SMOOTH_HALFWIDTH keeps its error.

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
        inflate = np.isfinite(disagreement)
        err_out[i_time, inflate] = np.hypot(
            err[i_time, inflate], 0.5 * disagreement[inflate]
        )
    return err_out


def open_shot_sources(shot: int) -> ShotSources | None:
    """Open one shot's stores and check they carry everything get_source_dataset reads.

    The groups open in the order of REQUIRED_LEVEL1_SIGNALS, then REQUIRED_LEVEL2_SIGNALS,
    and the first one missing a signal fails the shot.
    The shot window ends at the last plasma current above SHOT_WINDOW_MIN_IP.

    Args:
        shot: Shot number to open.

    Returns:
        The opened sources, or None when a store could not be reached,
        which is worth retrying on the next run.

    Raises:
        MissingSourceError: If the shot has no store, a required group or signal,
            plasma current, or usable Thomson slice.
    """
    groups = {}
    for store_path, required, level in (
        (LEVEL1_PATH, REQUIRED_LEVEL1_SIGNALS, "level 1"),
        (LEVEL2_PATH, REQUIRED_LEVEL2_SIGNALS, "level 2"),
    ):
        for group, names in required.items():
            ds_group = _open_store_group(store_path, shot, group)
            if ds_group is None:
                if _store_path_exists(f"{store_path}/{shot}.zarr/{group}"):
                    return None
                raise MissingSourceError(f"No {level} {group} group for this shot.")
            missing = [f"{group}/{name}" for name in names if name not in ds_group]
            if missing:
                raise MissingSourceError(
                    f"Missing {level} signals: {', '.join(missing)}."
                )
            groups[group] = ds_group

    summary = groups["summary"]
    summary_time = summary["time"].values
    ip = np.asarray(summary["ip"].values, dtype=float)
    in_shot = np.abs(ip) > SHOT_WINDOW_MIN_IP
    if in_shot.sum() < 2:
        raise MissingSourceError(f"No plasma current above {SHOT_WINDOW_MIN_IP:.0f} A.")
    timebase = make_uniform_1kHz_timebase(float(summary_time[in_shot][-1]))

    ds_thomson = _thomson_dataset(shot, groups["ayc"], timebase)
    if ds_thomson.sizes["idx"] == 0:
        raise MissingSourceError(
            "No Thomson slices with usable data within the shot window."
        )
    return ShotSources(
        summary=summary,
        efm=groups["efm"],
        esm=groups["esm"],
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


def _zero_d_dataset(
    shot: int,
    summary: xr.Dataset,
    efm: xr.Dataset,
    esm: xr.Dataset,
    timebase: np.ndarray,
) -> xr.Dataset:
    """Hold the 0D signals forward onto the timebase under standardized names.

    Every signal is held from its last finite sample (held_signal_on_grid), never interpolated,
    so no grid time draws on a later sample.
    Plasma current and toroidal field are stored as magnitudes, the signed
    versions live in the equilibrium signals (see _equilibrium_dataset).

    Args:
        shot: Shot number being read.
        summary: The level 2 summary group.
        efm: The level 1 efm group.
        esm: The level 1 esm group.
        timebase: Uniform 1 kHz timebase [s].

    Returns:
        Dataset of 0D signals on dim "idx", with "time" and "shot" coords.
    """
    summary_time = summary["time"].values
    eq_time = np.asarray(efm["time"].values, dtype=float)

    data = {
        name: held_signal_on_grid(eq_time, efm[source].values, timebase)
        for source, name in EQUILIBRIUM_SIGNALS.items()
    }
    data.update(
        {
            name: held_signal_on_grid(summary_time, summary[source].values, timebase)
            for source, name in SUMMARY_SIGNALS.items()
        }
    )

    ip = np.asarray(summary["ip"].values, dtype=float)
    ip_on_timebase = held_signal_on_grid(summary_time, ip, timebase)
    data["ip"] = np.abs(ip_on_timebase)
    # bvac_rmag is the vacuum field at the magnetic axis. Rescale it by 1/R to
    # the geometric axis, so b0 is referenced the same way as on the other
    # devices (C-Mod rout, D3D rsurf, TCV R_geom).
    bvac_rmag = held_signal_on_grid(eq_time, efm["bvac_rmag"].values, timebase)
    r_axis = held_signal_on_grid(eq_time, efm["magnetic_axis_r"].values, timebase)
    data["b0"] = np.abs(bvac_rmag * r_axis / data["geometric_axis_r"])
    # esm sits on a 20 us axis but only holds values at the reconstruction times,
    # so the hold runs on the clock of the samples that have a pphix.
    # Some converged reconstructions have none (97 of 1446 shots, up to 25 ms in 24891),
    # and a gap longer than the hold stays NaN.
    esm_time = np.asarray(esm["time"].values, dtype=float)
    pphix = np.asarray(esm["pphix"].values, dtype=float)
    has_pphix = np.isfinite(pphix)
    data["power_ohm"] = held_signal_on_grid(
        esm_time[has_pphix], pphix[has_pphix], timebase
    )
    data["power_nbi"] = held_signal_on_grid(
        summary_time, summary["power_nbi"].values, timebase
    )
    # MAST has no ICRF or lower hybrid, zero where ip is valid
    data["power_ic"] = data["ip"] * 0.0
    data["power_lh"] = data["ip"] * 0.0
    data["power_ec"] = data["ip"] * 0.0

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


def efm_flux_map(efm: xr.Dataset) -> xr.DataArray:
    """Read the poloidal flux map of the efm group on its own grid.

    psirz shares its radial dimension with the 129-point radial profiles (jr, qr, ...),
    so it is NaN everywhere off its own gridr points.
    Those points sit up to one float32 step (~1e-7 m) off gridr itself,
    and the union's closest points are 3e-4 m apart, hence the nearest match within 1e-6 m.

    Args:
        efm: The shot's level 1 efm group.

    Returns:
        The flux map [Wb/rad] on dims (time, major_radius, z), with gridr and gridz as coordinates.
    """
    r_grid = efm["gridr"].values
    z_grid = efm["gridz"].values
    psirz = efm["psirz"].sel(
        profile_r=r_grid, profile_z=z_grid, method="nearest", tolerance=1e-6
    )
    psirz = psirz.rename(profile_r="major_radius", profile_z="z")
    psirz = psirz.assign_coords(major_radius=r_grid, z=z_grid)
    return psirz.transpose("time", "major_radius", "z")


def _equilibrium_dataset(shot: int, efm: xr.Dataset) -> xr.Dataset:
    """Build the full GEQDSK reconstruction, on the EFIT timebase.

    Profiles and contours the group is missing are staged as NaN rather than dropping the shot.

    Args:
        shot: Shot number being read.
        efm: The shot's level 1 efm group.

    Returns:
        Dataset on dim "idx" with "time"/"shot" coords, in the freeqdsk
        canonical names (see machine.generic.make_geqdsk_dataset), carrying the
        COCOS number as an attribute.
    """
    eq_time = np.asarray(efm["time"].values, dtype=float)
    n_psi = efm.sizes["psi_norm"]
    psi_map = efm_flux_map(efm)
    psirz = np.asarray(psi_map.values, dtype=float)
    current = np.asarray(efm["plasma_current_c"].values, dtype=float)
    r_grid = np.asarray(efm["gridr"].values, dtype=float)
    # EFIT's own reference radius and the vacuum field there, bvac_r (a fixed 1.0 m) and bvac_val.
    # The same pair C-Mod's EFIT writes as RZERO and BCENTR, and bcentr * rcentr equals fpol at the boundary.
    bvac_r = np.asarray(efm["bvac_r"].values, dtype=float)
    rcentr = float(np.nanmedian(bvac_r))
    bcentr = np.asarray(efm["bvac_val"].values, dtype=float)
    r_axis = np.asarray(efm["magnetic_axis_r"].values, dtype=float)
    simagx = np.asarray(efm["psi_axis"].values, dtype=float)
    sibdry = np.asarray(efm["psi_boundary"].values, dtype=float)
    profiles = {
        name: _optional_rows(shot, efm, source, (eq_time.size, n_psi))
        for name, source in GEQDSK_PROFILES.items()
    }
    n_boundary = efm.sizes.get("lcfs_coords", 1)
    boundary = {
        name: _optional_rows(shot, efm, source, (eq_time.size, n_boundary))
        for name, source in (("rbdry", "lcfs_r"), ("zbdry", "lcfs_z"))
    }
    if "limiterr" not in efm or "limiterz" not in efm:
        logger.warning(f"Shot {shot}: no limiter contour in efm")
        rlim = zlim = None
    else:
        rlim = np.asarray(efm["limiterr"].values, dtype=float)
        zlim = np.asarray(efm["limiterz"].values, dtype=float)

    return make_geqdsk_dataset(
        shot_id=shot,
        times=eq_time,
        r_grid=r_grid,
        z_grid=np.asarray(efm["gridz"].values, dtype=float),
        rmagx=r_axis,
        zmagx=np.asarray(efm["magnetic_axis_z"].values, dtype=float),
        simagx=simagx,
        sibdry=sibdry,
        bcentr=bcentr,
        current=current,
        psirz=psirz,
        cocos_input=cocos_from_signs(current, bcentr, simagx, sibdry, profiles["qpsi"]),
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
