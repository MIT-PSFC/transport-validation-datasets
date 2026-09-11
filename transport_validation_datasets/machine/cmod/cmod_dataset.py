from dataclasses import dataclass, field, replace

import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets.dispy_utils import passive_log_settings, summary
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.cmod.dispy_methods import (
    CmodEfitMethods,
    CmodGeometryMethods,
    CmodThomsonMethods,
    UniformTimeSetting,
)
from transport_validation_datasets.machine.generic import (
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho,
    snap_to_grid,
    ts_channel_fit_rows,
)
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# Attributes of the signals this module makes rather than reads with
# attributes attached (DataWorkflow.signal_attrs). Everything else carries
# disruption-py's, rewritten to the shared convention at the stack stage.
SIGNAL_ATTRS = {
    "power_nbi": {
        "description": "Neutral beam heating power (none on C-Mod)",
        "units": "W",
        "ref": "/summary/heating_current_drive/power_nbi/value",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary (EFIT rout)",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/geometric_axis/r",
    },
}


@dataclass(frozen=True)
class CModSettings(DeviceSettings):
    """C-Mod settings, the [cmod] table of the config file.

    Attributes:
        efit_nickname_below: Shot-number threshold -> EFIT tree overrides,
            resolved before efit_nickname: a shot uses the tree of the
            smallest threshold it falls below, so multiple entries split
            the shot range into eras. The default sends every shot before
            1050204013 (the 2003-2004 campaigns) to the ANALYSIS tree. In
            TOML: [cmod.efit_nickname_below] with entries like
            "1050204013" = "analysis" (keys are strings, TOML tables
            require it). An empty table disables the overrides.
        efit_nickname: EFIT tree the equilibrium, the geometry signals, and
            the shot's 1 kHz timebase are read from, and so the one the
            Thomson channels are mapped onto rho with. A tree name (EFIT21,
            EFIT18, ...), or one of disruption-py's keys: "analysis" for the
            ANALYSIS tree, "disruption" for the disruption EFIT.
        channel_prefilters: Apply the legacy pre-fit channel conditioning
            (ported from the standalone fit_cmod.py) during fit staging, in
            the original's steady-mode order: ne error conditioning at
            ingest (_condition_ne_errors, in prepare_fit_input), then after
            windowing the per-row outlier culls (_cull_channel_outliers)
            and the SOL anchor points (_append_sol_anchor_points), both in
            condition_staged_fit_input so a pooled window is judged as one
            cloud and gets exactly one pair of anchors. Off by default:
            staged batches are shared by every fit method, and this
            conditioning was calibrated for the akho-style fits, though any
            method can opt in.
    """

    efit_nickname: str = "EFIT21"
    efit_nickname_below: dict[str, str] = field(
        default_factory=lambda: {"1050204013": "analysis"}
    )
    channel_prefilters: bool = False


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets."""

    settings_cls = CModSettings
    signal_attrs = SIGNAL_ATTRS

    min_pulse_length = 0.5
    min_usable_time = 0.2
    min_segment_length = 0.1
    valid_filter = {
        "ip": {"min_abs": 100e3},  # Only care about magnitude of ip
        "n_e_line_average": {"min": 1e18, "max": 4e20},
        "energy_mhd": {"min": 3e3},
        "beta_tor_norm": {"min": 0.08, "max": 2.0},
    }
    transient_filter = {
        "power_ohm": 5.0e6,
        "power_radiated": 2.5e6,
    }
    end_margin = 0.02
    shot_blacklist = []

    # GP fit staging knobs (values ported from the transport_study C-Mod
    # config, with their calibration notes).
    fit_rho = np.linspace(0.0, 1.1, 56)
    # Minimum valid (rho, value) pairs required per timestep to run the GP fit,
    # compared against the channel count AFTER the per-shot quality screens (_drop_broken_channels).
    # Many C-Mod shots carry exactly 10 channels, so tolerate one bad channel.
    fit_min_points = 9
    fit_scale_per_slice = True
    # Hyperparameter bounds for the GP fit, per variable.
    fit_bounds = {
        "te": FitBounds(l1_min=0.35),
        "ne": FitBounds(l1_min=0.55),
    }

    def get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from C-Mod SQL database.

        Returns:
            Shot numbers to process, filtered to days with blessed Thomson data.
        """
        data = summary(
            summary_table="summary",
            ipmax=self.valid_filter["ip"]["min_abs"],
            pulse_length=self.min_pulse_length,
            min_shot=1160500000,
            max_shot=1160932000,
            shots=False,
        )
        shotlist = data[:, 0].astype(int).tolist()

        # Days with blessed TS data, email from J. Hughes 2025-12-12
        blessed_days = [
            1160503,
            1160527,
            1160621,
            1160628,
            1160630,
            1160708,
        ]
        blessed_day_ranges = [
            [1160607, 1160610],
            [1160712, 1160719],
            [1160803, 1160820],
            [1160823, 1160903],
            [1160908, 1160916],
            [1160919, 1160924],
            [1160927, 1160931],
        ]
        for day_range in blessed_day_ranges:
            blessed_days.extend(range(day_range[0], day_range[1] + 1))

        # Filter to blessed Thomson days
        shotlist = [shot for shot in shotlist if shot // 1000 in blessed_days]
        return shotlist

    def _efit_nickname(self, shot: int) -> str:
        """Resolve the EFIT tree one shot's retrievals read from.

        Args:
            shot: Shot number being retrieved.

        Returns:
            The tree of the smallest efit_nickname_below threshold the shot
            falls below, or efit_nickname when it falls below none.
        """
        best: tuple[int, str] | None = None
        for threshold, tree in self.settings.efit_nickname_below.items():
            limit = int(threshold)
            if shot < limit and (best is None or limit < best[0]):
                best = (limit, tree)
        return best[1] if best is not None else self.settings.efit_nickname

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from MDSplus, through disruption-py, into standardized signals.

        Three retrievals rather than one, because their native timebases
        differ: the fast 0D signals (Ip, B0, shaping, density, power) and the
        EFIT reconstruction are both native 1 kHz, Thomson scattering is
        native ~20 Hz and gets snapped onto the 1 kHz grid.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        datasets = []
        for name, getter in (
            ("fast", _get_fast_dataset),
            ("efit", _get_efit_dataset),
            ("thomson", _get_thomson_dataset),
        ):
            ds = getter(shot, self._efit_nickname(shot))
            if ds is None:
                reason = f"Missing retrievable {name} data."
                logger.warning(f"Shot {shot} is {reason.lower()} Skipping.")
                self.record_failed_shot(shot, reason)
                return None
            datasets.append(ds)

        ds_merged = xr.merge(datasets, compat="no_conflicts", join="outer")
        # merge keeps the first dataset's attributes only, and COCOS is in the EFIT one
        for ds in datasets:
            if "cocos" in ds.attrs:
                ds_merged.attrs["cocos"] = ds.attrs["cocos"]
        ds_standardized = self.standardize_signal_names(ds_merged)
        if ds_standardized is None:
            logger.warning(f"Shot {shot} is missing critical signals. Skipping.")
            self.record_failed_shot(shot, "Missing critical signals.")
        return ds_standardized

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals to IMAS-like names, keeping freeqdsk names for EFIT signals.

        All signals stay in SI units. Plasma current and toroidal field are stored
        as magnitudes, sign conventions live in the geqdsk signals.

        Args:
            ds: Merged dataset with disruption-py signal names.

        Returns:
            Dataset with standardized signal names, or None if the shot is missing
            critical signals.
        """
        # disruption-py signal name -> IMAS-like name
        imas_rename = {
            # summary/global_quantities
            "bt": "b0",
            "wmhd": "energy_mhd",
            "beta_n": "beta_tor_norm",
            "p_oh": "power_ohm",
            "p_rad": "power_radiated",
            # summary/heating_current_drive
            "p_icrf": "power_ic",
            "p_lh": "power_lh",
            # summary/line_average
            "n_e": "n_e_line_average",
            # equilibrium/time_slice/boundary
            "a_minor": "minor_radius",
            "kappa": "elongation",
            "tritop": "triangularity_upper",
            "tribot": "triangularity_lower",
            "rout": "geometric_axis_r",
            # thomson_scattering/channel
            "ts_channel_ne": "ts_channel_n_e",
            "ts_channel_ne_error": "ts_channel_n_e_error",
            "ts_channel_te": "ts_channel_t_e",
            "ts_channel_te_error": "ts_channel_t_e_error",
        }

        ds = ds.rename({k: v for k, v in imas_rename.items() if k in ds})

        # TS profiles are the point of the dataset. When get_thomson_channels fails,
        # disruption-py still fills its declared columns, but with NaN on the 0D
        # timebase, so there is no ts_channel dim to plot or fit
        if "ts_channel_n_e" not in ds or "ts_channel" not in ds["ts_channel_n_e"].dims:
            logger.warning("No Thomson scattering channels retrieved for this shot.")
            return None

        # C-Mod has no NBI, zero where ip is valid
        if "ip" in ds:
            ds["power_nbi"] = ds["ip"] * 0.0
            ds["power_nbi"].attrs = {}
        for name, attrs in self.signal_attrs.items():
            if name in ds:
                ds[name].attrs.update(attrs)

        return ds

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        1: Maps the TS channels onto rho through magnetics-only EFIT
        2: convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: C-Mod channel quality screens and error floors, calibrated in those units

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ts_times, rho = map_ts_channels_to_rho(ds)
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(
            ds.squeeze("shot", drop=True), ts_times
        )

        # Drop density channels too uncertain to constrain the fit
        # (error > 1e20 m^-3). These are typically bad edge/SOL channels.
        # Seen on shot 1160609014: a ne~4, err~2 channel past the separatrix
        # (rho~1.05) drove a spike to ne~19 at rho=1.0.
        ne_y = np.where(ne_err > 1.0, np.nan, ne_y)
        # Drop density points past the separatrix (rho>1.0) reading > 0.9e20:
        # SOL density is low out there, so such a point is a bad channel
        ne_y = np.where((rho > 1.0) & (ne_y > 0.9), np.nan, ne_y)

        # If data or error bar is incredibly small, set to NaN since it is
        # probably bad data. At this point ne is in 1e20 m^-3 and Te in keV.
        te_y = np.where(te_y < 0.001, np.nan, te_y)
        te_err = np.where(te_err < 0.001, np.nan, te_err)
        ne_y = np.where(ne_y < 0.001, np.nan, ne_y)
        ne_err = np.where(ne_err < 0.001, np.nan, ne_err)

        # Near the magnetic axis, Te this low is not physically real
        core_problem = (rho >= 0.0) & (rho < 0.4) & (te_y < 0.4)
        te_y = np.where(core_problem, np.nan, te_y)

        # The legacy ne error conditioning, before the error floors so it
        # sees the measured errors -- the original applied it at ingest,
        # before its steady-mode pooling. The culls and the SOL anchors run
        # after windowing instead (condition_staged_fit_input).
        if self.settings.channel_prefilters:
            ne_err = _condition_ne_errors(rho, ne_y, ne_err)

        # Error floors. Sometimes C-Mod TS has extremely tiny error bars
        # which I don't think are real. This increases them where needed.
        # Te: absolute 0.1 keV
        # ne: Multiply 'measured' error 1.5x and floor at 10 percent of the value with a 1e18/m3 absolute floor.
        te_err = np.where(te_err < 0.1, 0.1, te_err)
        ne_err = np.maximum(1.5 * ne_err, np.maximum(0.10 * np.abs(ne_y), 0.01))

        # After the floors: the persistence screen must see the same errors
        # the fit will (its thresholds are calibrated on them).
        te_y = _drop_broken_channels("te", rho, te_y, te_err)
        ne_y = _drop_broken_channels("ne", rho, ne_y, ne_err)

        fit_input = ShotFitInput(
            x=rho, te_y=te_y, te_err=te_err, ne_y=ne_y, ne_err=ne_err, time=ts_times
        )
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no finite (rho, te, ne) channel data to fit")
            return None
        return fit_input

    def condition_staged_fit_input(
        self, shot: int, fit_input: ShotFitInput
    ) -> ShotFitInput:
        """Apply the legacy culls and SOL anchors to the rows as they are fit.

        Runs after windowing, matching the original fit_cmod.py's
        steady-mode order (its complicated filter and anchor injection came
        after the steady-time collapse): in averaging mode each pooled
        window is judged as one cloud with window-local core references and
        gets exactly one pair of anchor points; in per-sample mode each
        slice is judged on its own the same way. The anchors' errors are
        their weight, so no error floor touches them here.

        Args:
            shot: Shot number being staged.
            fit_input: The shot's fit input as _apply_windows staged it.

        Returns:
            The conditioned fit input.
        """
        if not self.settings.channel_prefilters:
            return fit_input
        te_y, ne_y = _cull_channel_outliers(
            shot, fit_input.x, fit_input.te_y, fit_input.ne_y
        )
        x, te_y, te_err, ne_y, ne_err = _append_sol_anchor_points(
            fit_input.x, te_y, fit_input.te_err, ne_y, fit_input.ne_err
        )
        return replace(
            fit_input, x=x, te_y=te_y, te_err=te_err, ne_y=ne_y, ne_err=ne_err
        )

    def fit_plot_channel_groups(self, shot: int) -> list | None:
        """Split the fit-plot channels into the core and edge TS systems.

        Args:
            shot: Shot number being plotted.

        Returns:
            (mask, color, label) triples for the two Thomson arrays.
        """
        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds:
            ts_array = ds["ts_array"].values
        return [
            (ts_array == "core", "tab:blue", "core TS"),
            (ts_array == "edge", "tab:orange", "edge TS"),
        ]


def _condition_ne_errors(
    rho: np.ndarray, ne_y: np.ndarray, ne_err: np.ndarray
) -> np.ndarray:
    """Apply the legacy ne error conditioning to the measured errors.

    Ported from the standalone fit_cmod.py's filter_and_flatten_data, which
    applied it at ingest, per point, before any pooling: edge channels
    (rho > 0.9) with relative error above 40 percent get their error shrunk
    0.3x, non-edge channels with relative error under 5 percent get it
    inflated 5x. Call before the fit's error floors, on the measured errors.

    Args:
        rho: (n_t, n_ch) channel rho positions.
        ne_y: (n_t, n_ch) ne values [1e20 m^-3], NaN where invalid.
        ne_err: (n_t, n_ch) measured ne errors [1e20 m^-3].

    Returns:
        The conditioned errors, as a copy.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = ne_err / np.abs(ne_y)
    ne_err = np.where((rho > 0.9) & (rel_err > 0.4), 0.3 * ne_err, ne_err)
    return np.where((rho <= 0.9) & (rel_err < 0.05), 5.0 * ne_err, ne_err)


# Outer edge of the core-reference region (_row_core's core max). The
# original fit_cmod.py used rho < 0.2 for both variables; ne's is widened
# to 0.35 because some scenario windows' innermost mapped channel sits at
# rho 0.25-0.31, and with no reading inside the reference region the
# tiny-fallback core max turns the 1.1x cull into "drop every ne reading".
# Te keeps the original region: its culls only arm WITH a core reference,
# so the fallback is harmless there.
_TE_CORE_RHO_MAX = 0.2
_NE_CORE_RHO_MAX = 0.35


def _row_core(
    r: np.ndarray, y: np.ndarray, core_rho_max: float
) -> tuple[float, float, float, float]:
    """Characterize one fit row's coverage for the legacy culls and anchors.

    A row is whatever will be fit as one profile: a Thomson slice, or a
    whole pooled time window in averaging mode.

    Args:
        r: (n_ch,) channel rho positions, NaN where padded.
        y: (n_ch,) channel values.
        core_rho_max: Outer edge of the core-reference region the core max
            is read from.

    Returns:
        (core max, core value, innermost rho, outermost rho) over the
        finite channels: core max is the brightest reading at
        rho < core_rho_max (a tiny fallback mirroring the original's
        0.01-in-raw-units default when that region is not covered), the
        core value is the mean reading of the innermost channel, 0.0 when
        that channel sits at rho >= 0.3 (matching the original, whose core
        reference stays unset then), and everything is
        (fallback, 0.0, inf, -inf) with no finite channel.
    """
    valid = np.isfinite(r) & np.isfinite(y)
    core = y[valid & (r < core_rho_max)]
    core_max = float(core.max()) if core.size else 1.0e-22
    if not valid.any():
        return core_max, 0.0, np.inf, -np.inf
    r_min = float(r[valid].min())
    r_max = float(r[valid].max())
    if r_min >= 0.3:
        return core_max, 0.0, r_min, r_max
    at_min = valid & np.isclose(r, r_min)
    return core_max, float(np.mean(y[at_min])), r_min, r_max


def _cull_channel_outliers(
    shot: int, rho: np.ndarray, te_y: np.ndarray, ne_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the legacy outlier culls to the rows as they will be fit.

    Ported from the standalone fit_cmod.py's filter_and_flatten_data (its
    "complicated filter"), which ran after the steady-time collapse -- so
    here every row (a slice, or a whole pooled window in averaging mode) is
    judged as one cloud with row-local core references, the way each
    scenario window was one invocation there. Converted from its raw units
    to the fit units (Te [keV], ne [1e20 m^-3]).

    Per row, with core max the row's brightest reading in the variable's
    core-reference region (rho < 0.2 for Te as in the original; widened to
    rho < 0.35 for ne, see _NE_CORE_RHO_MAX) and the core value the mean
    reading of the row's innermost channel (only defined when it sits at
    rho < 0.3):
      - ne: any reading above 1.1x the row's core max is dropped.
      - ne: with edge coverage (rho > 0.9) and a bright core (at least 0.5x
        the core max), edge readings above 0.8x the core value are dropped.
      - Te: with edge coverage and a core of at least 0.3x the core max,
        edge readings above 0.3x the core value are dropped.
      - Te: with a core hotter than 0.3 keV, readings of at most 0.08 keV
        inside rho <= 0.95 are dropped (dead or cold channels inside the
        plasma).

    Args:
        shot: Shot number, for the log line.
        rho: (n_rows, n_ch) channel rho positions.
        te_y: (n_rows, n_ch) Te values [keV], NaN where invalid.
        ne_y: (n_rows, n_ch) ne values [1e20 m^-3], NaN where invalid.

    Returns:
        (te_y, ne_y) with the culls applied, as copies.
    """
    te_y = te_y.copy()
    ne_y = ne_y.copy()
    n_culled = 0

    for t in range(rho.shape[0]):
        r = rho[t]

        core_max, core, r_min, r_max = _row_core(r, ne_y[t], _NE_CORE_RHO_MAX)
        cull = np.isfinite(ne_y[t]) & (ne_y[t] > 1.1 * core_max)
        if r_max > 0.9 and core >= 0.5 * core_max:
            cull |= (r >= 0.9) & (ne_y[t] > 0.8 * core)
        if cull.any():
            n_culled += int(np.count_nonzero(cull & np.isfinite(ne_y[t])))
            ne_y[t] = np.where(cull, np.nan, ne_y[t])

        core_max, core, r_min, r_max = _row_core(r, te_y[t], _TE_CORE_RHO_MAX)
        cull = np.zeros(r.shape, dtype=bool)
        if r_max > 0.9 and core >= 0.3 * core_max:
            cull |= (r >= 0.9) & (te_y[t] > 0.3 * core)
        if r_min < 0.95 and core > 0.3:
            cull |= (r <= 0.95) & (te_y[t] <= 0.08)
        if cull.any():
            n_culled += int(np.count_nonzero(cull & np.isfinite(te_y[t])))
            te_y[t] = np.where(cull, np.nan, te_y[t])

    if n_culled:
        logger.info(
            f"Shot {shot}: channel prefilters culled {n_culled} channel readings"
        )
    return te_y, ne_y


def _append_sol_anchor_points(
    rho: np.ndarray,
    te_y: np.ndarray,
    te_err: np.ndarray,
    ne_y: np.ndarray,
    ne_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Append the legacy synthetic SOL anchor points as two extra channels.

    Ported from fit_cmod.py's filter_and_flatten_data (its upper_rho_bc
    block), which injected them after the steady-time collapse: every fit
    row (a slice, or a whole pooled window in averaging mode) gets exactly
    two synthetic boundary-condition points at rho 1.05 and 1.08 pinning
    the fit to low SOL values, converted to the fit units. Te anchors are
    fixed at 0.040/0.030 keV (2x the legacy 0.020/0.015, per explicit
    direction; errors 0.010/0.007); ne anchors are 0.5/0.3 [1e20 m^-3]
    (0.5x the legacy 1.0/0.6, per explicit direction; errors 0.3/0.3)
    scaled per row by the row's brightest reading relative to its own core
    (rho < 0.2) maximum, as in the original. A row with no finite ne gets
    NaN ne anchors, so it stays unfittable.

    Call after windowing, the error floors, and the persistence screen: the
    anchor errors ARE the anchors' weight (the original never modified them
    after injection), and only real channels belong in the persistence
    statistics.

    Args:
        rho: (n_rows, n_ch) channel rho positions.
        te_y: (n_rows, n_ch) Te values [keV].
        te_err: (n_rows, n_ch) Te errors [keV].
        ne_y: (n_rows, n_ch) ne values [1e20 m^-3].
        ne_err: (n_rows, n_ch) ne errors [1e20 m^-3].

    Returns:
        (rho, te_y, te_err, ne_y, ne_err) with two anchor channels appended.
    """
    n_t = rho.shape[0]

    core_max = np.array(
        [_row_core(rho[t], ne_y[t], _NE_CORE_RHO_MAX)[0] for t in range(n_t)]
    )
    row_max = np.where(np.isfinite(ne_y), ne_y, -np.inf).max(axis=1)
    row_max = np.where(np.isfinite(row_max), row_max, np.nan)
    v = (row_max / core_max)[:, None]

    def rows(values: tuple[float, float]) -> np.ndarray:
        return np.tile(np.asarray(values, dtype=float), (n_t, 1))

    return (
        np.concatenate([rho, rows((1.05, 1.08))], axis=1),
        np.concatenate([te_y, rows((0.040, 0.030))], axis=1),
        np.concatenate([te_err, rows((0.010, 0.007))], axis=1),
        np.concatenate([ne_y, v * rows((0.5, 0.3))], axis=1),
        np.concatenate([ne_err, v * rows((0.3, 0.3))], axis=1),
    )


def _drop_broken_channels(
    var_name: str, data_x: np.ndarray, data_y: np.ndarray, err_y: np.ndarray
) -> np.ndarray:
    """NaN out channels biased against their neighbors in the same way for an entire shot.

    Per-slice outlier removal (the fit worker's LOO pass) judges each slice in
    isolation, so a channel that is only ~2-4 sigma off per slice can survive.
    Persistence across the shot is what separates broken hardware from real structure.
    A healthy channel has neighbors above and below it, a miscalibrated channel is
    significantly biased in the same direction all shot.
    (Note that slight biasing is expected because profiles are monatonic-ish,
    this just looks for consitently extreme cases like [4, 1, 3], [9, 2, 7], etc.)

    Per slice, each channel with both rho-neighbors finite gets
    z = (y - neighbor_mean) / combined sigma, and a channel is dropped when
    |median z| >= 3.5 with >= 90 percent of slices on the same side, over >= 10 slices.

    Args:
        var_name: Variable name for the log line.
        data_x: (n_t, n_ch) channel rho positions.
        data_y: (n_t, n_ch) channel values.
        err_y: (n_t, n_ch) channel errors, with the fit's floors applied.

    Returns:
        data_y with broken channels NaNed (a copy if any were).

    """
    n_t, n_ch = data_y.shape
    z_sum: list[list[float]] = [[] for _ in range(n_ch)]
    for t in range(n_t):
        valid = np.isfinite(data_x[t]) & np.isfinite(data_y[t]) & np.isfinite(err_y[t])
        if int(valid.sum()) < 5:
            continue
        idx = np.flatnonzero(valid)
        order = idx[np.argsort(data_x[t][idx])]
        yv, ev = data_y[t][order], err_y[t][order]
        for k in range(1, order.size - 1):
            nbr_mean = 0.5 * (yv[k - 1] + yv[k + 1])
            sigma = np.sqrt(ev[k] ** 2 + 0.25 * (ev[k - 1] ** 2 + ev[k + 1] ** 2))
            z_sum[order[k]].append(float((yv[k] - nbr_mean) / sigma))
    broken = []
    for c in range(n_ch):
        z = np.asarray(z_sum[c])
        if z.size < 10:
            continue
        med = float(np.median(z))
        if abs(med) < 3.5:
            continue
        if float(np.mean(np.sign(z) == np.sign(med))) >= 0.9:
            broken.append(c)
    if not broken:
        return data_y
    logger.info(f"ts {var_name}: dropped persistently-biased channel(s) {broken}")
    data_y = data_y.copy()
    data_y[:, broken] = np.nan
    return data_y


def _is_empty_result(result: xr.Dataset) -> bool:
    """Check whether get_shots_data returned no usable data for a shot.

    When retrieval fails (e.g. a missing MDSplus tree), get_shots_data logs the
    error and returns an empty dataset with no shot/time index variables. Reshaping
    that with set_index would raise, so callers use this to skip the shot instead.

    Returns:
        True if the result has no usable shot/time data, False otherwise.
    """
    return "shot" not in result or "time" not in result or result["time"].size == 0


def _get_fast_dataset(shot: int, efit_nickname: str) -> xr.Dataset | None:
    """Retrieve fast 0D signals and EFIT dataset.

    Args:
        shot: Shot number to retrieve data for.
        efit_nickname: EFIT tree to read, see CModSettings.

    Returns:
        Dataset with EFIT signals for the given shot, or None if retrieval
        returned no data.
    """
    cmod_dataset_signals = [
        "ip",  # Plasma current
        "bt",  # On-axis magnetic field
        "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
        "n_e",  # Line average electron density [m^-3]
        "beta_n",  # Normalized beta
        "a_minor",  # Plasma minor radius
        "kappa",  # Plasma elongation
        "tritop",  # Top triangularity
        "tribot",  # Bottom triangularity
        "rout",  # Geometric major radius [m]
        # Power sources and sinks
        "p_oh",  # Ohmic heating power
        "p_rad",  # Bulk radiated heating power
        "p_icrf",  # ICRF heating power
        "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=cmod_dataset_signals,
        time_setting=UniformTimeSetting(),
        efit_nickname_setting=efit_nickname,
        only_requested_columns=True,
        custom_physics_methods=[CmodGeometryMethods.get_geometric_major_radius],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if _is_empty_result(result):
        return None
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_efit_dataset(shot: int, efit_nickname: str) -> xr.Dataset | None:
    """Retrieve EFIT dataset for the given shot.

    Args:
        shot: Shot number to retrieve data for.
        efit_nickname: EFIT tree to read, see CModSettings.

    Returns:
        Dataset with GEQDSK signals for the given shot, or None if retrieval
        returned no data.
    """
    settings = RetrievalSettings(
        run_methods=["get_geqdsk_parameters"],
        efit_nickname_setting=efit_nickname,
        time_setting=UniformTimeSetting(),
        custom_physics_methods=[CmodEfitMethods.get_geqdsk_parameters],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if _is_empty_result(result):
        return None
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_thomson_dataset(shot: int, efit_nickname: str) -> xr.Dataset | None:
    """Retrieve Thomson scattering data for the given shot.

    Args:
        shot: Shot number to retrieve data for.
        efit_nickname: EFIT tree to read, see CModSettings.

    Returns:
        Dataset with Thomson channel signals snapped to the uniform 1 kHz grid,
        or None if retrieval returned no data.
    """
    retrieval_settings = RetrievalSettings(
        run_methods=["get_thomson_channels"],
        efit_nickname_setting=efit_nickname,
        only_requested_columns=False,
        custom_physics_methods=[CmodThomsonMethods.get_thomson_channels],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=[shot],
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if _is_empty_result(result):
        return None
    # Snap native ~20 Hz TS slices onto the uniform 1 kHz grid, no interpolation.
    # Grid times with no TS slice come back as NaN.
    timebase = make_uniform_1kHz_timebase(float(result["time"].values.max()))
    result = snap_to_grid(result, timebase)
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result
