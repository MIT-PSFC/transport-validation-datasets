from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets.cleaning import drop_in_both, relative_dips
from transport_validation_datasets.dispy_utils import passive_log_settings, summary
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.cmod.dispy_methods import (
    CmodAeqdskMethods,
    CmodEfitMethods,
    CmodPlasmaMethods,
    CmodPowerMethods,
    CmodThomsonMethods,
    UniformTimeSetting,
)
from transport_validation_datasets.machine.generic import (
    EQUILIBRIUM_HOLD_FLOOR,
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho_tor_norm,
    signal_on_grid,
    snap_to_grid,
    ts_channel_fit_rows,
)
from transport_validation_datasets.store_schema import apply_signal_attrs
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# Attributes of the signals this module makes rather than reads with
# attributes attached (DataWorkflow.signal_attrs). Everything else carries
# disruption-py's, rewritten to the shared convention at the stack stage.
SIGNAL_ATTRS = {
    "ip": {
        "description": "Plasma current, magnetics ip (Rogowski coil), signed, mean of each 1 ms grid step",
    },
    "n_e_line_average": {
        "description": "Line-averaged electron density, TCI chord 4 (nl_04 / 0.6 m), mean of each 1 ms grid step",
    },
    "power_radiated": {
        "description": (
            "Total radiated power, AXUV twopi_diode x 4.5 (cross-calibrated to the 2pi foil bolometer), "
            "mean of each 1 ms grid step, smoothed by a centered 50 ms boxcar applied twice (non-causal), clipped at 0"
        ),
    },
    "power_ic": {
        "description": "ICRF net heating power (rf_power_net), mean of each 1 ms grid step, zero outside its record",
    },
    "power_lh": {
        "description": "Lower hybrid net heating power (LH netpow), mean of each 1 ms grid step, zero outside its record",
    },
    "b0": {
        "description": "Vacuum toroidal field at r0, magnetics btor, signed, mean of each 1 ms grid step",
    },
    "r0": {
        "description": "Reference major radius btor is quoted at, the EFIT RZERO",
    },
    "beta_tor_norm": {
        "description": "Normalized toroidal beta from the EFIT tree (betan)",
    },
    "power_ohm": {
        "description": (
            "Ohmic heating power, Ip * V_loop minus the rate of change of the internal poloidal magnetic energy "
            "mu0 R_geo li Ip^2 / 4 with R_geo the EFIT rout (backward difference), smoothed by a centered 50 ms boxcar applied twice (non-causal), clipped at 0"
        ),
    },
    "power_nbi": {
        "description": "Neutral beam heating power (none on C-Mod)",
    },
    "power_ec": {
        "description": "Electron cyclotron heating power (none on C-Mod)",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary (EFIT rout)",
    },
}

# An EFIT tree whose reconstructions sit further apart than this is slow [s].
# Snapping its 0D signals would leave most of the 1 kHz grid NaN, so they are held forward instead.
# EFIT21 reconstructs every 1 ms, ANALYSIS every ~20 ms.
SLOW_EFIT_PERIOD = 1.5e-3

# Major radius the magnetics btor is quoted at, the store's r0 [m].
# btor, less its pre-shot baseline, matches EFIT bcentr to 0.9996-0.9999,
# and EFIT quotes bcentr at rcentr, stored as RZERO = 0.66 m.
R0 = 0.66


@dataclass(frozen=True)
class CModSettings(DeviceSettings):
    """C-Mod settings, the [cmod] table of the config file.

    Attributes:
        efit_trees: EFIT trees to read a shot from, in order of preference.
            The equilibrium, the geometry signals and the shot's 1 kHz timebase all come from one tree,
            and so does the rho_tor_norm mapping of the Thomson channels.
            A shot takes the first tree that serves every retrieval (see get_source_dataset).
    """

    efit_trees: list[str] = field(default_factory=lambda: ["EFIT21"])

    def __post_init__(self):
        super().__post_init__()
        # A bare string would be read as a list of one-letter trees
        if isinstance(self.efit_trees, str) or not self.efit_trees:
            raise ValueError(
                f"efit_trees must be a non-empty list of tree names, got {self.efit_trees!r}"
            )


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets."""

    settings_cls = CModSettings
    signal_attrs = SIGNAL_ATTRS

    min_pulse_length = 0.5
    min_filter = {
        "ip": 100e3,
        # Lowered from 3 kJ so the ramp-ups count, 2.7 kJ adds 3 s of kept time over 990 shots read from source.
        "energy_mhd": 2.7e3,
        # A broken interferometer record, the lowest kept value in a 40-shot sample is 2.4e19
        "n_e_line_average": 1e19,
    }
    max_filter = {
        "greenwald_fraction": 2.0,
    }
    # Input power tops out near 6 MW, so 5.5 MW radiated is a collapse or a broken record.
    # The 2.5 MW this replaces caught ICRF H-modes radiating 3 MW of 5 MW in.
    # Over the same 990 shots it fires in 48 and costs 28 s of kept time, 2 percent.
    transient_filter = {
        "power_ohm": 5.0e6,
        "power_radiated": 5.5e6,
    }
    end_margin = 0.02
    shot_blacklist = []
    # Shots radiate a median 25 percent of their heating power (it11 store).
    # The lowest live bolometer record, 1160928005, sits at 1.9 percent.
    # The two shots with no record at all are caught by the all-NaN check.
    min_radiated_fraction = 0.01
    # More radiated than put in. The highest in a 40-shot sample is 0.87.
    max_radiated_fraction = 1.0
    # Thomson n_e against the interferometer. Large disagreement indicates TS miscalibration.
    # 1160527001 and 1160527002 are 0.69, the next lowest shot is 0.76, the highest at 1.16
    density_ratio_bounds = (0.72, 1.3)

    # GP fit staging knobs
    # TS channels whose Te is dropped from every shot, their ne is kept.
    # Core channel 3 (Z = +0.082 m, rho ~0.29) reads Te 1.58x the ECE at the same rho,
    # where every other core channel reads 1.02-1.17x (26 shots with low-field-side ECE).
    te_faulty_channels = [3]
    # Minimum valid (rho_tor_norm, value) pairs required per timestep to run the GP fit,
    # compared against the channel count AFTER the per-shot quality screens (_drop_broken_channels).
    fit_min_points = 8  # Most have 10 active, but we're dropping one, so this tolerates one additional drop
    fit_scale_per_slice = True
    # Hyperparameter bounds for the GP fit, per variable.
    # Amplitude floor 1, the data scale since each slice is normalized to a max of 1.
    # Below it the fits fell into a low-amplitude basin under peaked cores (median var 0.29 against 1.8),
    # 53 Te slices of tuning iteration 7 with the axis under 0.7 of the core data.
    # On the probe shots the floor took those 10 to 1, Te flicker 39 to 20, and ne double dips 53 to 31.
    # ne core scale ceiling 1: a quarter of ne fits go past 0.7 given the room, and the ne double dips halve.
    # Te keeps the 0.7 ceiling, and a 0.5 or 0.6 one changes nothing.
    # Probe tables in scratch/agent/tune_fitting/probes/findings.md, "Hyperparameter ranges".
    fit_bounds = {
        "te": FitBounds(l1_min=0.35, var_min=1.0),
        "ne": FitBounds(l1_min=0.55, l1_max=1.0, var_min=1.0),
    }

    def get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from C-Mod SQL database.

        Returns:
            Shot numbers to process, filtered to days with blessed Thomson data.
        """
        data = summary(
            summary_table="summary",
            ipmax=self.min_filter["ip"],
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

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from MDSplus, through disruption-py, into standardized signals.

        Four retrievals rather than one, because their native timebases differ.
        The fast diagnostics (Ip, B0, density, powers) are averaged over each 1 ms grid step from their own faster native data.
        The EFIT reconstruction and Thomson scattering are snapped onto the grid without interpolation,
        so grid times between their real samples hold NaN.
        The EFIT 0D signals (stored energy, shaping) are held forward from each reconstruction,
        for at least EQUILIBRIUM_HOLD_FLOOR (see _get_efit0d_dataset).
        fresh_equilibrium in the stores marks the grid times a usable reconstruction landed on (usable_reconstructions),
        while the EFIT 0D signals take every reconstruction.

        All four open the EFIT tree, at least for their timebase,
        so a shot reads everything from one tree.
        The trees of efit_trees are tried in order, and the first that serves all four is kept
        and recorded as the efit_tree attribute.
        A tree fails when a retrieval comes back empty (a missing tree fails disruption-py's setup)
        or the reconstruction has no cocos attribute (a missing node, see CmodEfitMethods.get_geqdsk_parameters).
        A shot the later checks or the unprocessed filter reject does not try another tree,
        since those rejections have causes the tree does not change.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        efit_trees = self.settings.efit_trees
        tree_failures = []
        for i_tree, efit_tree in enumerate(efit_trees):
            datasets, reason = _read_with_efit_tree(shot, efit_tree)
            if reason is None:
                break
            tree_failures.append(f"{efit_tree}: {reason}")
            if i_tree + 1 < len(efit_trees):
                logger.warning(
                    f"Shot {shot}: EFIT tree {efit_tree} failed ({reason}), "
                    f"trying {efit_trees[i_tree + 1]}"
                )
        else:
            failure = "No usable EFIT tree. " + " ".join(tree_failures)
            logger.warning(f"Shot {shot}: {failure} Skipping.")
            self.record_failed_shot(shot, failure)
            return None

        ds_merged = xr.merge(datasets, compat="no_conflicts", join="outer")
        # merge keeps the first dataset's attributes only, and COCOS is in the EFIT one
        for ds in datasets:
            if "cocos" in ds.attrs:
                ds_merged.attrs["cocos"] = ds.attrs["cocos"]
        ds_merged.attrs["efit_tree"] = efit_tree
        ds_standardized = self.standardize_signal_names(ds_merged)
        if ds_standardized is None:
            logger.warning(f"Shot {shot} is missing critical signals. Skipping.")
            self.record_failed_shot(shot, "Missing critical signals.")
        return ds_standardized

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals to IMAS-like names, keeping freeqdsk names for EFIT signals.

        All signals stay in SI units.
        ip and b0 keep the sign of their source, the GEQDSK signals carry the COCOS convention.
        b0 is btor as read, the vacuum field at R0, which the r0 attribute carries.

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
            "betan": "beta_tor_norm",
            "p_ohm": "power_ohm",
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

        # Per shot like cocos, the stack stage stores it as the r0 variable
        ds.attrs["r0"] = R0

        # C-Mod has no NBI or ECH, zero where ip is valid
        if "ip" in ds:
            ds["power_nbi"] = ds["ip"] * 0.0
            ds["power_nbi"].attrs = {}
            ds["power_ec"] = ds["ip"] * 0.0
            ds["power_ec"].attrs = {}
        apply_signal_attrs(ds, self.signal_attrs)

        return ds

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        1: Maps the TS channels onto rho_tor_norm through magnetics-only EFIT
        2: convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: C-Mod channel quality screens and error floors, calibrated in those units.
           A reading a screen drops in Te or ne takes the other's reading of that channel with it (cleaning.drop_in_both).
           The Te of te_faulty_channels is dropped before any screen, so their ne stays.
        4: TODO: optionally correct density with interferometry

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ts_times, rho_tor_norm = map_ts_channels_to_rho_tor_norm(
            ds, self.settings.sol_extension
        )
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        ds_shot = ds.squeeze("shot", drop=True)
        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(ds_shot, ts_times)
        # Before the raw validity below, so drop_in_both never takes the sound ne of a faulty Te channel
        te_y[:, self.te_faulty_channels] = np.nan
        # What each variable offers before the screens, so what they drop can be coupled
        te_valid_raw = np.isfinite(te_y) & np.isfinite(te_err)
        ne_valid_raw = np.isfinite(ne_y) & np.isfinite(ne_err)

        # Drop density channels too uncertain to constrain the fit
        # (error > 1e20 m^-3). These are typically bad edge/SOL channels.
        # Seen on shot 1160609014: a ne~4, err~2 channel past the separatrix
        # (rho~1.05) drove a spike to ne~19 at rho=1.0.
        ne_y = np.where(ne_err > 1.0, np.nan, ne_y)
        # Drop density points past the separatrix (rho_tor_norm>1.0) reading > 0.9e20:
        # SOL density is low out there, so such a point is a bad channel
        ne_y = np.where((rho_tor_norm > 1.0) & (ne_y > 0.9), np.nan, ne_y)

        # If data or error bar is incredibly small, set to NaN since it is
        # probably bad data. At this point ne is in 1e20 m^-3 and Te in keV.
        te_y = np.where(te_y < 0.001, np.nan, te_y)
        te_err = np.where(te_err < 0.001, np.nan, te_err)
        ne_y = np.where(ne_y < 0.001, np.nan, ne_y)
        ne_err = np.where(ne_err < 0.001, np.nan, ne_err)

        # Near the magnetic axis, Te this low is not physically real
        core_problem = (rho_tor_norm >= 0.0) & (rho_tor_norm < 0.4) & (te_y < 0.4)
        te_y = np.where(core_problem, np.nan, te_y)

        # A reading far under both its rho neighbours is probably a dead channel
        te_dips = relative_dips(rho_tor_norm, te_y)
        ne_dips = relative_dips(rho_tor_norm, ne_y)
        te_y = np.where(te_dips, np.nan, te_y)
        ne_y = np.where(ne_dips, np.nan, ne_y)
        if te_dips.any() or ne_dips.any():
            logger.info(
                f"Shot {shot}: dropped {int(te_dips.sum())} te and {int(ne_dips.sum())} ne readings "
                "under 0.35x both rho neighbours"
            )

        # Error floors.
        # Sometimes C-Mod TS has extremely tiny error bars which I don't think are real.
        # Hughes et al., RSI 72, 1107 (2001) quote 10-20 percent systematic errors in Te and ne,
        # and an edge Te range starting at 15 eV.
        # Te: 15 percent of the value with a 15 eV absolute floor.
        # ne: Floor at 10 percent of the value with a 1e18/m3 absolute floor.
        te_err = np.maximum(te_err, np.maximum(0.15 * np.abs(te_y), 0.015))
        ne_err = np.maximum(ne_err, np.maximum(0.10 * np.abs(ne_y), 0.01))

        # After the floors: the persistence screen must see the same errors
        # the fit will (its thresholds are calibrated on them).
        te_y = _drop_broken_channels("te", rho_tor_norm, te_y, te_err)
        ne_y = _drop_broken_channels("ne", rho_tor_norm, ne_y, ne_err)

        te_valid = np.isfinite(te_y) & np.isfinite(te_err)
        ne_valid = np.isfinite(ne_y) & np.isfinite(ne_err)
        te_y, ne_y, n_te_taken, n_ne_taken = drop_in_both(
            te_y, ne_y, te_valid_raw & ~te_valid, ne_valid_raw & ~ne_valid
        )
        if n_te_taken or n_ne_taken:
            logger.info(
                f"Shot {shot}: the C-Mod screens took {n_te_taken} te and {n_ne_taken} ne "
                "readings along with the other variable's drops"
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
        """Split the fit-plot channels into the core and edge TS systems.

        Args:
            shot: Shot number being plotted.
            fit_input: The shot's staged fit input (unused, the split is fixed per channel).

        Returns:
            (mask, color, label) triples for the two Thomson arrays.
        """
        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds:
            ts_array = ds["ts_array"].values
        return [
            (ts_array == "core", "tab:blue", "core TS"),
            (ts_array == "edge", "tab:orange", "edge TS"),
        ]

    def fit_plot_dropped_readings(self, shot: int) -> tuple | None:
        """Read the Te of te_faulty_channels, which staging drops from every fit, for the fit plots.

        They map onto rho_tor_norm as in prepare_fit_input, and keep their raw errors.

        Args:
            shot: Shot number being plotted.

        Returns:
            (times, rho_tor_norm, {"te": (y, err)}), see DataWorkflow.fit_plot_dropped_readings.
        """
        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds:
            ts_times, rho_tor_norm = map_ts_channels_to_rho_tor_norm(
                ds, self.settings.sol_extension
            )
            ds_shot = ds.squeeze("shot", drop=True)
            te_y, te_err, _, _ = ts_channel_fit_rows(ds_shot, ts_times)
        faulty = self.te_faulty_channels
        return (
            ts_times,
            rho_tor_norm[:, faulty],
            {"te": (te_y[:, faulty], te_err[:, faulty])},
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

    Per slice, each channel with both rho_tor_norm neighbors finite gets
    z = (y - neighbor_mean) / combined sigma, and a channel is dropped when
    |median z| >= 3.5 with >= 90 percent of slices on the same side, over >= 10 slices.

    Args:
        var_name: Variable name for the log line.
        data_x: (n_t, n_ch) channel rho_tor_norm positions.
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


def _read_with_efit_tree(
    shot: int, efit_tree: str
) -> tuple[list[xr.Dataset], str | None]:
    """Run the four retrievals of one shot against one EFIT tree.

    Stops at the first retrieval that fails, the rest would read the same tree.

    Args:
        shot: Shot number to read.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        (datasets, reason): the fast, EFIT 0D, EFIT and Thomson datasets and None,
        or no datasets and why the tree failed.
    """
    datasets = []
    for name, getter in (
        ("fast", _get_fast_dataset),
        ("efit0d", _get_efit0d_dataset),
        ("efit", _get_efit_dataset),
        ("thomson", _get_thomson_dataset),
    ):
        ds = getter(shot, efit_tree)
        if ds is None:
            return [], f"missing retrievable {name} data."
        # A missing GEQDSK node leaves NaN columns and no COCOS number
        if name == "efit" and "cocos" not in ds.attrs:
            return [], "no GEQDSK reconstruction."
        datasets.append(ds)
    return datasets, None


def _get_fast_dataset(shot: int, efit_tree: str) -> xr.Dataset | None:
    """Retrieve the fast-diagnostic 0D signals on the uniform 1 kHz grid.

    Only signals sampled faster than the grid belong here
    (magnetics, TCI, bolometry, RF and LH power),
    and each grid time takes the mean of the preceding millisecond (signal_on_grid).
    The EFIT-derived 0D signals are in _get_efit0d_dataset instead.
    p_ohm stays here because the fast loop voltage and Ip set its time resolution.
    Its EFIT li and R are held forward from the last reconstruction,
    and it is NaN outside the EFIT time range (CmodPowerMethods.get_ohmic_power).
    Only the custom methods run, selected by name.
    Selected by column, the disruption-py built-ins serving the same columns would run too,
    and they interpolate.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with the fast 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    fast_methods = [
        "get_plasma_current",  # ip
        "get_toroidal_field",  # bt, the vacuum field at R0
        "get_line_average_density",  # n_e [m^-3]
        # Power sources and sinks
        "get_ohmic_power",  # p_ohm
        "get_radiated_power",  # p_rad
        # p_icrf, p_lh (lower hybrid heating on C-Mod, NOT the L-H threshold power as on TCV)
        "get_heating_powers",
    ]

    retrieval_settings = RetrievalSettings(
        run_methods=fast_methods,
        time_setting=UniformTimeSetting(),
        efit_nickname_setting=efit_tree,
        only_requested_columns=False,
        custom_physics_methods=[CmodPlasmaMethods, CmodPowerMethods],
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


def _get_efit0d_dataset(shot: int, efit_tree: str) -> xr.Dataset | None:
    """Retrieve the EFIT-derived 0D signals and place them on the 1 kHz grid.

    time_setting="efit" makes params.times the EFIT tree's own timebase,
    so the final interp1 in the disruption-py methods is an identity.
    A tree on the grid's cadence is snapped onto it like the equilibrium and Thomson,
    so its 0D and 2D sit on the same grid times.
    A slow tree (see SLOW_EFIT_PERIOD) is placed straight from its own times.
    Either way each reconstruction is then held forward (signal_on_grid) for at least EQUILIBRIUM_HOLD_FLOOR,
    so a few missing reconstructions do not cut the shot, and none draws on a later reconstruction.
    Grid times outside the tree's time range hold NaN.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with the EFIT 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    efit0d_signals = [
        "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
        "betan",  # Normalized beta, EFIT's own node (CmodAeqdskMethods.get_normalized_beta)
        "a_minor",  # Plasma minor radius
        "kappa",  # Plasma elongation
        "tritop",  # Top triangularity
        "tribot",  # Bottom triangularity
        "rout",  # Geometric major radius [m]
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=efit0d_signals,
        time_setting="efit",
        efit_nickname_setting=efit_tree,
        only_requested_columns=True,
        custom_physics_methods=[
            CmodAeqdskMethods.get_geometric_major_radius,
            CmodAeqdskMethods.get_normalized_beta,
        ],
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
    efit_times = result["time"].values
    timebase = make_uniform_1kHz_timebase(float(efit_times.max()))
    slow_tree = (
        efit_times.size > 1 and float(np.median(np.diff(efit_times))) > SLOW_EFIT_PERIOD
    )
    if slow_tree:
        result = _hold_onto_grid(result, timebase)
    else:
        result_snapped = snap_to_grid(result, timebase)
        result = _hold_onto_grid(result_snapped, timebase)
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _hold_onto_grid(ds: xr.Dataset, grid_times: np.ndarray) -> xr.Dataset:
    """Hold an EFIT retrieval (dim 'idx', 'time'/'shot' coords) forward onto grid_times, at least EQUILIBRIUM_HOLD_FLOOR.

    Args:
        ds: Retrieval with dim 'idx' and 'time'/'shot' coords, 1D signals only.
        grid_times: Uniform timebase to hold onto [s].

    Returns:
        The signals on grid_times, laid out like snap_to_grid's output,
        NaN where no sample is held.
    """
    source_times = ds["time"].values
    shot_id = ds["shot"].values[0]
    data_vars = {}
    for name, variable in ds.data_vars.items():
        values_on_grid = signal_on_grid(
            source_times, variable.values, grid_times, EQUILIBRIUM_HOLD_FLOOR
        )
        data_vars[name] = ("idx", values_on_grid, variable.attrs)
    coords = {
        "time": ("idx", grid_times),
        "shot": ("idx", np.repeat(shot_id, grid_times.size)),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


def _get_efit_dataset(shot: int, efit_tree: str) -> xr.Dataset | None:
    """Retrieve EFIT dataset for the given shot.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with GEQDSK signals for the given shot,
        or None if retrieval returned no data.
    """
    settings = RetrievalSettings(
        run_methods=["get_geqdsk_parameters"],
        efit_nickname_setting=efit_tree,
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


def _get_thomson_dataset(shot: int, efit_tree: str) -> xr.Dataset | None:
    """Retrieve Thomson scattering data for the given shot.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with Thomson channel signals snapped to the uniform 1 kHz grid,
        or None if retrieval returned no data.
    """
    retrieval_settings = RetrievalSettings(
        run_methods=["get_thomson_channels"],
        efit_nickname_setting=efit_tree,
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
