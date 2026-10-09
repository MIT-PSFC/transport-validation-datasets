from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets import TIME_COORD
from transport_validation_datasets.cleaning import (
    DIP_RATIO,
    drop_broken_channels,
    drop_in_both,
    relative_dips,
)
from transport_validation_datasets.dispy_utils import (
    empty_result,
    passive_log_settings,
    summary,
)
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
    POWER_SMOOTHING_WINDOW,
    absent_heating_powers,
    channel_fit_rows,
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho_tor_norm,
    snap_to_grid,
    values_held_from_usable,
)
from transport_validation_datasets.store_schema import apply_signal_attrs
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# Attributes of the signals this module makes rather than reads with
# attributes attached (DataWorkflow.signal_attrs). Everything else carries
# disruption-py's, rewritten to the shared convention at the stack stage.
# The units and IMAS paths of the store signals come from store_schema.STORE_SIGNAL_ATTRS.
SIGNAL_ATTRS = {
    "ip": {
        "description": "Plasma current, magnetics ip (Rogowski coil), signed, mean of each 1 ms grid step",
    },
    "n_e_line_average": {
        "description": (
            "Line-averaged electron density, TCI chord 4 line integral nl_04 (mean of each 1 ms grid step) "
            "over its in-plasma length, the EFIT rco2v of that chord held from the last usable reconstruction"
        ),
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
        "description": "Reference major radius btor is quoted at, the fixed 0.66 m where EFIT quotes bcentr (RZERO)",
    },
    "beta_tor_norm": {
        "description": (
            "Normalized toroidal beta as IMAS defines it, 100 beta_tor aout |bcentr| / |cpasma| with beta_tor = 2 mu0 <p> / bcentr^2, "
            "<p> = 2 wplasm / (3 vout) and bcentr the vacuum field at rcencm = r0, "
            "not the EFIT betan node, which takes |btaxp|"
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

# Major radius the magnetics btor is quoted at, the store's r0 [m].
# The raw btor, with no pre-shot baseline subtracted, matches EFIT bcentr to 0.9991-0.9998 (median of 6 shots),
# and EFIT quotes bcentr at rcentr, stored as RZERO = 0.66 m.
R0 = 0.66

# C-Mod channel screens in the fit units, Te [keV] and ne [1e20 m^-3], see prepare_fit_input
NE_ERROR_MAX = 1.0
SOL_NE_MAX = 0.9
READING_MIN = 0.001
CORE_RHO_MAX = 0.4
CORE_TE_MIN = 0.4
TE_ERROR_FLOOR_FRACTION = 0.15
TE_ERROR_FLOOR = 0.015
NE_ERROR_FLOOR_FRACTION = 0.10
NE_ERROR_FLOOR = 0.01

# The core TS channel at this height [m] reads Te 1.58x the ECE at the same rho (~0.29),
# where every other core channel reads 1.02-1.17x (26 shots with low-field-side ECE).
# Its Te is dropped from every fit (_te_faulty_channels), its ne is kept.
TE_FAULTY_CHANNEL_Z = 0.082
# A channel sits at TE_FAULTY_CHANNEL_Z when its height is this close [m]
TS_CHANNEL_Z_TOL = 1e-3


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
        # 2.7 kJ keeps the ramp-ups, 3 s of kept time over 990 shots more than 3 kJ would
        "energy_mhd": 2.7e3,
        # A broken interferometer record, the lowest kept value in a 40-shot sample is 2.4e19
        "n_e_line_average": 1e19,
    }
    max_filter = {
        "greenwald_fraction": 2.0,
    }
    # Input power tops out near 6 MW, so 5.5 MW radiated is a collapse or a broken record.
    # Over 990 shots it fires in 48 and costs 28 s of kept time, 2 percent.
    transient_filter = {
        "power_ohm": 5.0e6,
        "power_radiated": 5.5e6,
    }
    # One smoothing window, so the smoothed powers never carry the current quench (POWER_SMOOTHING_WINDOW)
    end_margin = POWER_SMOOTHING_WINDOW
    shot_blacklist = []
    # Shots radiate a median 25 percent of their heating power.
    # The lowest live bolometer record, 1160928005, sits at 1.9 percent.
    # The two shots with no record at all are caught by the all-finite check of slice_filter_mask.
    min_radiated_fraction = 0.01
    # More radiated than put in. The highest in a 40-shot sample is 0.87.
    max_radiated_fraction = 1.0
    # Thomson n_e against the interferometer. Large disagreement indicates TS miscalibration.
    # 1160527001 and 1160527002 are 0.69, the next lowest shot is 0.76, the highest at 1.16
    density_ratio_bounds = (0.72, 1.3)

    # GP fit staging knobs
    # Minimum valid (rho_tor_norm, value) pairs required per timestep to run the GP fit,
    # compared against the channel count AFTER the per-shot quality screens (cleaning.drop_broken_channels).
    # Many C-Mod shots carry exactly 10 core channels and the faulty one's Te is always dropped,
    # so this tolerates one more bad channel.
    fit_min_points = 8
    # Hyperparameter bounds for the GP fit, per variable.
    # Amplitude floor 1, the data scale since each slice is normalized to a max of 1.
    # Below it the fits fall into a low-amplitude mean regression under peaked cores, with the axis under 0.7 of the core data.
    # ne core scale ceiling 1: a quarter of the ne fits go past 0.7 given the room, and the ne double dips halve.
    # Te keeps the 0.7 ceiling, a 0.5 or 0.6 one changes nothing.
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

        # Days with blessed TS data, email from J. Hughes 2025-12-19
        blessed_days = [
            1160503,
            1160527,
            1160621,
            1160628,
            1160630,
            1160708,
        ]
        # First and last day, both blessed
        blessed_day_ranges = [
            [1160607, 1160609],
            [1160712, 1160718],
            [1160803, 1160819],
            [1160823, 1160902],
            [1160908, 1160915],
            [1160919, 1160923],
            [1160927, 1160930],
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
        The EFIT 0D signals (stored energy, shaping) are snapped too,
        and the workflow holds them from the last usable reconstruction (hold_from_usable_reconstructions),
        so they change exactly where fresh_equilibrium in the stores is 1.

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
            "p_rad": "power_radiated",
            # summary/heating_current_drive
            "p_icrf": "power_ic",
            "p_lh": "power_lh",
            # equilibrium/time_slice/boundary
            "a_minor": "minor_radius",
            "kappa": "elongation",
            "tritop": "triangularity_upper",
            "tribot": "triangularity_lower",
            "rout": "geometric_axis_r",
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

        # C-Mod has no NBI or ECH
        ds = ds.assign(absent_heating_powers(ds["ip"], ("power_nbi", "power_ec")))
        apply_signal_attrs(ds, self.signal_attrs)

        return ds

    def add_equilibrium_signals(
        self, ds_standardized: xr.Dataset, reconstruction_usable: np.ndarray
    ) -> xr.Dataset:
        """Hold the reconstruction's signals (DataWorkflow.add_equilibrium_signals) and form n_e_line_average.

        n_e_line_average is the TCI chord 4 line integral tci_nl_04 over the chord's in-plasma length tci_chord_04,
        EFIT's rco2v, held from the last usable reconstruction like every signal taken from one (values_held_from_usable).

        Args:
            ds_standardized: One shot's standardized dataset on (shot, time), with its GEQDSK block. Modified in place.
            reconstruction_usable: (n_t,) True at the usable reconstructions (usable_reconstructions).

        Returns:
            The dataset with n_e_line_average in place of tci_nl_04 and tci_chord_04.
        """
        ds = super().add_equilibrium_signals(ds_standardized, reconstruction_usable)
        grid = np.asarray(ds[TIME_COORD].values, dtype=float)
        chord_length = ds["tci_chord_04"].transpose(..., TIME_COORD)
        chord_length_values = values_held_from_usable(
            chord_length.values, reconstruction_usable, grid
        )
        chord_length_held = chord_length.copy(data=chord_length_values)
        ds["n_e_line_average"] = ds["tci_nl_04"] / chord_length_held
        apply_signal_attrs(ds, self.signal_attrs)
        return ds.drop_vars(["tci_nl_04", "tci_chord_04"])

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        1: Map the TS channels onto rho_tor_norm through the magnetics-only EFIT
        2: Convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: C-Mod channel quality screens and error floors, calibrated in those units.
           A reading a screen drops in Te or ne takes the other's reading of that channel with it (cleaning.drop_in_both).
           The Te of _te_faulty_channels is dropped before any screen, so their ne stays,
           and cleaning.drop_broken_channels runs after, so a shot-long bias in one variable keeps the other.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has no Thomson slice to fit.
        """
        ts_times, rho_tor_norm = map_ts_channels_to_rho_tor_norm(
            ds, self.settings.sol_extension
        )
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        ds_shot = ds.squeeze("shot", drop=True)
        te_y, te_err, ne_y, ne_err = channel_fit_rows(ds_shot, ts_times)
        # Before the raw validity below, so drop_in_both never takes the sound ne of a faulty Te channel
        te_faulty = _te_faulty_channels(ds_shot)
        te_y[:, te_faulty] = np.nan
        # What each variable offers before the screens, so what they drop can be coupled
        te_valid_raw = np.isfinite(te_y) & np.isfinite(te_err)
        ne_valid_raw = np.isfinite(ne_y) & np.isfinite(ne_err)

        # Density channels too uncertain to constrain the fit, typically bad edge or SOL channels.
        # On 1160609014 a ne ~4, err ~2 channel past the separatrix drove a spike to ne ~19 at rho 1.0.
        ne_y = np.where(ne_err > NE_ERROR_MAX, np.nan, ne_y)
        # SOL density is low, so a point past the separatrix reading above SOL_NE_MAX is a bad channel
        ne_y = np.where((rho_tor_norm > 1.0) & (ne_y > SOL_NE_MAX), np.nan, ne_y)

        # Readings or error bars under READING_MIN are analysis artifacts:
        # Te pinned near 9 eV with a 0.02 to 0.05 eV error bar on 40 of 272,000 readings,
        # which the error floors below would otherwise keep at 15 eV.
        te_y = np.where(te_y < READING_MIN, np.nan, te_y)
        te_err = np.where(te_err < READING_MIN, np.nan, te_err)
        ne_y = np.where(ne_y < READING_MIN, np.nan, ne_y)
        ne_err = np.where(ne_err < READING_MIN, np.nan, ne_err)

        # Near the magnetic axis, Te this low is not physically real
        core_problem = (rho_tor_norm < CORE_RHO_MAX) & (te_y < CORE_TE_MIN)
        te_y = np.where(core_problem, np.nan, te_y)

        # A reading far under both its rho neighbours is probably a dead channel
        te_dips = relative_dips(rho_tor_norm, te_y)
        ne_dips = relative_dips(rho_tor_norm, ne_y)
        te_y = np.where(te_dips, np.nan, te_y)
        ne_y = np.where(ne_dips, np.nan, ne_y)
        if te_dips.any() or ne_dips.any():
            logger.info(
                f"Shot {shot}: dropped {int(te_dips.sum())} te and {int(ne_dips.sum())} ne readings "
                f"under {DIP_RATIO}x both rho neighbours"
            )

        # Error floors, since C-Mod TS error bars run far below the systematic errors.
        # Hughes et al., RSI 72, 1107 (2001) quote 10-20 percent systematic errors in Te and ne,
        # and an edge Te range starting at 15 eV.
        te_err_floor = np.maximum(TE_ERROR_FLOOR_FRACTION * te_y, TE_ERROR_FLOOR)
        te_err = np.maximum(te_err, te_err_floor)
        ne_err_floor = np.maximum(NE_ERROR_FLOOR_FRACTION * ne_y, NE_ERROR_FLOOR)
        ne_err = np.maximum(ne_err, ne_err_floor)

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

        # After drop_in_both: a channel biased in one variable all shot is a calibration fault of that variable,
        # like _te_faulty_channels, so its other variable stays.
        # After the floors: the persistence screen must see the errors the fit will.
        te_y = drop_broken_channels("te", rho_tor_norm, te_y, te_err)
        ne_y = drop_broken_channels("ne", rho_tor_norm, ne_y, ne_err)

        return ShotFitInput(
            x=rho_tor_norm,
            te_y=te_y,
            te_err=te_err,
            ne_y=ne_y,
            ne_err=ne_err,
            time=ts_times,
        )

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
        """Read the Te of _te_faulty_channels, which staging drops from every fit, for the fit plots.

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
            te_y, te_err, _, _ = channel_fit_rows(ds_shot, ts_times)
            te_faulty = _te_faulty_channels(ds_shot)
        return (
            ts_times,
            rho_tor_norm[:, te_faulty],
            {"te": (te_y[:, te_faulty], te_err[:, te_faulty])},
        )


def _te_faulty_channels(ds_shot: xr.Dataset) -> np.ndarray:
    """Find the channels whose Te every fit drops, the core channel at TE_FAULTY_CHANNEL_Z.

    Selected by position, since the channel count of the concatenated core and edge arrays varies between shots.

    Args:
        ds_shot: The shot's unprocessed dataset, shot dim squeezed out.

    Returns:
        (n_faulty,) channel indices along the ts_channel dim.
    """
    height = ds_shot["ts_channel_z"]
    dims_to_reduce = [dim for dim in height.dims if dim != "ts_channel"]
    channel_height = height.median(dim=dims_to_reduce, skipna=True).values
    ts_array = ds_shot["ts_array"].values
    at_height = np.abs(channel_height - TE_FAULTY_CHANNEL_Z) < TS_CHANNEL_Z_TOL
    return np.flatnonzero((ts_array == "core") & at_height)


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
    The EFIT-derived 0D signals are in _get_efit0d_dataset instead,
    and power_ohm comes from the GEQDSK block (DataWorkflow.add_equilibrium_signals).
    Avoiding disruption-py internals because they interpolate.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with the fast 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    fast_methods = [
        "get_plasma_current",  # ip
        "get_toroidal_field",  # bt, the vacuum field at 0.66 m
        "get_line_integral_density",  # tci_nl_04 [m^-2]
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
    if empty_result(result):
        return None
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_efit0d_dataset(shot: int, efit_tree: str) -> xr.Dataset | None:
    """Retrieve the EFIT-derived 0D signals and place them on the 1 kHz grid.

    time_setting="efit" makes params.times the EFIT tree's own timebase,
    so the final interp1 in the disruption-py methods is an identity.
    The tree is snapped onto the grid like the equilibrium and Thomson,
    so its 0D and 2D sit on the same grid times, NaN between them.
    DataWorkflow.make_unprocessed_data_files holds them from the last usable reconstruction.

    Args:
        shot: Shot number to retrieve data for.
        efit_tree: EFIT tree to read, see CModSettings.efit_trees.

    Returns:
        Dataset with the EFIT 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    efit0d_signals = [
        "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
        "beta_tor_norm",  # As IMAS defines it, from wplasm and vout
        "a_minor",  # Plasma minor radius
        "kappa",  # Plasma elongation
        "tritop",  # Top triangularity
        "tribot",  # Bottom triangularity
        "rout",  # Geometric major radius [m]
        "tci_chord_04",  # In-plasma length of TCI chord 4 [m], under n_e_line_average
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=efit0d_signals,
        time_setting="efit",
        efit_nickname_setting=efit_tree,
        only_requested_columns=True,
        custom_physics_methods=[CmodAeqdskMethods],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if empty_result(result):
        return None
    efit_times = result["time"].values
    timebase = make_uniform_1kHz_timebase(float(efit_times.max()))
    result = snap_to_grid(result, timebase)
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


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
    if empty_result(result):
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
    if empty_result(result):
        return None
    # Snap native ~20 Hz TS slices onto the uniform 1 kHz grid, no interpolation.
    # Grid times with no TS slice come back as NaN.
    timebase = make_uniform_1kHz_timebase(float(result["time"].values.max()))
    result = snap_to_grid(result, timebase)
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result
