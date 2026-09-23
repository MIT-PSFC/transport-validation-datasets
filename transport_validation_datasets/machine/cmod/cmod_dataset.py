from dataclasses import dataclass, field

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
    map_ts_channels_to_flux_coordinates,
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
            the shot range into eras. Empty by default (no overrides):
            EFIT21 was verified to exist at full 1 kHz cadence for every
            pre-2004 shot in the scenario list (2026-09-14 MDSplus audit),
            so nothing is routed to the sparse ANALYSIS tree anymore. In
            TOML: [cmod.efit_nickname_below] with entries like
            "1050204013" = "analysis" (keys are strings, TOML tables
            require it).
        efit_nickname: EFIT tree the equilibrium, the geometry signals, and
            the shot's 1 kHz timebase are read from, and so the one the
            Thomson channels are mapped onto rho with. A tree name (EFIT21,
            EFIT18, ...), or one of disruption-py's keys: "analysis" for the
            ANALYSIS tree, "disruption" for the disruption EFIT.
    """

    efit_nickname: str = "EFIT21"
    efit_nickname_below: dict[str, str] = field(default_factory=dict)


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

        Four retrievals rather than one, because their native timebases
        differ: the fast diagnostics (Ip, B0, density, powers) are sampled
        at the uniform 1 kHz grid times from their own faster-than-grid
        native data; the EFIT-derived 0D signals (stored energy, shaping)
        stay on the EFIT tree's native timebase; the EFIT reconstruction
        stays on its native timebase; Thomson scattering is native ~20 Hz.
        The EFIT and Thomson retrievals are snapped onto the uniform grid
        without interpolation, so grid times between real samples hold NaN,
        never fabricated values (exact relabeling for EFIT21, which is
        native 1 kHz).

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        datasets = []
        for name, getter in (
            ("fast", _get_fast_dataset),
            ("efit0d", _get_efit0d_dataset),
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

        1: Maps the TS channels onto rho (and psi_norm) through magnetics-only EFIT
        2: convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: C-Mod channel quality screens and error floors, calibrated in rho
        4: TODO: optionally correct density with interferometry

        psi_norm and qpsi are also staged (unused unless a batch is fit with
        FitBatch.fit_coordinate != "rho", see worker_akho.py's module
        docstring); the rho-based quality screens/error floors below are
        unaffected either way -- they run once here, before any per-method
        coordinate choice.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ts_times, rho, psi_norm = map_ts_channels_to_flux_coordinates(ds)
        if ts_times.size == 0:
            logger.warning(f"Shot {shot}: no Thomson slices to fit")
            return None

        ds_shot = ds.squeeze("shot", drop=True)
        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(ds_shot, ts_times)

        qpsi = None
        if "qpsi" in ds_shot:
            ts_mask = np.isin(ds_shot["time"].values, ts_times)
            qpsi_values = ds_shot["qpsi"].transpose("time", "psi_idx").values
            qpsi = np.asarray(qpsi_values, dtype=float)[ts_mask]

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
            x=rho,
            te_y=te_y,
            te_err=te_err,
            ne_y=ne_y,
            ne_err=ne_err,
            time=ts_times,
            psi_norm=psi_norm,
            qpsi=qpsi,
        )
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no finite (rho, te, ne) channel data to fit")
            return None
        return fit_input

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
    """Retrieve the fast-diagnostic 0D signals on the uniform 1 kHz grid.

    Only signals whose native timebase is at least as fast as the grid
    belong here (magnetics, TCI, bolometry, RF power): sampling them at the
    1 kHz grid times only ever discards surplus resolution, it never
    fabricates values between real measurements. Every EFIT-derived 0D
    signal lives in _get_efit0d_dataset instead, on the EFIT tree's own
    timebase. p_oh stays here: its information rate is set by the fast loop
    voltage and Ip, and its EFIT-li inductance correction is exact at the
    ms-aligned EFIT21 slice times and NaN outside the EFIT range.

    Args:
        shot: Shot number to retrieve data for.
        efit_nickname: EFIT tree to read, see CModSettings.

    Returns:
        Dataset with the fast 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    fast_signals = [
        "ip",  # Plasma current
        "bt",  # On-axis magnetic field
        "n_e",  # Line average electron density [m^-3]
        # Power sources and sinks
        "p_oh",  # Ohmic heating power
        "p_rad",  # Bulk radiated heating power
        "p_icrf",  # ICRF heating power
        "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=fast_signals,
        time_setting=UniformTimeSetting(),
        efit_nickname_setting=efit_nickname,
        only_requested_columns=True,
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


def _get_efit0d_dataset(shot: int, efit_nickname: str) -> xr.Dataset | None:
    """Retrieve the EFIT-derived 0D signals on the native EFIT timebase.

    time_setting="efit" makes params.times the EFIT tree's own timebase, so
    the stock disruption-py methods' final interp1 onto params.times is an
    identity: no values are fabricated between reconstructions. The result
    is then snapped onto the uniform 1 kHz grid like the equilibrium and
    Thomson retrievals (nearest grid point, no interpolation, NaN where the
    tree has no slice) -- an exact relabeling for EFIT21 (native 1 kHz),
    and honest NaN gaps for any sparser tree.

    Args:
        shot: Shot number to retrieve data for.
        efit_nickname: EFIT tree to read, see CModSettings.

    Returns:
        Dataset with the EFIT 0D signals for the given shot, or None if
        retrieval returned no data.
    """
    efit0d_signals = [
        "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
        "beta_n",  # Normalized beta
        "a_minor",  # Plasma minor radius
        "kappa",  # Plasma elongation
        "tritop",  # Top triangularity
        "tribot",  # Bottom triangularity
        "rout",  # Geometric major radius [m]
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=efit0d_signals,
        time_setting="efit",
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
    # Native EFIT samples onto the uniform 1 kHz grid, no interpolation.
    timebase = make_uniform_1kHz_timebase(float(result["time"].values.max()))
    result = snap_to_grid(result, timebase)
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
