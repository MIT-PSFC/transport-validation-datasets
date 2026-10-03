"""TCV data workflow, built from the DEFUSE exports and the LIUQE reconstructions of the MEQ databases.

- DEFUSE, one MATLAB v7.3 export per shot: the 0D signals and the raw Thomson channels.
- MEQ databases: the LIUQE reconstructions, as the full GEQDSK block (sources.liuqe_geqdsk_dataset),
  whose flux map also places the Thomson channels in rho_tor_norm.
DEFUSE's own spline fits of the profiles are not read, the channels are GP fit here like C-Mod's and MAST's.
Only the shots with both a DEFUSE export and a MEQ database are built.
TCV data has no release permission, so the datasets stop at the internal store (publishable False).
Both sources sit on the PSFC NFS (TCVSettings), so the unprocessed stage runs where that is mounted.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from transport_validation_datasets.cleaning import (
    branch_disagreement_errors,
    drop_in_both,
    low_side_channels,
    persistently_low_channels,
)
from transport_validation_datasets.gp_fitting.batch_io import FitBounds, ShotFitInput
from transport_validation_datasets.machine.generic import (
    EQUILIBRIUM_HOLD_FLOOR,
    MU0,
    channel_fit_rows,
    channel_rows_at_times,
    make_uniform_1kHz_timebase,
    map_ts_channels_to_rho_tor_norm,
    ohmic_power,
    signal_on_grid,
    smoothed_power,
    snap_to_grid,
    ts_channel_dataset,
)
from transport_validation_datasets.machine.tcv.sources import (
    DefuseSignal,
    DefuseThomson,
    defuse_path,
    find_tcv_shots,
    liuqe_geqdsk_dataset,
    meqdb_path,
    read_defuse_signals,
    read_defuse_thomson,
    read_liuqe,
)
from transport_validation_datasets.store_schema import apply_signal_attrs
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# Standardized name -> DEFUSE signal, SI in DEFUSE. ip and b0 keep their source sign (negative in the usual TCV configuration).
ZERO_D_SOURCES = {
    "ip": "I_P",
    "b0": "BZERO",
    "energy_mhd": "Wtot",
    "n_e_line_average": "NEavg",
    "minor_radius": "a_minor",
    "geometric_axis_r": "R_geom",
    "elongation": "KAPPA",
    "triangularity_upper": "DELTA_TOP",
    "triangularity_lower": "DELTA_BOTTOM",
    "power_radiated": "PradTot",
}
# Heating power -> DEFUSE signals summed into it, in MW.
# A system a shot does not have is absent from its export, or an empty placeholder, and counts as zero.
HEATING_SOURCES_MW = {
    "power_nbi": ("NBI", "NBI2"),
    "power_ec": ("ECRH",),
}

# DEFUSE signals of the LIUQE reconstruction, BZERO (LIUQE rBt / r0) among them.
# Each is held for at least EQUILIBRIUM_HOLD_FLOOR, so a few missing reconstructions are bridged.
LIUQE_SOURCES = (
    "Wtot",
    "VOL",
    "a_minor",
    "R_geom",
    "KAPPA",
    "DELTA_TOP",
    "DELTA_BOTTOM",
    "LI",
    "BZERO",
)
# PradTot follows the Thomson cadence (~17 ms) but often skips one or two samples (33-50 ms steps),
# or comes in bursts 50 ms apart (61056).
# The 1.5-step hold left gaps that cut ~10 percent of the kept time, and 60 ms bridges both.
# A causal zero-order hold, not the 50 ms triangle of smoothed_power, since PradTot already reads smooth.
PRAD_TOT_HOLD_FLOOR_S = 60e-3
# DEFUSE signal -> shortest hold [s], 0 for the rest
HOLD_FLOORS_S = {
    **dict.fromkeys(LIUQE_SOURCES, EQUILIBRIUM_HOLD_FLOOR),
    "PradTot": PRAD_TOT_HOLD_FLOOR_S,
}

# DEFUSE signals power_ohm is computed from (ohmic_power), DEFUSE POHM has no documented definition
OHMIC_POWER_SOURCES = ("I_P", "Vloop", "LI", "R_geom")
# DEFUSE signals beta_tor_norm is computed from (_normalized_beta) in the B_geo convention of every store.
# DEFUSE BETAN normalizes beta_tor by the volume-averaged vacuum field and multiplies by |BZERO| at r0,
# which reads a median 5.6 percent below it.
NORMALIZED_BETA_SOURCES = ("Wtot", "VOL", "a_minor", "R_geom", "BZERO", "I_P")
# DEFUSE Vloop has the opposite sign convention to I_P:
# Ip * Vloop is negative at flat-top on all 39 shots checked, of both current polarities
DEFUSE_VLOOP_SIGN = -1.0
# Every DEFUSE 0D signal a shot needs
REQUIRED_DEFUSE_SIGNALS = tuple(
    dict.fromkeys(
        [*ZERO_D_SOURCES.values(), *OHMIC_POWER_SOURCES, *NORMALIZED_BETA_SOURCES]
    )
)
# Every DEFUSE 0D signal read from an export, the heating systems optional
DEFUSE_SIGNALS = (
    *REQUIRED_DEFUSE_SIGNALS,
    *(raw_name for raw_names in HEATING_SOURCES_MW.values() for raw_name in raw_names),
)

# The timebase ends at the last time the plasma current magnitude exceeds this [A]
IP_TIMEBASE_MIN_A = 50e3

# Fringe jumps of the FIR interferometer, removed from the raw NEavg samples (_remove_fringe_jumps).
# Smallest level shift read as a fringe jump [m^-3]. The clean jumps in 185 shots are 1.1-2.5e19.
FRINGE_JUMP_MIN_M3 = 1e19
# A fringe jump completes within a few raw samples, and no real density change is that fast.
# So a jump is looked for between the medians of this long on either side of each sample [s].
FRINGE_SHARP_WINDOW_S = 0.25e-3
# Sharp shifts closer together than this are one episode, such as a dropout and its recovery [s]
FRINGE_EPISODE_GAP_S = 5e-3
# The levels on either side of an episode are the medians from FRINGE_SETTLE_S to FRINGE_LEVEL_WINDOW_S away from it,
# so a spike decaying back to the level it left is not read as a jump [s]
FRINGE_SETTLE_S = 2e-3
FRINGE_LEVEL_WINDOW_S = 10e-3
# An episode longer than FRINGE_BURST_WINDOW_S, or this many corrected episodes within it,
# means the interferometer has lost count, and the rest of the record is cut
FRINGE_BURST_EPISODES = 3
FRINGE_BURST_WINDOW_S = 0.05

# Thomson channel screens and error floors of prepare_fit_input, in the fit units (Te [keV], ne [1e20 m^-3]).
# A reading whose error exceeds this many times its value constrains nothing.
MAX_RELATIVE_ERROR = 1.0
# Error floors, fractional with an absolute floor, the TCV Thomson calibration is good to a few percent.
TE_ERROR_FLOOR_FRACTION = 0.05
TE_ERROR_FLOOR_KEV = 0.01
NE_ERROR_FLOOR_FRACTION = 0.05
NE_ERROR_FLOOR_1E20 = 0.01
# Channels past this sit in the far SOL (DEFUSE reaches rho_pol 1.65),
# where the mapping through a magnetics-only reconstruction is not trustworthy and the value anchors take over.
MAX_FIT_RHO_TOR_NORM = 1.1

# Per-variable attributes. The store signals take their units and refs from STORE_SIGNAL_ATTRS.
SIGNAL_ATTRS = {
    "ip": {"description": "Measured plasma current, signed (DEFUSE I_P)"},
    "b0": {
        "description": "Vacuum toroidal field at r0, signed, DEFUSE BZERO (LIUQE rBt / r0)",
    },
    "r0": {"description": "Reference major radius b0 is given at, LIUQE's r0"},
    "energy_mhd": {"description": "Stored energy on the LIUQE timebase (DEFUSE Wtot)"},
    "beta_tor_norm": {
        "description": (
            "Normalized toroidal beta with B_geo, 100 beta_tor a B_geo / Ip[MA] with beta_tor = 2 mu0 <p> / B_geo^2, "
            "<p> = 2 Wtot / (3 VOL) and B_geo = |BZERO| r0 / R_geom (DEFUSE Wtot, VOL, a_minor, R_geom, BZERO, I_P), "
            "not DEFUSE BETAN"
        ),
    },
    "n_e_line_average": {
        "description": (
            "Line-averaged electron density from the FIR interferometer (DEFUSE NEavg), "
            "fringe jumps removed from the raw samples (non-causal), NaN from where the interferometer lost count"
        ),
    },
    "minor_radius": {
        "description": "Minor radius of the plasma boundary, LIUQE (DEFUSE a_minor)",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary, LIUQE (DEFUSE R_geom)",
    },
    "elongation": {
        "description": "Elongation of the plasma boundary, LIUQE (DEFUSE KAPPA)",
    },
    "triangularity_upper": {
        "description": "Upper triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_TOP)",
    },
    "triangularity_lower": {
        "description": "Lower triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_BOTTOM)",
    },
    "power_ohm": {
        "description": (
            "Ohmic heating power, Ip * V_loop minus the rate of change of the internal poloidal magnetic energy "
            "mu0 R_geo li Ip^2 / 4 (DEFUSE I_P, Vloop, LI, R_geom, backward difference), "
            "smoothed by a centered 50 ms boxcar applied twice (non-causal), clipped at 0"
        ),
    },
    "power_radiated": {
        "description": (
            "Total radiated power including the divertor, bolometry (DEFUSE PradTot on its ~17 ms cadence, "
            "not smoothed further, held causally for at least 60 ms over skipped samples), clipped at 0"
        ),
    },
    "power_nbi": {
        "description": "Neutral beam power, summed over both beamlines (DEFUSE NBI + NBI2), zero where a beam is absent",
    },
    "power_ic": {"description": "Ion cyclotron heating power, zero (TCV has no ICRH)"},
    "power_lh": {
        "description": "Lower hybrid heating power, zero (TCV has no LHCD, DEFUSE P_LH is the L-H threshold power)",
    },
    "power_ec": {
        "description": "Electron cyclotron power, summed over gyrotrons (DEFUSE ECRH), zero where absent",
    },
    "ts_channel_r": {
        "description": "Major radius of the TS channel scattering volumes, the vertical laser chord (DEFUSE los/rchord)",
        "units": "m",
        "ref": "/thomson_scattering/channel(i1)/position/r",
    },
    "ts_channel_z": {
        "description": "Height of the TS channel scattering volumes (DEFUSE los/zchord)",
        "units": "m",
        "ref": "/thomson_scattering/channel(i1)/position/z",
    },
    "ts_channel_n_e": {
        "description": "Electron density measured by TS channels, calibrated to the FIR interferometer (DEFUSE Ne_rho raw)",
        "units": "m^-3",
        "ref": "/thomson_scattering/channel(i1)/n_e/data",
    },
    "ts_channel_n_e_error": {
        "description": "Electron density uncertainty of the TS channels (DEFUSE Ne_rho raw error_bar)",
        "units": "m^-3",
        "ref": "/thomson_scattering/channel(i1)/n_e/data_error_upper",
    },
    "ts_channel_t_e": {
        "description": "Electron temperature measured by TS channels (DEFUSE Te_rho raw)",
        "units": "eV",
        "ref": "/thomson_scattering/channel(i1)/t_e/data",
    },
    "ts_channel_t_e_error": {
        "description": "Electron temperature uncertainty of the TS channels (DEFUSE Te_rho raw error_bar)",
        "units": "eV",
        "ref": "/thomson_scattering/channel(i1)/t_e/data_error_upper",
    },
    "bcentr": {
        "description": (
            "Vacuum toroidal field at the GEQDSK reference radius rcentr, signed, LIUQE rBt / r0. "
            "The standardized b0 is the same field from DEFUSE BZERO, and r0 is rcentr."
        ),
    },
}


@dataclass(frozen=True)
class TCVSettings(DeviceSettings):
    """TCV settings, the [tcv] table of the config file.

    Attributes:
        defuse_dir: Directory of the DEFUSE exports, one TCVno{shot}.h5 per shot.
        meqdb_dir: Directory of the MEQ databases holding the LIUQE reconstructions, one TCV{shot}_meqdb.mat per shot.
    """

    defuse_dir: str = "/usr/local/mfe/ml_data_dump/TCV/DEFUSE/DEFUSE_DB/DB_mat/TCV"
    meqdb_dir: str = "/usr/local/mfe/ml_data_dump/TCV/meq_data/databases"


class TCVDataWorkflow(DataWorkflow):
    """TCV specific data workflow for creating and processing datasets."""

    settings_cls = TCVSettings
    signal_attrs = SIGNAL_ATTRS
    # No release permission for TCV data yet
    publishable = False
    # The vertical laser chord crosses the axis, the channels below it are one branch and those above the other
    fit_plot_branch_position = "ts_channel_z"
    fit_plot_branch_labels = ("TS below axis", "TS above axis")

    min_pulse_length = 0.5
    min_filter = {
        "ip": 5e4,
        "energy_mhd": 1e3,
        # A broken FIR record reads ~0 or negative, and Thomson calibrated to it reads ~0 too (70353, 70356).
        # The lowest real plasma in 280 shots is 3.4e18 (74082, where Thomson agrees).
        "n_e_line_average": 2e18,
    }
    max_filter = {
        # Bad interferometer data can pass an absolute density cap at low ip
        "greenwald_fraction": 2.0,
    }
    # P_oh as on DIII-D. In 246 shots only 75026 has a 5 ms P_rad peak above 3 MW (12 MW).
    transient_filter = {"power_ohm": 2e6, "power_radiated": 5e6}
    end_margin = 0.05
    # TCV bolometry reads a few percent of the input power or more, a dead bolometer far less
    min_radiated_fraction = 0.025
    # 5 of 246 shots radiate more than is put in, two by far (75026 4.1x with a 12 MW PradTot spike, 78926 1.7x),
    # while the 99th percentile is 1.17.
    max_radiated_fraction = 1.0
    # Thomson n_e is calibrated to the FIR, so this only catches a broken calibration of either
    density_ratio_bounds = (0.7, 1.3)
    shot_blacklist = []

    # GP fit staging knobs, C-Mod's to start with, each slice normalized to a max of 1
    fit_min_points = 10
    fit_scale_per_slice = True
    fit_bounds = {
        "te": FitBounds(l1_min=0.35, var_min=1.0),
        "ne": FitBounds(l1_min=0.55, l1_max=1.0, var_min=1.0),
    }

    def get_shotlist_from_source(self) -> list[int]:
        """Every shot with both a DEFUSE export and a LIUQE MEQ database.

        Returns:
            The shots, sorted.

        Raises:
            FileNotFoundError: If no shot has both.
        """
        shots = find_tcv_shots(self.settings.defuse_dir, self.settings.meqdb_dir)
        if not shots:
            raise FileNotFoundError(
                "No shot has both a DEFUSE export and a LIUQE MEQ database, see TCVSettings"
            )
        return shots

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from its DEFUSE export and MEQ database into standardized signals.

        The 0D signals are placed causally (signal_on_grid, _zero_d_dataset),
        while the LIUQE reconstructions and the Thomson slices are snapped to the nearest grid time
        without interpolation, since neither is meaningful interpolated.
        LIUQE reconstructs every millisecond and Thomson fires every ~17 ms.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        defuse_file = defuse_path(self.settings.defuse_dir, shot)
        meqdb_file = meqdb_path(self.settings.meqdb_dir, shot)
        if not (defuse_file.exists() and meqdb_file.exists()):
            self.record_failed_shot(shot, "No DEFUSE export or no MEQ database.")
            return None

        signals = read_defuse_signals(defuse_file, DEFUSE_SIGNALS)
        missing = [name for name in REQUIRED_DEFUSE_SIGNALS if name not in signals]
        if missing:
            self.record_failed_shot(shot, f"Missing DEFUSE signals {missing}.")
            return None
        ip = signals["I_P"]
        mask_ip_valid = np.abs(ip.values) > IP_TIMEBASE_MIN_A
        if not mask_ip_valid.any():
            self.record_failed_shot(
                shot, f"|I_P| never exceeds {IP_TIMEBASE_MIN_A:.0f} A."
            )
            return None
        ip_end_time = float(ip.time[mask_ip_valid].max())
        timebase = make_uniform_1kHz_timebase(ip_end_time)

        thomson = read_defuse_thomson(defuse_file)
        if thomson is None:
            self.record_failed_shot(shot, "No raw Thomson Te and ne in DEFUSE.")
            return None
        ds_thomson_native = _thomson_dataset(shot, thomson, timebase)
        has_te = np.isfinite(ds_thomson_native["ts_channel_t_e"].values).any()
        has_ne = np.isfinite(ds_thomson_native["ts_channel_n_e"].values).any()
        if not (has_te and has_ne):
            self.record_failed_shot(
                shot, "No valid Thomson Te or no valid ne reading in the shot."
            )
            return None

        liuqe = read_liuqe(meqdb_file)
        r0 = liuqe["r0"]
        ds_equilibrium_liuqe = liuqe_geqdsk_dataset(liuqe, shot)
        # The full struct is ~250 MB, only the block is kept
        del liuqe
        ds_equilibrium = snap_to_grid(ds_equilibrium_liuqe, timebase)
        ds_thomson = snap_to_grid(ds_thomson_native, timebase)
        ds_0d = _zero_d_dataset(shot, signals, r0, timebase)

        ds = xr.merge(
            [ds_0d, ds_equilibrium, ds_thomson], compat="no_conflicts", join="outer"
        )
        ds = ds.set_index(idx=["shot", "time"]).unstack("idx")
        ds.attrs = dict(ds_equilibrium.attrs)
        # Per shot like cocos, the stack stage stores it as the r0 variable
        ds.attrs["r0"] = r0
        apply_signal_attrs(ds, SIGNAL_ATTRS)
        return ds

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        1: Map the TS channels onto rho_tor_norm through the nearest LIUQE reconstruction
        2: Convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: Drop every channel whose Te or ne error exceeds MAX_RELATIVE_ERROR times its value, in both
        4: Drop, for the whole shot, the channels of each variable reading far under their neighbours (cleaning.persistently_low_channels)
        5: Floor the errors (TE_ and NE_ERROR_FLOOR_*)
        6: Inflate the errors where the branches below and above the axis disagree (cleaning.branch_disagreement_errors)
        7: Drop the channels past MAX_FIT_RHO_TOR_NORM

        The vertical chord at R = 0.9 m crosses every flux surface twice, below and above the axis.
        Te below the axis reads a few percent above Te above it (3-8 percent on four shots, ne 1-4 percent),
        so step 6 makes both branches consistent with a profile between them.
        Step 4 runs before it, so a broken channel does not inflate the errors of the good ones around it.
        The channels under the lower X-point are left unmapped by map_ts_channels_to_rho_tor_norm (private flux region).

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

        te_y, te_err, ne_y, ne_err = channel_fit_rows(ds_shot, ts_times)

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

        te_low = persistently_low_channels(rho_tor_norm, te_y)
        ne_low = persistently_low_channels(rho_tor_norm, ne_y)
        te_y[:, te_low] = np.nan
        ne_y[:, ne_low] = np.nan
        if te_low.any() or ne_low.any():
            logger.info(
                f"Shot {shot}: dropped persistently low te channels {np.flatnonzero(te_low).tolist()} "
                f"and ne channels {np.flatnonzero(ne_low).tolist()}"
            )

        te_floor = np.maximum(
            TE_ERROR_FLOOR_FRACTION * np.abs(te_y), TE_ERROR_FLOOR_KEV
        )
        ne_floor = np.maximum(
            NE_ERROR_FLOOR_FRACTION * np.abs(ne_y), NE_ERROR_FLOOR_1E20
        )
        te_err = np.maximum(te_err, te_floor)
        ne_err = np.maximum(ne_err, ne_floor)

        z_channel = channel_rows_at_times(ds_shot["ts_channel_z"], ts_times)
        below_axis = low_side_channels(rho_tor_norm, z_channel)
        te_err = branch_disagreement_errors(rho_tor_norm, te_y, te_err, below_axis)
        ne_err = branch_disagreement_errors(rho_tor_norm, ne_y, ne_err, below_axis)

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


def _thomson_dataset(
    shot: int, thomson: DefuseThomson, timebase: np.ndarray
) -> xr.Dataset:
    """Collect the raw Thomson channels on their laser timebase (ts_channel_dataset).

    DEFUSE's exclusion flags are negative values or errors, which ts_channel_dataset drops with the NaN.

    Args:
        shot: Shot number being read.
        thomson: The shot's raw Thomson channels.
        timebase: Uniform 1 kHz timebase, used to drop slices outside the shot [s].

    Returns:
        Dataset on dims ("idx", "ts_channel") with "time"/"shot" coords.
    """
    readings_shape = thomson.te.shape
    r_rows = np.broadcast_to(thomson.r_channel, readings_shape)
    z_rows = np.broadcast_to(thomson.z_channel, readings_shape)
    return ts_channel_dataset(
        shot,
        thomson.time,
        r_rows,
        z_rows,
        te=thomson.te,
        te_error=thomson.te_error,
        ne=thomson.ne,
        ne_error=thomson.ne_error,
        timebase=timebase,
    )


def _zero_d_dataset(
    shot: int, signals: dict[str, DefuseSignal], r0: float, timebase: np.ndarray
) -> xr.Dataset:
    """Place the DEFUSE 0D signals onto the timebase under standardized names.

    Every signal is placed causally (signal_on_grid), never interpolated, so no grid time draws on a later sample,
    the LIUQE signals and PradTot held for at least their HOLD_FLOORS_S.
    The fringe jumps of NEavg are removed from its raw samples first (_remove_fringe_jumps, non-causal),
    and power_ohm is smoothed non-causally (smoothed_power), as on every device.
    Plasma current and toroidal field keep their source sign.

    Args:
        shot: Shot number being read.
        signals: The DEFUSE 0D signals, every REQUIRED_DEFUSE_SIGNALS among them.
        r0: Major radius BZERO is the vacuum field at [m].
        timebase: Uniform 1 kHz timebase [s].

    Returns:
        Dataset of 0D signals on dim "idx", with "time" and "shot" coords.
    """
    density = signals[ZERO_D_SOURCES["n_e_line_average"]]
    density_corrected, cut_time = _remove_fringe_jumps(density.time, density.values)
    if cut_time is not None:
        logger.info(
            f"Shot {shot}: the FIR interferometer lost count, NEavg cut from {cut_time:.3f} s"
        )
    signals_corrected = {
        **signals,
        ZERO_D_SOURCES["n_e_line_average"]: DefuseSignal(
            time=density.time, values=density_corrected
        ),
    }
    placed = {
        name: signal_on_grid(
            signal.time, signal.values, timebase, HOLD_FLOORS_S.get(name, 0.0)
        )
        for name, signal in signals_corrected.items()
    }

    data = {
        store_name: placed[raw_name] for store_name, raw_name in ZERO_D_SOURCES.items()
    }
    data["power_ohm"] = _ohmic_power(timebase, placed)
    data["beta_tor_norm"] = _normalized_beta(placed, r0)
    for store_name, raw_names in HEATING_SOURCES_MW.items():
        power_MW = np.zeros(timebase.size)
        for raw_name in raw_names:
            if raw_name in placed:
                power_MW = power_MW + np.nan_to_num(placed[raw_name])
        data[store_name] = power_MW * 1e6
    # TCV has no ICRF or lower hybrid, zero where ip is valid
    data["power_ic"] = data["ip"] * 0.0
    data["power_lh"] = data["ip"] * 0.0

    return xr.Dataset(
        data_vars={name: ("idx", values) for name, values in data.items()},
        coords={
            "time": ("idx", timebase),
            "shot": ("idx", np.repeat(shot, timebase.size)),
        },
    )


def _ohmic_power(timebase: np.ndarray, placed: dict[str, np.ndarray]) -> np.ndarray:
    """Ohmic power Ip V_loop - dW_pol/dt (ohmic_power) from the DEFUSE signals on the timebase.

    Vloop is flipped onto the sign convention of I_P (DEFUSE_VLOOP_SIGN),
    and the result is smoothed non-causally (smoothed_power), as on C-Mod and MAST.

    Args:
        timebase: Uniform 1 kHz timebase [s].
        placed: DEFUSE signals on the timebase, by DEFUSE name.

    Returns:
        (n_t,) ohmic power [W].
    """
    v_loop = DEFUSE_VLOOP_SIGN * placed["Vloop"]
    p_ohm_raw = ohmic_power(
        timebase, placed["I_P"], v_loop, placed["LI"], placed["R_geom"]
    )
    time_steps = np.diff(timebase)
    dt = float(np.median(time_steps))
    return smoothed_power(p_ohm_raw, dt)


def _normalized_beta(placed: dict[str, np.ndarray], r0: float) -> np.ndarray:
    """Normalized toroidal beta with B_geo, from the LIUQE signals and I_P on the timebase.

    beta_tor = 2 mu0 <p> / B_geo^2 with the volume-averaged pressure <p> = 2 Wtot / (3 VOL),
    and beta_N = 100 beta_tor a B_geo / Ip[MA], the convention every store holds.
    B_geo = |BZERO| r0 / R_geom carries LIUQE's vacuum field at r0 out to the geometric axis.

    Args:
        placed: DEFUSE signals on the timebase, by DEFUSE name.
        r0: Major radius BZERO is the vacuum field at [m].

    Returns:
        (n_t,) normalized beta.
    """
    pressure_mean = 2.0 * placed["Wtot"] / (3.0 * placed["VOL"])
    b_center_magnitude = np.abs(placed["BZERO"])
    b_geo = b_center_magnitude * r0 / placed["R_geom"]
    beta_tor = 2.0 * MU0 * pressure_mean / b_geo**2
    ip_magnitude = np.abs(placed["I_P"])
    ip_magnitude_ma = ip_magnitude / 1e6
    return 100.0 * beta_tor * placed["a_minor"] * b_geo / ip_magnitude_ma


def _sharp_shift_samples(density: np.ndarray, n_sharp: int) -> np.ndarray:
    """Find the samples across which the density shifts sharply.

    A sample counts when the median of the next n_sharp samples differs from the median of the n_sharp up to it
    by FRINGE_JUMP_MIN_M3 or more.

    Args:
        density: (n,) NEavg samples [m^-3].
        n_sharp: Samples in each median window.

    Returns:
        The samples, in order.
    """
    sample_windows = sliding_window_view(density, n_sharp)
    window_medians = np.median(sample_windows, axis=1)
    # Shift across the boundary after sample k, for k from n_sharp - 1 to n - n_sharp - 1
    sharp_shift = window_medians[n_sharp:] - window_medians[:-n_sharp]
    sharp_shift_magnitude = np.abs(sharp_shift)
    idx_window_pair = np.flatnonzero(sharp_shift_magnitude >= FRINGE_JUMP_MIN_M3)
    return idx_window_pair + n_sharp - 1


def _fringe_episode_spans(
    sample_time: np.ndarray, idx_sharp: np.ndarray, n_sharp: int
) -> tuple[np.ndarray, np.ndarray]:
    """First and last sample of each episode: sharp shifts within FRINGE_EPISODE_GAP_S, with their sharp windows.

    Args:
        sample_time: (n,) sample times [s].
        idx_sharp: The sharp shifts (_sharp_shift_samples).
        n_sharp: Samples in each median window.

    Returns:
        (span_first, span_last): the first and last sample of every episode.
    """
    sharp_times = sample_time[idx_sharp]
    sharp_gaps = np.diff(sharp_times)
    mask_episode_start = np.r_[True, sharp_gaps > FRINGE_EPISODE_GAP_S]
    episode_start = np.flatnonzero(mask_episode_start)
    episode_stop = np.r_[episode_start[1:], idx_sharp.size]
    idx_first_shift = idx_sharp[episode_start]
    idx_last_shift = idx_sharp[episode_stop - 1]
    span_first = np.maximum(idx_first_shift - n_sharp + 1, 0)
    span_last = np.minimum(idx_last_shift + n_sharp, sample_time.size - 1)
    return span_first, span_last


def _level_samples(
    idx_settled: np.ndarray, idx_adjacent: np.ndarray, n_sharp: int
) -> np.ndarray:
    """The settled samples on one side of an episode, or the ones next to it when a neighbor leaves too few.

    Args:
        idx_settled: The settled samples on that side.
        idx_adjacent: The samples next to the episode on that side.
        n_sharp: Fewest settled samples to take.

    Returns:
        The samples the side's level is the median of.
    """
    if idx_settled.size >= n_sharp:
        return idx_settled
    return idx_adjacent


def _remove_fringe_jumps(
    sample_time: np.ndarray, density: np.ndarray
) -> tuple[np.ndarray, float | None]:
    """The raw NEavg samples with the interferometer fringe jumps removed, non-causally.

    A sharp shift is a sample after which the median of the next FRINGE_SHARP_WINDOW_S
    differs from the median of the FRINGE_SHARP_WINDOW_S up to it by FRINGE_JUMP_MIN_M3 or more.
    Sharp shifts within FRINGE_EPISODE_GAP_S of each other form one episode.
    When the settled levels on either side of an episode differ by FRINGE_JUMP_MIN_M3 or more,
    the difference is removed from everything after it.
    The samples inside every episode are replaced by a straight line between its edges.
    So a spike that decays back, or a dropout that recovers, only loses its inside.
    An episode longer than FRINGE_BURST_WINDOW_S, or FRINGE_BURST_EPISODES corrections within it,
    means the interferometer has lost count, and every sample from the start of the first such episode on is NaN.

    Args:
        sample_time: (n,) sorted, unique sample times [s].
        density: (n,) NEavg [m^-3].

    Returns:
        (n,) the corrected samples, and the time the record is cut from, None when it is not.
    """
    density_corrected = density.copy()
    sample_steps = np.diff(sample_time)
    sample_step = float(np.median(sample_steps))
    n_sharp_window = round(FRINGE_SHARP_WINDOW_S / sample_step)
    n_sharp = max(3, n_sharp_window)
    if density.size < 2 * n_sharp + 1:
        return density_corrected, None
    idx_sharp = _sharp_shift_samples(density, n_sharp)
    if idx_sharp.size == 0:
        return density_corrected, None
    span_first, span_last = _fringe_episode_spans(sample_time, idx_sharp, n_sharp)

    idx_samples = np.arange(density.size)
    corrected_episode_times = []
    lost_count_time = np.inf
    for i_episode in range(span_first.size):
        first = span_first[i_episode]
        last = span_last[i_episode]
        # The levels stop short of the neighboring episodes
        previous_last = span_last[i_episode - 1] if i_episode > 0 else -1
        next_first = (
            span_first[i_episode + 1]
            if i_episode + 1 < span_first.size
            else density.size
        )
        time_first = sample_time[first]
        time_last = sample_time[last]
        if time_last - time_first > FRINGE_BURST_WINDOW_S:
            lost_count_time = time_first
            break
        mask_before = (sample_time >= time_first - FRINGE_LEVEL_WINDOW_S) & (
            sample_time < time_first - FRINGE_SETTLE_S
        )
        mask_after = (sample_time > time_last + FRINGE_SETTLE_S) & (
            sample_time <= time_last + FRINGE_LEVEL_WINDOW_S
        )
        mask_before &= idx_samples > previous_last
        mask_after &= idx_samples < next_first
        adjacent_before_first = max(previous_last + 1, first - n_sharp)
        adjacent_after_stop = min(next_first, last + 1 + n_sharp)
        idx_adjacent_before = np.arange(adjacent_before_first, first)
        idx_adjacent_after = np.arange(last + 1, adjacent_after_stop)
        if idx_adjacent_before.size == 0 or idx_adjacent_after.size == 0:
            continue
        idx_settled_before = np.flatnonzero(mask_before)
        idx_settled_after = np.flatnonzero(mask_after)
        idx_before = _level_samples(idx_settled_before, idx_adjacent_before, n_sharp)
        idx_after = _level_samples(idx_settled_after, idx_adjacent_after, n_sharp)
        level_before = np.nanmedian(density_corrected[idx_before])
        level_after = np.nanmedian(density_corrected[idx_after])
        level_shift = level_after - level_before
        if np.abs(level_shift) >= FRINGE_JUMP_MIN_M3:
            density_corrected[last + 1 :] -= level_shift
            corrected_episode_times.append(time_first)

        # A straight line across the episode, between the samples next to it
        edge_before = np.nanmedian(density_corrected[idx_adjacent_before])
        edge_after = np.nanmedian(density_corrected[idx_adjacent_after])
        edge_times = [sample_time[first - 1], sample_time[last + 1]]
        span_times = sample_time[first : last + 1]
        density_corrected[first : last + 1] = np.interp(
            span_times, edge_times, [edge_before, edge_after]
        )

    episode_times = np.asarray(corrected_episode_times)
    for episode_time in episode_times:
        mask_burst = (episode_times >= episode_time) & (
            episode_times <= episode_time + FRINGE_BURST_WINDOW_S
        )
        if mask_burst.sum() >= FRINGE_BURST_EPISODES:
            lost_count_time = min(lost_count_time, episode_time)
            break
    if not np.isfinite(lost_count_time):
        return density_corrected, None
    density_corrected[sample_time >= lost_count_time] = np.nan
    return density_corrected, float(lost_count_time)
