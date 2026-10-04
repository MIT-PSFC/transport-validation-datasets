"""Readers for the two TCV sources, and the LIUQE GEQDSK block.

DEFUSE exports are MATLAB v7.3 files (HDF5), read with h5py:
the 0D signals and the raw Thomson channels.
The LIUQE reconstructions of the MEQ databases are MATLAB v5 files, read with scipy.
Only what the workflow uses is read: a few MB of each DEFUSE file and the liuqe_data struct of each MEQ database.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import xarray as xr
from loguru import logger

from transport_validation_datasets.machine.generic import (
    cocos_from_signs,
    geqdsk_psi_n_grid,
    make_geqdsk_dataset,
)

DEFUSE_FILE_RE = re.compile(r"TCVno(\d+)\.h5")
MEQDB_FILE_RE = re.compile(r"TCV(\d+)_meqdb\.mat")

# DEFUSE signals stored one row per actuator, summed into one trace (ECRH: one row per gyrotron in newer shots)
SUMMED_ROW_SIGNALS = ("ECRH",)

# The DEFUSE profile groups whose raw subgroup holds the Thomson channels
DEFUSE_THOMSON_GROUPS = {"te": "Te_rho", "ne": "Ne_rho"}

# LIUQE's poloidal flux is per 2 pi radians (COCOS 17), the GEQDSK block's is per radian
LIUQE_FLUX_PER_RADIAN = 1.0 / (2.0 * np.pi)


@dataclass(frozen=True)
class DefuseSignal:
    """One DEFUSE 0D signal on its own timebase."""

    time: np.ndarray  # (n,) sorted, unique times [s]
    values: np.ndarray  # (n,)


@dataclass(frozen=True)
class DefuseThomson:
    """The raw Thomson channels of one DEFUSE export, unscreened.

    DEFUSE flags a reading it excluded with a negative error (-1 for Te)
    or a negative value and error (-firrat for ne), and leaves a missing one NaN.
    """

    time: np.ndarray  # (n_t,) laser times [s]
    r_channel: np.ndarray  # (n_ch,) major radius of each channel [m]
    z_channel: np.ndarray  # (n_ch,) height of each channel [m]
    te: np.ndarray  # (n_t, n_ch) [eV]
    te_error: np.ndarray  # (n_t, n_ch) [eV]
    ne: np.ndarray  # (n_t, n_ch) [m^-3], calibrated to the FIR interferometer
    ne_error: np.ndarray  # (n_t, n_ch) [m^-3]


def defuse_path(defuse_dir: Path | str, shot: int) -> Path:
    """Path of a shot's DEFUSE export.

    Args:
        defuse_dir: Directory of the DEFUSE exports.
        shot: Shot number.

    Returns:
        defuse_dir/TCVno{shot}.h5.
    """
    return Path(defuse_dir) / f"TCVno{shot}.h5"


def meqdb_path(meqdb_dir: Path | str, shot: int) -> Path:
    """Path of a shot's MEQ database.

    Args:
        meqdb_dir: Directory of the MEQ databases.
        shot: Shot number.

    Returns:
        meqdb_dir/TCV{shot}_meqdb.mat.
    """
    return Path(meqdb_dir) / f"TCV{shot}_meqdb.mat"


def _shots_in(directory: Path, file_re: re.Pattern) -> set[int]:
    shots = set()
    for path in directory.iterdir():
        match = file_re.fullmatch(path.name)
        if match:
            shots.add(int(match.group(1)))
    return shots


def find_tcv_shots(defuse_dir: Path | str, meqdb_dir: Path | str) -> list[int]:
    """Every shot with both a DEFUSE export and a LIUQE MEQ database.

    Args:
        defuse_dir: Directory of the DEFUSE exports.
        meqdb_dir: Directory of the MEQ databases.

    Returns:
        The shots, sorted.
    """
    defuse_shots = _shots_in(Path(defuse_dir), DEFUSE_FILE_RE)
    meqdb_shots = _shots_in(Path(meqdb_dir), MEQDB_FILE_RE)
    return sorted(defuse_shots & meqdb_shots)


def _is_placeholder(dataset: h5py.Dataset) -> bool:
    """Whether a DEFUSE dataset is a stand-in for a missing signal rather than data.

    MATLAB v7.3 stores an empty array as its shape, flagged by a MATLAB_empty attribute.
    DEFUSE also marks a missing fit with a uint64 [0 0], while real data is always single or double.

    Args:
        dataset: The DEFUSE dataset.

    Returns:
        True for a placeholder.
    """
    is_empty = bool(dataset.attrs.get("MATLAB_empty", 0))
    is_floating = dataset.attrs.get("MATLAB_class") in (b"single", b"double")
    return is_empty or not is_floating


def _unique_finite_times(time: np.ndarray) -> np.ndarray:
    """Indices that sort the finite times and drop repeats, since placing onto the timebase needs increasing times.

    Args:
        time: (n,) sample times [s].

    Returns:
        Indices of the kept samples, in time order.
    """
    idx_finite = np.flatnonzero(np.isfinite(time))
    _, idx_unique = np.unique(time[idx_finite], return_index=True)
    return idx_finite[idx_unique]


def _read_signal(group: h5py.Group, name: str) -> DefuseSignal | None:
    """One 0D signal of a DEFUSE export.

    Args:
        group: The signal's group under SIG.
        name: Its DEFUSE name, for the log and SUMMED_ROW_SIGNALS.

    Returns:
        The signal, None when absent, empty, or laid out in a way it should not be.
    """
    if "signal" not in group or "time" not in group:
        return None
    signal = group["signal"]
    if (
        not isinstance(signal, h5py.Dataset)
        or _is_placeholder(signal)
        or _is_placeholder(group["time"])
    ):
        return None
    time = np.asarray(group["time"][()], dtype=np.float64).ravel()
    values_rows = np.atleast_2d(np.asarray(signal[()], dtype=np.float64))
    if values_rows.shape[0] == time.size and values_rows.shape[1] != time.size:
        values_rows = values_rows.T
    if values_rows.shape[1] != time.size:
        logger.warning(
            f"DEFUSE {name} has shape {values_rows.shape} against {time.size} times, treating it as absent"
        )
        return None
    if values_rows.shape[0] > 1 and name not in SUMMED_ROW_SIGNALS:
        logger.warning(
            f"DEFUSE {name} has {values_rows.shape[0]} rows, treating it as absent"
        )
        return None
    # An actuator row that is NaN contributes nothing, a time with every row NaN stays NaN
    mask_any_finite = np.isfinite(values_rows).any(axis=0)
    values_summed = np.nansum(values_rows, axis=0)
    values = np.where(mask_any_finite, values_summed, np.nan)
    idx_keep = _unique_finite_times(time)
    return DefuseSignal(time=time[idx_keep], values=values[idx_keep])


def read_defuse_signals(
    path: Path, signal_names: tuple[str, ...]
) -> dict[str, DefuseSignal]:
    """The requested 0D signals of one DEFUSE export, each left out when absent or empty.

    Units are DEFUSE's own: SI, except NBI, NBI2 and ECRH in MW.

    Args:
        path: The DEFUSE export.
        signal_names: DEFUSE names of the signals to read.

    Returns:
        The signals read, by DEFUSE name.
    """
    signals = {}
    with h5py.File(path, "r") as defuse_file:
        signal_root = defuse_file["SIG"]
        for name in signal_names:
            if name in signal_root:
                signal = _read_signal(signal_root[name], name)
                if signal is not None:
                    signals[name] = signal
    return signals


def _read_raw_thomson_group(
    group: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One DEFUSE raw Thomson group: times, channel R and Z, readings and errors.

    Args:
        group: The Te_rho or Ne_rho group under SIG.

    Returns:
        (time, r_channel, z_channel, readings, errors) with the readings time-first,
        None when the group is absent, a placeholder, or laid out inconsistently.
    """
    raw = group.get("signal/raw")
    if not isinstance(raw, h5py.Group):
        return None
    members = ("t", "z", "error_bar", "los/rchord", "los/zchord")
    if not all(member in raw for member in members):
        return None
    if any(_is_placeholder(raw[member]) for member in members):
        return None
    time = np.asarray(raw["t"][()], dtype=np.float64).ravel()
    r_channel = np.asarray(raw["los/rchord"][()], dtype=np.float64).ravel()
    z_channel = np.asarray(raw["los/zchord"][()], dtype=np.float64).ravel()
    readings = np.asarray(raw["z"][()], dtype=np.float64)
    errors = np.asarray(raw["error_bar"][()], dtype=np.float64)
    shape_time_first = (time.size, r_channel.size)
    if readings.shape != shape_time_first and readings.T.shape == shape_time_first:
        readings = readings.T
        errors = errors.T
    if readings.shape != shape_time_first or errors.shape != shape_time_first:
        return None
    if z_channel.size != r_channel.size:
        return None
    return time, r_channel, z_channel, readings, errors


def read_defuse_thomson(path: Path) -> DefuseThomson | None:
    """The raw Thomson channels of one DEFUSE export, both variables on one laser timebase.

    The DEFUSE fits are not read, the channels are fit here.
    Nothing is screened yet, see DefuseThomson for DEFUSE's exclusion flags.

    Args:
        path: The DEFUSE export.

    Returns:
        The channels, or None when either variable has no raw group,
        or the two disagree on their times or channels.
    """
    groups = {}
    with h5py.File(path, "r") as defuse_file:
        signal_root = defuse_file["SIG"]
        for var, group_name in DEFUSE_THOMSON_GROUPS.items():
            if group_name not in signal_root:
                return None
            raw_group = _read_raw_thomson_group(signal_root[group_name])
            if raw_group is None:
                return None
            groups[var] = raw_group
    te_time, te_r, te_z, te, te_error = groups["te"]
    ne_time, ne_r, ne_z, ne, ne_error = groups["ne"]
    same_layout = (
        te_time.shape == ne_time.shape
        and np.allclose(te_time, ne_time)
        and np.allclose(te_r, ne_r)
        and np.allclose(te_z, ne_z)
    )
    if not same_layout:
        logger.warning(f"{path.name}: Te and ne raw Thomson disagree on their layout")
        return None
    idx_keep = _unique_finite_times(te_time)
    return DefuseThomson(
        time=te_time[idx_keep],
        r_channel=te_r,
        z_channel=te_z,
        te=te[idx_keep],
        te_error=te_error[idx_keep],
        ne=ne[idx_keep],
        ne_error=ne_error[idx_keep],
    )


def read_liuqe(path: Path) -> dict[str, np.ndarray]:
    """The LIUQE reconstructions of one MEQ database, only what the GEQDSK block needs.

    Loads only the liuqe_data struct, about 250 MB in memory while it is read.
    Every per-time array is returned with time on its last axis, whatever MATLAB squeezed.

    Args:
        path: The MEQ database.

    Returns:
        The LY fields t, Fx (n_z, n_r, n_t), FA, FB, rA, zA, Ip, rBt, lB (n_t,),
        TQ, PQ, TTpQ, PpQ, iqQ (n_q, n_t), the last flux surface's contour rB_lcfs, zB_lcfs (n_theta, n_t),
        and the machine description pQ (n_q,), rx (n_r,), zx (n_z,), rl, zl (n_lim,) and r0.
    """
    mat = scipy.io.loadmat(path, variable_names=["liuqe_data"], simplify_cells=True)
    liuqe = mat["liuqe_data"]
    reconstructions = liuqe["LY"]
    machine = liuqe["L"]
    grid = machine["G"]
    time = np.atleast_1d(np.asarray(reconstructions["t"], dtype=np.float64))
    n_t = time.size
    rho_pol_surfaces = np.asarray(machine["pQ"], dtype=np.float64).ravel()
    r_grid = np.asarray(grid["rx"], dtype=np.float64).ravel()
    z_grid = np.asarray(grid["zx"], dtype=np.float64).ravel()
    fields = {
        "t": time,
        "pQ": rho_pol_surfaces,
        "rx": r_grid,
        "zx": z_grid,
        "rl": np.asarray(grid["rl"], dtype=np.float64).ravel(),
        "zl": np.asarray(grid["zl"], dtype=np.float64).ravel(),
        "r0": float(machine["P"]["r0"]),
    }
    flux_shape = (z_grid.size, r_grid.size, n_t)
    fields["Fx"] = np.asarray(reconstructions["Fx"], dtype=np.float64).reshape(
        flux_shape
    )
    for name in ("FA", "FB", "rA", "zA", "Ip", "rBt", "lB"):
        fields[name] = np.asarray(reconstructions[name], dtype=np.float64).reshape(n_t)
    surfaces_shape = (rho_pol_surfaces.size, n_t)
    for name in ("TQ", "PQ", "TTpQ", "PpQ", "iqQ"):
        fields[name] = np.asarray(reconstructions[name], dtype=np.float64).reshape(
            surfaces_shape
        )
    # Contours of the flux surfaces outside the axis, (n_theta, n_surfaces, n_t), the last one is the LCFS
    contour_r = np.asarray(reconstructions["rq"], dtype=np.float64)
    contour_z = np.asarray(reconstructions["zq"], dtype=np.float64)
    n_theta = contour_r.shape[0]
    contour_shape = (n_theta, contour_r.size // (n_theta * n_t), n_t)
    fields["rB_lcfs"] = contour_r.reshape(contour_shape)[:, -1, :]
    fields["zB_lcfs"] = contour_z.reshape(contour_shape)[:, -1, :]
    return fields


def _profiles_on_psi_n_grid(
    values_surfaces: np.ndarray, psi_n_surfaces: np.ndarray, psi_n_grid: np.ndarray
) -> np.ndarray:
    """Resample per-time flux-surface profiles onto a psi_N grid, linearly.

    Args:
        values_surfaces: (n_q, n_t) values on the LIUQE surfaces.
        psi_n_surfaces: (n_q,) increasing psi_N of the surfaces.
        psi_n_grid: (n_psi,) psi_N grid inside the surfaces' span.

    Returns:
        (n_t, n_psi) values on the grid.
    """
    n_t = values_surfaces.shape[1]
    values_grid = np.empty((n_t, psi_n_grid.size))
    for i_time in range(n_t):
        values_grid[i_time] = np.interp(
            psi_n_grid, psi_n_surfaces, values_surfaces[:, i_time]
        )
    return values_grid


def _liuqe_time_first(values: np.ndarray, mask_time_missing: np.ndarray) -> np.ndarray:
    """A LIUQE array with its time axis moved first, NaN at the missing times.

    Args:
        values: Array with time on its last axis.
        mask_time_missing: (n_t,) True where the time has no usable reconstruction.

    Returns:
        A float copy with time on axis 0.
    """
    values_time_first = np.moveaxis(values, -1, 0).astype(np.float64)
    values_time_first[mask_time_missing] = np.nan
    return values_time_first


def liuqe_geqdsk_dataset(liuqe: dict[str, np.ndarray], shot: int) -> xr.Dataset:
    """Build the GEQDSK block of every LIUQE reconstruction of a shot (make_geqdsk_dataset).

    LIUQE gives its flux per 2 pi radians (COCOS 17), the block takes it per radian,
    so psi is scaled by LIUQE_FLUX_PER_RADIAN and the d/dpsi profiles FF' and p' by its inverse.
    Its flux-function profiles live on 41 surfaces evenly spaced in rho_pol = sqrt(psi_N),
    and are resampled linearly onto the GEQDSK psi_N grid of one point per R grid column, the format's convention.
    q is resampled as LIUQE's 1/q, which is 0 where q diverges at the LCFS of a diverted plasma,
    so q is infinite there, which phi_n_map handles.
    A reconstruction LIUQE found no boundary for (lB 0) is NaN, so usable_reconstructions drops it.
    The flux map is float32, it dominates the block's size.

    Args:
        liuqe: The shot's reconstructions, from read_liuqe.
        shot: Shot number.

    Returns:
        The block on dim "idx" with "time" and "shot" coords, its COCOS as the "cocos" attribute.
    """
    times = liuqe["t"]
    r_grid = liuqe["rx"]
    z_grid = liuqe["zx"]
    mask_no_boundary = liuqe["lB"] == 0
    flux_scale = LIUQE_FLUX_PER_RADIAN

    psi_n_surfaces = liuqe["pQ"] ** 2
    psi_n_grid = geqdsk_psi_n_grid(r_grid.size)
    profiles = {}
    for name, source in (
        ("fpol", "TQ"),
        ("pres", "PQ"),
        ("ffprime", "TTpQ"),
        ("pprime", "PpQ"),
        ("inverse_q", "iqQ"),
    ):
        values_grid = _profiles_on_psi_n_grid(liuqe[source], psi_n_surfaces, psi_n_grid)
        values_grid[mask_no_boundary] = np.nan
        profiles[name] = values_grid
    profiles["ffprime"] = profiles["ffprime"] / flux_scale
    profiles["pprime"] = profiles["pprime"] / flux_scale
    inverse_q = profiles.pop("inverse_q")
    with np.errstate(divide="ignore"):
        qpsi = 1.0 / inverse_q

    # Fx is (z, r, t), the block takes (t, r, z)
    flux_r_z = np.transpose(liuqe["Fx"], (1, 0, 2))
    flux_time_first = _liuqe_time_first(flux_r_z, mask_no_boundary)
    psirz = flux_time_first * flux_scale
    psirz_single = psirz.astype(np.float32)
    flux_axis = _liuqe_time_first(liuqe["FA"], mask_no_boundary)
    flux_boundary = _liuqe_time_first(liuqe["FB"], mask_no_boundary)
    simagx = flux_axis * flux_scale
    sibdry = flux_boundary * flux_scale
    current = _liuqe_time_first(liuqe["Ip"], mask_no_boundary)
    rcentr = liuqe["r0"]
    vacuum_field_times_r0 = _liuqe_time_first(liuqe["rBt"], mask_no_boundary)
    bcentr = vacuum_field_times_r0 / rcentr
    cocos = cocos_from_signs(current, bcentr, simagx, sibdry, qpsi)
    return make_geqdsk_dataset(
        shot_id=shot,
        times=times,
        r_grid=r_grid,
        z_grid=z_grid,
        rmagx=_liuqe_time_first(liuqe["rA"], mask_no_boundary),
        zmagx=_liuqe_time_first(liuqe["zA"], mask_no_boundary),
        simagx=simagx,
        sibdry=sibdry,
        bcentr=bcentr,
        current=current,
        psirz=psirz_single,
        rbdry=_liuqe_time_first(liuqe["rB_lcfs"], mask_no_boundary),
        zbdry=_liuqe_time_first(liuqe["zB_lcfs"], mask_no_boundary),
        cocos_input=cocos,
        rcentr=rcentr,
        rlim=liuqe["rl"],
        zlim=liuqe["zl"],
        qpsi=qpsi,
        **profiles,
    )
