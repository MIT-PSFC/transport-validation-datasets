import re
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr
from eqdsk.cocos import identify_cocos
from loguru import logger
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicHermiteSpline, RegularGridInterpolator
from scipy.special import xlogy

IMAS_DOCS_URL = "https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids"

# Vacuum permeability [H/m]
MU0 = 4e-7 * np.pi

# Attributes of the GEQDSK block make_geqdsk_dataset builds, freeqdsk names.
# Units are those of COCOS 1 to 8, the range cocos_from_signs returns, where the poloidal flux is per radian.
# The shot's COCOS number rides on the dataset's "cocos" attribute,
# and in the store as the per-shot cocos variable.
# "ref" is the IMAS data dictionary path, as for every other signal.
GEQDSK_SIGNAL_ATTRS = {
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
    "simagx": {
        "description": "Poloidal flux at the magnetic axis",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/psi_axis",
    },
    "sibdry": {
        "description": "Poloidal flux at the plasma boundary (LCFS)",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/psi_boundary",
    },
    "bcentr": {
        "description": "Vacuum toroidal field at rcentr, from the equilibrium reconstruction",
        "units": "T",
        "ref": "/equilibrium/vacuum_toroidal_field/b0",
    },
    "current": {
        "description": "Plasma current from the equilibrium reconstruction",
        "units": "A",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/ip",
    },
    "rcentr": {
        "description": "Major radius bcentr is given at",
        "units": "m",
        "ref": "/equilibrium/vacuum_toroidal_field/r0",
    },
    "rleft": {
        "description": "Major radius of the inner edge of the psirz grid, r_grid[0]",
        "units": "m",
    },
    "rdim": {
        "description": "Radial extent of the psirz grid, r_grid[-1] - r_grid[0]",
        "units": "m",
    },
    "zmid": {
        "description": "Height of the center of the psirz grid",
        "units": "m",
    },
    "zdim": {
        "description": "Vertical extent of the psirz grid, z_grid[-1] - z_grid[0]",
        "units": "m",
    },
    "fpol": {
        "description": "Poloidal current function F = R B_phi on the psi_idx grid",
        "units": "T m",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/f",
    },
    "pres": {
        "description": "Plasma pressure on the psi_idx grid",
        "units": "Pa",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/pressure",
    },
    "ffprime": {
        "description": "F dF/dpsi on the psi_idx grid",
        "units": "T^2 m^2 rad/Wb",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/f_df_dpsi",
    },
    "pprime": {
        "description": "dp/dpsi on the psi_idx grid",
        "units": "Pa rad/Wb",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/dpressure_dpsi",
    },
    "qpsi": {
        "description": "Safety factor on the psi_idx grid",
        "units": "dimensionless",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/q",
    },
    "psirz": {
        "description": "Poloidal flux on the (r_grid, z_grid) grid",
        "units": "Wb/rad",
        "ref": "/equilibrium/time_slice(itime)/profiles_2d(i1)/psi",
    },
    "rbdry": {
        "description": "Major radius of the plasma boundary contour points, NaN padded",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/outline/r",
    },
    "zbdry": {
        "description": "Height of the plasma boundary contour points, NaN padded",
        "units": "m",
        "ref": "/equilibrium/time_slice(itime)/boundary/outline/z",
    },
    "rlim": {
        "description": "Major radius of the limiter contour points",
        "units": "m",
        "ref": "/wall/description_2d(i1)/limiter/unit(i2)/outline/r",
    },
    "zlim": {
        "description": "Height of the limiter contour points",
        "units": "m",
        "ref": "/wall/description_2d(i1)/limiter/unit(i2)/outline/z",
    },
    # Coordinates
    "r_grid": {"description": "Major radius of the psirz grid points", "units": "m"},
    "z_grid": {"description": "Height of the psirz grid points", "units": "m"},
    "psi_idx": {
        "description": "Index on the uniform normalized poloidal flux grid of the "
        "1D profiles, 0 at the magnetic axis to 1 at the boundary"
    },
    "boundary_idx": {"description": "Index along the plasma boundary contour"},
    "limiter_idx": {"description": "Index along the limiter contour"},
}


def imas_url(ref: str) -> str:
    """Documentation URL of an IMAS data dictionary path.

    The form disruption-py records next to its paths: the IDS page, and an
    anchor of the path with the array indices ((itime), (i1)) dropped.

    Args:
        ref: Data dictionary path, e.g. /equilibrium/time_slice(itime)/profiles_1d/q.

    Returns:
        The URL.
    """
    parts = [re.sub(r"\(.*?\)", "", part) for part in ref.strip("/").split("/")]
    return f"{IMAS_DOCS_URL}/{parts[0]}.html#{'-'.join(parts)}"


def standardize_signal_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Bring every variable's attributes onto one convention, for the stores.

    The GEQDSK signals and coordinates get GEQDSK_SIGNAL_ATTRS where the
    device set nothing (files from before make_geqdsk_dataset set them),
    a data dictionary path under "imas" (disruption-py's key) moves to "ref"
    (this package's), and every ref without a url gets one (imas_url).
    A device's own description, units, or url always win.

    Args:
        ds: Dataset whose variables and coordinates are updated in place.

    Returns:
        The same dataset.
    """
    for name in list(ds.variables):
        attrs = dict(ds[name].attrs)
        for key, value in GEQDSK_SIGNAL_ATTRS.get(name, {}).items():
            attrs.setdefault(key, value)
        if "imas" in attrs:
            path = attrs.pop("imas")
            attrs.setdefault("ref", path)
        if "ref" in attrs and "url" not in attrs:
            attrs["url"] = imas_url(attrs["ref"])
        ds[name].attrs = attrs
    return ds


# Width of the centered boxcar power_ohm and power_radiated are smoothed with, applied twice [s], see smoothed_power.
# Unsmoothed, the inductive swings of P_oh and the bolometer noise of P_rad are larger than the signals at 1 kHz.
# DIII-D's sources come smoothed this way or more (prad_tot, EFIT poh), so the other devices match them.
POWER_SMOOTHING_WINDOW = 50e-3

# Step of the uniform timebase every device is placed on [s], see make_uniform_1kHz_timebase.
UNIFORM_TIMEBASE_DT = 1e-3

# A grid time carries a sample of its own when it sits this close to one [s].
# Only absorbs float round-off, everything shares the staged 1 kHz timebase.
SAMPLE_TIME_TOL = 1e-6

# How long an equilibrium or a 0D signal slower than the grid
# is held forward onto the 1 kHz timebase, in periods of its own sampling.
# Above 1 to tolerate jitter in the sampling, low enough that nothing is carried across a real gap:
# the end of the shot, or a diagnostic dropping out.
MAX_HOLD_PERIODS = 1.5

# Longest a fitted profile slice is held forward onto the 1 kHz timebase on every device, whatever its cadence [s].
# Long enough to bridge dropped Thomson slices and burst-mode gaps, fresh_profile marks the slices themselves.
PROFILE_MAX_HOLD = 100e-3

# Shortest an equilibrium reconstruction is held, whatever its clock [s].
# Bridges the dropouts of single reconstructions (a 1 kHz EFIT failing a few slices,
# one missing 5 ms MAST reconstruction) that would otherwise cut every equilibrium signal.
# fresh_equilibrium still marks only the grid times a usable reconstruction lands on.
EQUILIBRIUM_HOLD_FLOOR = 10e-3

# How far a TS slice may sit from the reconstruction it maps through,
# in periods of the reconstruction's own sampling.
# Above 1 to tolerate clock jitter, low enough that nothing is borrowed across a real gap.
EQ_MATCH_MAX_PERIODS = 1.5

# How Phi_N continues past the LCFS, see phi_n_map.
SOL_EXTENSIONS = ("secant", "tangent")

# The secant SOL extension takes its slope over psi_N from here to the LCFS.
SECANT_PSI_N = 0.95

# Outermost finite-q surfaces the logarithmic q tail of a diverted plasma is fit to, see phi_n_map.
Q_TAIL_FIT_SURFACES = 4

# Points of the dense psi_N table PhiNMap.psi_n inverts Phi_N on
NUM_INVERSE_POINTS = 4097

# A reconstruction whose boundary and axis psi sit closer than this has no usable flux map.
MIN_PSI_RANGE = 1e-10


def make_uniform_1kHz_timebase(max_time: float) -> np.ndarray:
    """Create a uniform timebase at 1 kHz up to the specified maximum time.

    This is the timebase used for all datasets.
    Built from an integer millisecond count to avoid problems with float accumulation.

    Args:
        max_time: The maximum time for the timebase [s].

    Returns:
        Times from 0 to max_time in 1 ms steps [s].
    """
    steps_to_max = np.round(max_time / UNIFORM_TIMEBASE_DT, 6)
    last_step = int(np.ceil(steps_to_max))
    step_counts = np.arange(last_step + 1, dtype=np.float64)
    times = np.round(step_counts * UNIFORM_TIMEBASE_DT, 3)
    return times.astype("float32")


def make_geqdsk_dataset(
    shot_id,
    times,
    r_grid,
    z_grid,
    rmagx,
    zmagx,
    simagx,
    sibdry,
    bcentr,
    current,
    fpol,
    pres,
    ffprime,
    pprime,
    qpsi,
    psirz,
    rbdry,
    zbdry,
    cocos_input,
    rcentr,
    rlim=None,
    zlim=None,
):
    """Build an Xarray dataset holding every signal needed to recreate a GEQDSK file.

    FreeQDSK canonical names, COCOS 1.

    Args:
        shot_id: Shot number, repeated along 'idx' as the 'shot' coordinate.
        times: (n_t,) times of the reconstruction slices [s].
        r_grid: (n_r,) major radii of the psi grid columns [m], which also set rleft and rdim.
        z_grid: (n_z,) heights of the psi grid rows [m], which also set zmid and zdim.
        rmagx: (n_t,) major radius of the magnetic axis [m].
        zmagx: (n_t,) height of the magnetic axis [m].
        simagx: (n_t,) poloidal flux at the magnetic axis [Wb/rad].
        sibdry: (n_t,) poloidal flux at the plasma boundary [Wb/rad].
        bcentr: (n_t,) vacuum toroidal field at rcentr [T].
        current: (n_t,) plasma current [A].
        fpol: (n_t, n_psi) poloidal current function R*B_t [m*T].
        pres: (n_t, n_psi) plasma pressure [Pa].
        ffprime: (n_t, n_psi) F dF/dpsi [m^2*T^2/(Wb/rad)].
        pprime: (n_t, n_psi) dp/dpsi [Pa/(Wb/rad)].
        qpsi: (n_t, n_psi) safety factor.
        psirz: (n_t, n_r, n_z) poloidal flux on the (r_grid, z_grid)
            grid [Wb/rad].
        rbdry: (n_t, n_bdry) major radii of the boundary contour [m].
        zbdry: (n_t, n_bdry) heights of the boundary contour [m].
        cocos_input: COCOS convention the inputs follow, stored as the
            dataset's 'cocos' attribute.
        rcentr: Reference radius bcentr is quoted at [m].
            A machine or reconstruction constant that the grid does not determine.
        rlim: (n_lim,) major radii of the limiter contour [m].
            Static, and stored only when zlim is given too.
        zlim: (n_lim,) heights of the limiter contour [m].
            Static, and stored only when rlim is given too.

    Returns:
        Dataset with all GEQDSK signals on dim 'idx', with 'time'/'shot' coords.
    """
    rleft = r_grid[0]
    rdim = r_grid[-1] - r_grid[0]
    zmid = z_grid[len(z_grid) // 2]
    zdim = z_grid[-1] - z_grid[0]

    ds_geqdsk = xr.Dataset(
        data_vars={
            # 0D time-invariant signals
            "rcentr": ([], rcentr),
            "rleft": ([], rleft),
            "zmid": ([], zmid),
            "rdim": ([], rdim),
            "zdim": ([], zdim),
            # 0D signals
            "rmagx": ("idx", rmagx),
            "zmagx": ("idx", zmagx),
            "simagx": ("idx", simagx),
            "sibdry": ("idx", sibdry),
            "bcentr": ("idx", bcentr),
            "current": ("idx", current),
            # 1D signals
            "qpsi": (("idx", "psi_idx"), qpsi),
            "fpol": (("idx", "psi_idx"), fpol),
            "pres": (("idx", "psi_idx"), pres),
            "ffprime": (("idx", "psi_idx"), ffprime),
            "pprime": (("idx", "psi_idx"), pprime),
            # 2D signals
            "psirz": (("idx", "r_grid", "z_grid"), psirz),
            # Boundary
            "rbdry": (("idx", "boundary_idx"), rbdry),
            "zbdry": (("idx", "boundary_idx"), zbdry),
            # Limiter (static - not time-dependent)
            **(
                {
                    "rlim": ("limiter_idx", rlim),
                    "zlim": ("limiter_idx", zlim),
                }
                if rlim is not None and zlim is not None
                else {}
            ),
        },
        coords={
            "time": ("idx", times),
            "shot": ("idx", np.repeat(shot_id, len(times), axis=0)),
            "r_grid": r_grid,
            "z_grid": z_grid,
            "psi_idx": np.arange(qpsi.shape[1]),
            "boundary_idx": np.arange(rbdry.shape[1]),
            **(
                {
                    "limiter_idx": np.arange(len(rlim)),
                }
                if rlim is not None
                else {}
            ),
        },
        attrs={
            "cocos": cocos_input,
        },
    )
    for name in ds_geqdsk.variables:
        if name in GEQDSK_SIGNAL_ATTRS:
            ds_geqdsk[name].attrs.update(GEQDSK_SIGNAL_ATTRS[name])

    return ds_geqdsk


def snap_to_grid(ds: xr.Dataset, grid_times: np.ndarray) -> xr.Dataset:
    """Snap an EFIT reconstruction (dim 'idx', 'time'/'shot' coords) onto grid_times.

    Each EFIT slice goes to its nearest grid time, no interpolation, no fill.
    The snap only absorbs sub-step jitter in the EFIT clock, so ties break toward
    the later grid point: an EFIT at 9.5 ms on a 1 ms grid lands at 10 ms, never
    feeding the 9 ms slice a future reconstruction. Slices further than half a
    grid step outside the grid are dropped rather than piled onto the first or
    last grid time, since a reconstruction from before or after the shot window
    is not a measurement of either end of it. Grid times left with no slice come
    back as NaN.

    Args:
        ds: EFIT reconstruction with dim 'idx' and 'time'/'shot' coords.
        grid_times: Uniform timebase to snap onto [s].

    Returns:
        The reconstruction on grid_times, NaN at grid times with no EFIT slice.
    """
    grid_times = np.asarray(grid_times)
    step = float(np.median(np.diff(grid_times)))
    # tol sits well below any real EFIT timing gap but far above float32
    # round-off at the grid step
    tol = 1e-4 * step

    shot_id = ds["shot"].values[0]
    in_range = (ds["time"].values >= grid_times[0] - 0.5 * step - tol) & (
        ds["time"].values <= grid_times[-1] + 0.5 * step + tol
    )
    ds = ds.isel(idx=in_range)
    efit_times = ds["time"].values

    # Nearest grid point per EFIT slice
    pos = np.searchsorted(grid_times, efit_times, side="left")
    left = np.clip(pos - 1, 0, len(grid_times) - 1)
    right = np.clip(pos, 0, len(grid_times) - 1)
    take_left = (grid_times[right] - efit_times) - (efit_times - grid_times[left]) > tol
    snapped = grid_times[np.where(take_left, left, right)]

    # snapped is non-decreasing (both arrays sorted), so duplicate slots are adjacent
    # keep the last EFIT slice that lands in each, then NaN-fill the empty grid times
    keep = np.append(np.diff(snapped) != 0, True)[: snapped.size]
    ds = ds.drop_vars(["time", "shot"]).assign_coords(idx=snapped).isel(idx=keep)
    ds = ds.reindex(idx=grid_times).reset_index("idx", drop=True)
    return ds.assign_coords(
        time=("idx", grid_times),
        shot=("idx", np.repeat(shot_id, len(grid_times))),
    )


def hold_onto_grid(
    grid: np.ndarray,
    sample_times: np.ndarray,
    forward_fill: bool,
    period: float | None = None,
    max_hold_periods: float = MAX_HOLD_PERIODS,
    hold_floor: float = 0.0,
    max_hold_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every grid time onto the sample it takes its values from.

    Each grid time takes the most recent sample at or before it,
    held for at most max_hold_periods sampling periods, or hold_floor when that is longer,
    so that nothing is carried across a long gap:
    the end of the shot, a diagnostic dropping out, or a stretch the filtering cut away.
    max_hold_time sets the longest hold directly instead, whatever the sampling period.
    A sample within SAMPLE_TIME_TOL after a grid time counts as at it.
    A grid time is fresh when its sample falls in its own grid step (t - grid step, t],
    so each sample is fresh at the first grid time that holds it, on the grid or between grid times.
    With forward_fill False nothing is held and only the fresh grid times come out.

    Args:
        grid: The shot's 1 kHz timebase [s].
        sample_times: Times of the samples to place on it [s], ascending.
        forward_fill: Hold each sample forward until the next one.
        period: The sampling period to hold for [s]. None takes the median
            spacing of sample_times, which is right when they are every
            sample there is, and wrong when they are a windowed subset.
        max_hold_periods: Longest hold, in sampling periods.
        hold_floor: The hold reaches at least this far [s].
        max_hold_time: Longest hold [s], replaces max_hold_periods and hold_floor when given.

    Returns:
        (sample_index, fresh): sample_index[i] is the index of the sample
        that grid time i draws from, -1 where it draws from none, and
        fresh[i] marks the grid times whose sample falls in their own grid step.
    """
    sample_index = np.full(grid.size, -1, dtype=int)
    fresh = np.zeros(grid.size, dtype=bool)
    if sample_times.size == 0:
        return sample_index, fresh

    # float64, since a float32 grid time can sit just below the sample at the same millisecond
    grid_float64 = grid.astype(np.float64)
    sample_times_float64 = sample_times.astype(np.float64)
    grid_steps = np.diff(grid_float64)
    # A one-point grid has no step, so its sample is fresh wherever it lies before it
    grid_step = float(np.median(grid_steps)) if grid_steps.size else np.inf
    # Index of the last sample at or before each grid time
    previous_sample = (
        np.searchsorted(
            sample_times_float64, grid_float64 + SAMPLE_TIME_TOL, side="right"
        )
        - 1
    )
    has_previous = previous_sample >= 0
    previous_sample_clipped = np.clip(previous_sample, 0, None)
    age = grid_float64 - sample_times_float64[previous_sample_clipped]
    fresh = has_previous & (age < grid_step - SAMPLE_TIME_TOL)
    if not forward_fill:
        sample_index[fresh] = previous_sample[fresh]
        return sample_index, fresh
    if max_hold_time is not None:
        max_hold = max_hold_time
    else:
        # One sample on its own has no period to hold for,
        # so it only fills the grid step it sits on
        if period is None:
            sample_spacing = (
                np.diff(sample_times_float64) if sample_times.size > 1 else grid_steps
            )
            period = float(np.median(sample_spacing))
        max_hold = max(max_hold_periods * period, hold_floor)
    still_held = has_previous & (age <= max_hold)
    sample_index[still_held] = previous_sample[still_held]
    return sample_index, fresh


def values_on_grid(sample_values: np.ndarray, sample_index: np.ndarray) -> np.ndarray:
    """Per-sample values picked onto the grid by hold_onto_grid's sample_index.

    Args:
        sample_values: (n_samples, ...) one row per sample.
        sample_index: (n_grid,) the sample each grid time draws from, -1 where none.

    Returns:
        (n_grid, ...) the rows, float64, NaN where no sample holds.
    """
    mask_held = sample_index >= 0
    values_shape = (sample_index.size, *sample_values.shape[1:])
    values = np.full(values_shape, np.nan)
    values[mask_held] = sample_values[sample_index[mask_held]]
    return values


def signal_on_grid(
    source_times: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    hold_floor: float = 0.0,
) -> np.ndarray:
    """Place a 0D signal on the grid causally, so no grid value draws on a later sample.

    A source sampled faster than the grid is averaged over each grid step (_window_mean_on_grid).
    A slower one is held forward from its last sample (_held_on_grid), at least hold_floor.
    Only finite samples count.
    The source's period is the median spacing of its finite samples,
    so a fast clock populated only at a slower cadence (MAST esm) is held on that cadence.
    Fewer than two finite samples have no period, and give all NaN.

    Args:
        source_times: (n_source,) ascending sample times of the source [s].
        values: (n_source,) the signal at those times, NaN where missing.
        grid: The shot's 1 kHz timebase [s].
        hold_floor: The hold of a slower source reaches at least this far [s].

    Returns:
        (n_grid,) the signal on the grid, NaN where it has no value.
    """
    mask_finite = np.isfinite(values)
    if mask_finite.sum() < 2:
        return np.full(grid.size, np.nan)
    times_finite = source_times[mask_finite]
    values_finite = values[mask_finite]
    source_steps = np.diff(times_finite)
    source_period = float(np.median(source_steps))
    grid_float64 = grid.astype(np.float64)
    grid_steps = np.diff(grid_float64)
    grid_step = float(np.median(grid_steps))
    if source_period < grid_step - SAMPLE_TIME_TOL:
        return _window_mean_on_grid(
            times_finite, values_finite, grid_float64, grid_step
        )
    return _held_on_grid(times_finite, values_finite, grid_float64, hold_floor)


def _held_on_grid(
    source_times: np.ndarray, values: np.ndarray, grid: np.ndarray, hold_floor: float
) -> np.ndarray:
    """Each grid time takes the last sample at or before it (hold_onto_grid).

    A sample is held for at most MAX_HOLD_PERIODS of the median sample spacing, or hold_floor when longer,
    so a longer gap in the source stays NaN.

    Args:
        source_times: (n_source,) ascending finite sample times [s], at least two.
        values: (n_source,) the samples.
        grid: (n_grid,) the timebase [s].
        hold_floor: The hold reaches at least this far [s].

    Returns:
        (n_grid,) the held signal, NaN where nothing is held.
    """
    sample_index, _ = hold_onto_grid(grid, source_times, True, hold_floor=hold_floor)
    return values_on_grid(values, sample_index)


def _window_mean_on_grid(
    source_times: np.ndarray, values: np.ndarray, grid: np.ndarray, grid_step: float
) -> np.ndarray:
    """Each grid time t takes the mean of the samples in (t - grid_step, t].

    A sample within SAMPLE_TIME_TOL after a grid time counts as at it, as in hold_onto_grid.

    Args:
        source_times: (n_source,) ascending finite sample times [s].
        values: (n_source,) the samples.
        grid: (n_grid,) the uniform timebase [s], float64.
        grid_step: Its step [s].

    Returns:
        (n_grid,) the window means, NaN where a window holds no sample.
    """
    first_window_start = grid[0] - grid_step
    window_edges = np.concatenate([[first_window_start], grid])
    window_index = (
        np.searchsorted(window_edges, source_times - SAMPLE_TIME_TOL, side="left") - 1
    )
    mask_on_grid = (window_index >= 0) & (window_index < grid.size)
    window_index_on_grid = window_index[mask_on_grid]
    values_in_windows = values[mask_on_grid]
    window_sums = np.bincount(
        window_index_on_grid, weights=values_in_windows, minlength=grid.size
    )
    window_counts = np.bincount(window_index_on_grid, minlength=grid.size)
    window_means = np.full(grid.size, np.nan)
    mask_sampled = window_counts > 0
    window_means[mask_sampled] = window_sums[mask_sampled] / window_counts[mask_sampled]
    return window_means


def injected_power_on_grid(
    record_times: np.ndarray, power: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """An injected heating power record placed on the grid (signal_on_grid), 0 outside the record.

    A record of fewer than two samples is taken as a heating system that did not run, 0 throughout.

    Args:
        record_times: (n_record,) ascending sample times of the record [s].
        power: (n_record,) the power, in the record's units.
        grid: (n_grid,) the shot's 1 kHz timebase [s].

    Returns:
        (n_grid,) the power on the grid.
    """
    if record_times.size < 2:
        return np.zeros(grid.size)
    power_on_grid = signal_on_grid(record_times, power, grid)
    mask_outside_record = (grid < record_times[0]) | (grid > record_times[-1])
    power_on_grid[mask_outside_record] = 0.0
    return power_on_grid


def centered_boxcar_mean(values: np.ndarray, window: float, dt: float) -> np.ndarray:
    """Smooth a uniformly sampled signal with a boxcar centered on each sample, along the last axis.

    The boxcar spans an odd number of samples, so it stays centered on the present one.
    NaN samples are skipped, and near the ends and around NaNs it averages over the samples present.

    Args:
        values: (..., n) the signal, uniformly sampled along its last axis.
        window: Width of the boxcar [s].
        dt: The sample spacing [s].

    Returns:
        (..., n) the smoothed signal, NaN only where the whole window is.
    """
    n_samples = max(1, round(window / dt))
    if n_samples % 2 == 0:
        n_samples += 1
    signal = xr.DataArray(values)
    last_dim = signal.dims[-1]
    smoothed = signal.rolling({last_dim: n_samples}, center=True, min_periods=1).mean()
    return smoothed.values


def smoothed_power(values: np.ndarray, dt: float) -> np.ndarray:
    """Smooth a power on the uniform grid with the centered POWER_SMOOTHING_WINDOW boxcar applied twice.

    The kernel is a triangle twice the window wide at its base, DIII-D's prad_tot kernel,
    so every device's power_ohm and power_radiated carry the same smoothing.
    It draws on samples up to one window later, so the result is not causal.
    A NaN sample stays NaN, so a gap in the record stays a gap for the filters.

    Args:
        values: (n,) the power on the uniform grid.
        dt: The grid step [s].

    Returns:
        (n,) the smoothed power.
    """
    smoothed_once = centered_boxcar_mean(values, POWER_SMOOTHING_WINDOW, dt)
    smoothed_twice = centered_boxcar_mean(smoothed_once, POWER_SMOOTHING_WINDOW, dt)
    mask_missing = np.isnan(values)
    smoothed_twice[mask_missing] = np.nan
    return smoothed_twice


def ohmic_power(
    times: np.ndarray,
    ip: np.ndarray,
    v_loop: np.ndarray,
    li: np.ndarray,
    major_radius: np.ndarray,
) -> np.ndarray:
    """Ohmic power P_oh = Ip V_loop - dW_pol/dt, causal.

    The internal poloidal field energy is W_pol = L_i Ip^2 / 2 with L_i = mu0 R_geo li / 2,
    R_geo the major radius of the geometric center of the boundary.
    dW_pol/dt is a backward difference, so no sample draws on a later one,
    and the first sample is NaN.
    The sign of Ip and V_loop cancels as long as they share a convention.

    Args:
        times: (n,) ascending sample times [s].
        ip: (n,) plasma current [A].
        v_loop: (n,) loop voltage [V].
        li: (n,) internal inductance.
        major_radius: (n,) major radius of the geometric center of the boundary [m].

    Returns:
        (n,) ohmic power [W].
    """
    w_pol = MU0 * major_radius * li * ip**2 / 4.0
    dw_pol = np.diff(w_pol, prepend=np.nan)
    dt = np.diff(times, prepend=np.nan)
    dw_pol_dt = dw_pol / dt
    return ip * v_loop - dw_pol_dt


def greenwald_density(ip, minor_radius):
    """The Greenwald density n_GW = Ip / (pi a^2), in 1e20 m^-3, MA and m.

    Args:
        ip: Plasma current [A], either sign.
        minor_radius: Minor radius [m].

    Returns:
        n_GW [m^-3], shaped like the broadcast inputs.
    """
    ip_magnitude = abs(ip)
    ip_magnitude_ma = ip_magnitude / 1e6
    return 1e20 * ip_magnitude_ma / (np.pi * minor_radius**2)


def greenwald_fraction(ip, minor_radius, n_e_line_average):
    """Line-averaged density over the Greenwald density (greenwald_density).

    Args:
        ip: Plasma current [A], either sign.
        minor_radius: Minor radius [m].
        n_e_line_average: Line-averaged electron density [m^-3].

    Returns:
        The Greenwald fraction, shaped like the broadcast inputs.
    """
    n_greenwald = greenwald_density(ip, minor_radius)
    return n_e_line_average / n_greenwald


def end_of_shot_index(
    ip_magnitude: np.ndarray, times: np.ndarray, ip_min: float, end_margin: float
) -> int | None:
    """Index of the first grid time the end-of-shot cut removes.

    The plasma ends at the last grid time with |ip| at or above ip_min,
    since a record can hold ip near 0 long after the plasma is gone.
    Everything after end_margin before that is cut, to leave out the termination, often a disruption.
    The margin is counted in grid steps, so float round-off in the times never moves the cut.

    Args:
        ip_magnitude: (n_t,) |ip| on the uniform grid [A], NaN where unknown.
        times: (n_t,) the grid [s].
        ip_min: The ip min_filter threshold [A].
        end_margin: Margin before the end of the plasma [s].

    Returns:
        The index, or None when |ip| never reaches ip_min.
    """
    with np.errstate(invalid="ignore"):
        mask_plasma = ip_magnitude >= ip_min
    if not mask_plasma.any():
        return None
    last_plasma_idx = int(np.flatnonzero(mask_plasma)[-1])
    grid_steps = np.diff(times)
    margin_steps = round(end_margin / float(np.median(grid_steps)))
    return last_plasma_idx - margin_steps + 1


def kept_segments(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the runs of kept samples.

    Args:
        keep: Mask over the uniform 1 kHz grid.

    Returns:
        (starts, ends): the index of each run's first sample and one past its last.
    """
    # Pad with False on both sides so a run touching either end still has an edge
    keep_padded = np.concatenate(([False], keep, [False]))
    edges = np.diff(keep_padded.astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def keep_longest_segment(
    keep: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """Clear every run of kept samples but the longest.

    A run is measured from its first to its last sample, the same way kept_span measures the one left,
    so a run of n samples on the 1 kHz grid is n - 1 milliseconds long.
    The earliest of equally long runs is kept.

    Args:
        keep: Mask over times, True where the sample survived the filters. Not modified.
        times: The shot's timebase [s].

    Returns:
        The mask with only the longest run left, and the lengths [s] of the runs cleared.
    """
    keep = np.asarray(keep, dtype=bool)
    keep_longest = np.zeros(keep.size, dtype=bool)
    starts, ends = kept_segments(keep)
    if starts.size == 0:
        return keep_longest, []
    run_lengths = times[ends - 1] - times[starts]
    longest = int(np.argmax(run_lengths))
    keep_longest[starts[longest] : ends[longest]] = True
    dropped_lengths = np.delete(run_lengths, longest)
    return keep_longest, dropped_lengths.tolist()


def kept_span(keep: np.ndarray, times: np.ndarray) -> float:
    """Time from the first to the last kept sample, for the min_pulse_length gate.

    Args:
        keep: Mask over times, True where the sample survived the filters.
        times: The shot's timebase [s].

    Returns:
        The span [s], 0 when fewer than two samples are kept.
    """
    keep = np.asarray(keep, dtype=bool)
    kept_times = times[keep]
    if kept_times.size < 2:
        return 0.0
    return float(kept_times[-1] - kept_times[0])


def usable_reconstructions(ds: xr.Dataset) -> np.ndarray:
    """Mark the times that carry a reconstruction the Thomson mapping and the stores can use.

    A reconstruction is usable when its axis and boundary psi are finite and meaningfully different,
    its whole psirz and qpsi are finite,
    and its q profile gives a Phi_N map (phi_n_map), so q keeps one sign.

    Args:
        ds: One shot's dataset with the GEQDSK block on its time axis.

    Returns:
        (n_t,) boolean mask over the time axis, False where nothing was reconstructed.
    """
    if "shot" in ds.dims:
        ds = ds.squeeze("shot", drop=True)
    simagx = ds["simagx"].transpose("time").values
    sibdry = ds["sibdry"].transpose("time").values
    psi_range = sibdry - simagx
    psi_range_usable = np.isfinite(psi_range) & (np.abs(psi_range) > MIN_PSI_RANGE)
    qpsi = ds["qpsi"].transpose("time", "psi_idx").values
    psirz_finite = (
        np.isfinite(ds["psirz"]).all(("r_grid", "z_grid")).transpose("time").values
    )
    psi_n_grid = geqdsk_psi_n_grid(qpsi.shape[1])
    q_mappable = mappable_q_profiles(psi_n_grid, qpsi)
    return psi_range_usable & q_mappable & psirz_finite


def reconstruction_clock_period(ds: xr.Dataset, grid: np.ndarray) -> float:
    """Median spacing of a shot's reconstructions, the unusable ones included.

    The Thomson mapping's reach and the stores' equilibrium hold both run on this clock,
    so dropping an unusable reconstruction (usable_reconstructions) stretches neither.
    A lone reconstruction has no spacing of its own, so the grid step stands in,
    and a one-sample grid has none either, giving 0.

    Args:
        ds: One shot's dataset with the GEQDSK block on its time axis.
        grid: The times of that axis [s].

    Returns:
        The period [s].
    """
    if "shot" in ds.dims:
        ds = ds.squeeze("shot", drop=True)
    simagx = ds["simagx"].transpose("time").values
    clock_times = grid[np.isfinite(simagx)]
    if clock_times.size > 1:
        clock_steps = np.diff(clock_times)
    elif grid.size > 1:
        clock_steps = np.diff(grid)
    else:
        return 0.0
    return float(np.median(clock_steps))


def geqdsk_psi_n_grid(num_psi: int) -> np.ndarray:
    """The psi_N grid of a GEQDSK profile such as qpsi, uniform from 0 at the axis to 1 at the LCFS by the format's definition.

    Args:
        num_psi: Number of points of the profile.

    Returns:
        (num_psi,) psi_N grid.
    """
    return np.linspace(0.0, 1.0, num_psi)


def cumulative_q_integral(psi_n_grid: np.ndarray, qpsi: np.ndarray) -> np.ndarray:
    """Integrate the safety factor over normalized poloidal flux, outward from the axis, by Simpson's rule.

    The toroidal flux is phi = integral q dpsi,
    so this is phi in units of (psi_boundary - psi_axis),
    and dividing it by its last value gives the normalized toroidal flux Phi_N.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis.
        qpsi: (..., n_psi) safety factor on psi_n_grid.

    Returns:
        (..., n_psi) integral of q dpsi_N from 0 to each grid point, starting at 0.
    """
    return cumulative_simpson(qpsi, x=psi_n_grid, initial=0.0)


def _q_tail_integral(
    psi_n_from: np.ndarray | float,
    psi_n_to: np.ndarray | float,
    q_offset: float,
    q_log_slope: float,
) -> np.ndarray | float:
    """Integral of q = q_offset - q_log_slope ln(1 - psi_N) over psi_N, finite up to psi_N = 1.

    The antiderivative of -ln(1 - psi_N) is (1 - psi_N) ln(1 - psi_N) - (1 - psi_N),
    and xlogy keeps it 0 at psi_N = 1.

    Args:
        psi_n_from: Lower limit.
        psi_n_to: Upper limit, any shape broadcasting with psi_n_from.
        q_offset: q_offset of the tail.
        q_log_slope: q_log_slope of the tail.

    Returns:
        The integral, shaped like the broadcast limits.
    """
    one_minus_from = 1.0 - psi_n_from
    one_minus_to = 1.0 - psi_n_to
    antiderivative_from = q_offset * psi_n_from + q_log_slope * (
        xlogy(one_minus_from, one_minus_from) - one_minus_from
    )
    antiderivative_to = q_offset * psi_n_to + q_log_slope * (
        xlogy(one_minus_to, one_minus_to) - one_minus_to
    )
    return antiderivative_to - antiderivative_from


@dataclass(frozen=True)
class PhiNMap:
    """Phi_N(psi_N) of one equilibrium, built by phi_n_map.

    Phi_N is the integral of |q| over psi_N, normalized to 1 at the LCFS.
    Inside the last surface of finite q the integral is Simpson's rule over the surfaces,
    interpolated between them by a cubic Hermite spline whose slope is q itself,
    so dPhi_N/dpsi_N stays continuous and gradients have no kinks at the surfaces.
    From there to the LCFS it is the analytic integral of the logarithmic q tail,
    zero width when q is finite at the LCFS.
    Outside the LCFS Phi_N continues linearly in psi_N with sol_slope.
    """

    q_integral_spline: CubicHermiteSpline
    psi_n_join: float  # last surface of finite q, 1 when q is finite at the LCFS
    q_integral_join: float  # integral of |q| from the axis to psi_n_join
    q_offset: float  # q_offset of the tail, 0 when q is finite at the LCFS
    q_log_slope: float  # q_log_slope of the tail, 0 when q is finite at the LCFS
    q_integral_total: float  # integral of |q| from the axis to the LCFS
    sol_slope: float  # dPhi_N/dpsi_N outside the LCFS

    def phi_n(self, psi_n: np.ndarray) -> np.ndarray:
        """Phi_N at psi_N, any shape, NaN where psi_n is.

        psi_N below 0, which interpolation can give next to the axis, maps to 0.

        Args:
            psi_n: Normalized poloidal flux.

        Returns:
            Phi_N shaped like psi_n, exactly 1 at the LCFS.
        """
        psi_n_clipped = np.maximum(psi_n, 0.0)
        psi_n_inside = np.minimum(psi_n_clipped, 1.0)
        psi_n_interior = np.minimum(psi_n_inside, self.psi_n_join)
        q_integral_interior = self.q_integral_spline(psi_n_interior)
        q_integral_tail = self.q_integral_join + _q_tail_integral(
            self.psi_n_join, psi_n_inside, self.q_offset, self.q_log_slope
        )
        with np.errstate(invalid="ignore"):
            q_integral = np.where(
                psi_n_inside <= self.psi_n_join, q_integral_interior, q_integral_tail
            )
            phi_n_inside = q_integral / self.q_integral_total
            phi_n_inside = np.where(psi_n_inside == 1.0, 1.0, phi_n_inside)
            phi_n_outside = 1.0 + self.sol_slope * (psi_n_clipped - 1.0)
            return np.where(psi_n_clipped <= 1.0, phi_n_inside, phi_n_outside)

    def psi_n(self, phi_n: np.ndarray) -> np.ndarray:
        """psi_N at Phi_N, the inverse of PhiNMap.phi_n.

        Inside the LCFS Phi_N is inverted by interpolation on a dense table of NUM_INVERSE_POINTS,
        outside it through the linear continuation.

        Args:
            phi_n: Normalized toroidal flux, any shape, NaN where unknown.

        Returns:
            psi_N shaped like phi_n, NaN where phi_n is.
        """
        psi_n_table = np.linspace(0.0, 1.0, NUM_INVERSE_POINTS)
        phi_n_table = self.phi_n(psi_n_table)
        psi_n_inside = np.interp(phi_n, phi_n_table, psi_n_table)
        psi_n_outside = 1.0 + (phi_n - 1.0) / self.sol_slope
        with np.errstate(invalid="ignore"):
            return np.where(phi_n <= 1.0, psi_n_inside, psi_n_outside)


def phi_n_map(
    psi_n_grid: np.ndarray, qpsi: np.ndarray, sol_extension: str
) -> PhiNMap | None:
    """Build the Phi_N(psi_N) map of one equilibrium from its q profile.

    The sign of q cancels in Phi_N, so |q| is integrated once q is known to keep one sign.
    q is infinite on the surfaces of a diverted plasma where it diverges at the LCFS.
    Past the last surface of finite q it is integrated analytically
    as q = a - b ln(1 - psi_N), fit to the Q_TAIL_FIT_SURFACES surfaces inside it.
    Outside the LCFS Phi_N continues linearly in psi_N,
    with the secant slope (1 - Phi_N(SECANT_PSI_N)) / (1 - SECANT_PSI_N)
    or the tangent slope |q(1)| / integral_0^1 |q| dpsi_N,
    where q(1) is the outermost finite q when q diverges.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis to 1 at the LCFS.
        qpsi: (n_psi,) safety factor on psi_n_grid, infinite where it diverges.
        sol_extension: One of SOL_EXTENSIONS.

    Returns:
        The map, or None when the q profile is unusable:
        a NaN q, a finite q that changes sign, fewer than 2 surfaces of finite q from the axis,
        a diverging q with fewer than Q_TAIL_FIT_SURFACES finite surfaces to fit the tail to,
        a q integral that is not increasing, or a tail fit with q not positive and increasing.

    Raises:
        ValueError: If sol_extension is not one of SOL_EXTENSIONS.
    """
    if sol_extension not in SOL_EXTENSIONS:
        raise ValueError(
            f"sol_extension must be one of {SOL_EXTENSIONS}, got {sol_extension!r}"
        )
    if np.isnan(qpsi).any():
        return None
    mask_q_infinite = np.isinf(qpsi)
    num_q_finite = (
        int(np.argmax(mask_q_infinite)) if mask_q_infinite.any() else qpsi.size
    )
    q_diverges = num_q_finite < qpsi.size
    if num_q_finite < 2 or (q_diverges and num_q_finite < Q_TAIL_FIT_SURFACES):
        return None
    q_finite = qpsi[:num_q_finite]
    if not ((q_finite > 0).all() or (q_finite < 0).all()):
        return None
    psi_n_inside = psi_n_grid[:num_q_finite]
    q_inside = np.abs(q_finite)
    q_integral_inside = cumulative_q_integral(psi_n_inside, q_inside)
    q_integral_steps = np.diff(q_integral_inside)
    if not (q_integral_steps > 0).all():
        return None

    psi_n_join = float(psi_n_inside[-1])
    q_offset, q_log_slope = 0.0, 0.0
    if q_diverges:
        psi_n_fit = psi_n_inside[-Q_TAIL_FIT_SURFACES:]
        q_fit = q_inside[-Q_TAIL_FIT_SURFACES:]
        log_term_fit = -np.log(1.0 - psi_n_fit)
        constant_term_fit = np.ones_like(psi_n_fit)
        design = np.stack([constant_term_fit, log_term_fit], axis=1)
        (q_offset, q_log_slope), *_ = np.linalg.lstsq(design, q_fit, rcond=None)
        log_term_join = -np.log(1.0 - psi_n_join)
        q_at_join = q_offset + q_log_slope * log_term_join
        if q_log_slope <= 0 or q_at_join <= 0:
            return None
    q_integral_join = float(q_integral_inside[-1])
    q_integral_tail = _q_tail_integral(psi_n_join, 1.0, q_offset, q_log_slope)
    q_integral_total = q_integral_join + float(q_integral_tail)
    q_integral_spline = CubicHermiteSpline(psi_n_inside, q_integral_inside, q_inside)

    # The secant slope needs Phi_N inside the LCFS, which does not depend on the slope
    phi_n_mapping_inside = PhiNMap(
        q_integral_spline=q_integral_spline,
        psi_n_join=psi_n_join,
        q_integral_join=q_integral_join,
        q_offset=float(q_offset),
        q_log_slope=float(q_log_slope),
        q_integral_total=q_integral_total,
        sol_slope=np.nan,
    )
    if sol_extension == "secant":
        phi_n_secant_start = phi_n_mapping_inside.phi_n(SECANT_PSI_N)
        sol_slope = (1.0 - float(phi_n_secant_start)) / (1.0 - SECANT_PSI_N)
    else:
        sol_slope = q_inside[-1] / q_integral_total
    return replace(phi_n_mapping_inside, sol_slope=float(sol_slope))


def mappable_q_profiles(psi_n_grid: np.ndarray, qpsi: np.ndarray) -> np.ndarray:
    """Mark the reconstructions whose q profile gives a Phi_N map (phi_n_map).

    Whether a q profile maps does not depend on the SOL extension.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis to 1 at the LCFS.
        qpsi: (n_eq, n_psi) safety factor of each reconstruction on psi_n_grid, infinite where it diverges.

    Returns:
        (n_eq,) True where the q profile maps.
    """
    mask_mappable = [
        phi_n_map(psi_n_grid, qpsi_slice, "secant") is not None for qpsi_slice in qpsi
    ]
    return np.array(mask_mappable, dtype=bool)


def _geqdsk_phi_n_map(qpsi: np.ndarray, sol_extension: str) -> PhiNMap:
    """phi_n_map of a GEQDSK qpsi, which must be usable (usable_reconstructions).

    Args:
        qpsi: (n_psi,) safety factor on the GEQDSK psi_N grid.
        sol_extension: One of SOL_EXTENSIONS.

    Returns:
        The map.

    Raises:
        ValueError: If the q profile gives no map.
    """
    psi_n_grid = geqdsk_psi_n_grid(qpsi.size)
    phi_n_mapping = phi_n_map(psi_n_grid, qpsi, sol_extension)
    if phi_n_mapping is None:
        raise ValueError("q profile is unusable, see phi_n_map")
    return phi_n_mapping


def rho_tor_norm_from_psi_n(
    psi_n: np.ndarray, qpsi: np.ndarray, sol_extension: str
) -> np.ndarray:
    """Map normalized poloidal flux onto rho_tor_norm through one equilibrium's q profile.

    rho_tor_norm = sqrt(Phi_N), with Phi_N from phi_n_map.

    Args:
        psi_n: Normalized poloidal flux, any shape, NaN where unknown.
        qpsi: (n_psi,) safety factor on the GEQDSK psi_N grid.
        sol_extension: One of SOL_EXTENSIONS.

    Returns:
        rho_tor_norm shaped like psi_n, NaN where psi_n is.
    """
    phi_n_mapping = _geqdsk_phi_n_map(qpsi, sol_extension)
    phi_n = phi_n_mapping.phi_n(psi_n)
    return np.sqrt(phi_n)


def psi_n_from_rho_tor_norm(
    rho_tor_norm: np.ndarray, qpsi: np.ndarray, sol_extension: str
) -> np.ndarray:
    """Map rho_tor_norm back onto normalized poloidal flux, the inverse of rho_tor_norm_from_psi_n.

    Args:
        rho_tor_norm: Any shape, NaN where unknown.
        qpsi: (n_psi,) safety factor on the GEQDSK psi_N grid.
        sol_extension: One of SOL_EXTENSIONS.

    Returns:
        psi_N shaped like rho_tor_norm, NaN where rho_tor_norm is.
    """
    phi_n_mapping = _geqdsk_phi_n_map(qpsi, sol_extension)
    phi_n = np.square(rho_tor_norm)
    return phi_n_mapping.psi_n(phi_n)


def map_ts_channels_to_rho_tor_norm(
    ds_shot: xr.Dataset, sol_extension: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map TS channel (R, Z) positions onto rho_tor_norm per slice.

    Each channel's psi_N is a bilinear interpolation of the equilibrium's psirz at the channel position,
    which rho_tor_norm_from_psi_n maps through that equilibrium's qpsi.

    Only times with at least one finite TS value are mapped.
    The equilibrium is not necessarily reconstructed at each of those times
    (EFIT21 on C-Mod is native 1 kHz, but ANALYSIS runs on a ~20 ms clock),
    so each TS slice maps through the usable reconstruction (usable_reconstructions) nearest in time,
    accepted within EQ_MATCH_MAX_PERIODS of the reconstruction clock's period.
    The clock counts the unusable reconstructions too,
    so a slice whose nearest reconstruction is unusable can map through a neighbour of it.
    A slice with no usable reconstruction in reach keeps a NaN row,
    and the fit-staging min-points gate then skips it.

    Args:
        ds_shot: One shot's unprocessed dataset with standardized names
            (ts_channel_r/z, ts_channel_t_e/n_e, psirz, simagx, sibdry, qpsi, r_grid, z_grid).
        sol_extension: How Phi_N continues outside the LCFS, one of SOL_EXTENSIONS.

    Returns:
        (ts_times, rho_tor_norm): the (n_t,) times of the TS slices [s] and the
        (n_t, n_ch) channel rho_tor_norm positions, NaN where the mapping failed.
    """
    if "shot" in ds_shot.dims:
        ds_shot = ds_shot.squeeze("shot", drop=True)

    ts_valid = ds_shot["ts_channel_t_e"].notnull() | ds_shot["ts_channel_n_e"].notnull()
    ts_mask = ts_valid.any(dim="ts_channel").transpose("time").values
    ts_idxs = np.flatnonzero(ts_mask)  # Indices where TS exists
    ts_times = ds_shot["time"].values[ts_idxs]

    # guarantee load has correct array layout with explicit named dimension transpose
    psirz = ds_shot["psirz"].transpose("time", "r_grid", "z_grid").values
    simagx = ds_shot["simagx"].transpose("time").values
    sibdry = ds_shot["sibdry"].transpose("time").values
    qpsi = ds_shot["qpsi"].transpose("time", "psi_idx").values
    ts_r = ds_shot["ts_channel_r"].transpose("time", "ts_channel").values
    ts_z = ds_shot["ts_channel_z"].transpose("time", "ts_channel").values
    r_grid = ds_shot["r_grid"].values
    z_grid = ds_shot["z_grid"].values

    # Each TS slice maps through the usable reconstruction nearest in time,
    # in reach of the reconstruction clock, unusable reconstructions included.
    all_times = ds_shot["time"].values
    usable = usable_reconstructions(ds_shot)
    eq_rows = np.flatnonzero(usable)
    eq_times = all_times[eq_rows]
    eq_period = reconstruction_clock_period(ds_shot, all_times)
    eq_tol = EQ_MATCH_MAX_PERIODS * eq_period

    rho_tor_norm = np.full((ts_idxs.size, ts_r.shape[1]), np.nan)
    n_no_equilibrium = 0
    for i, ts_idx in enumerate(ts_idxs):
        if eq_rows.size == 0:
            n_no_equilibrium += 1
            continue
        nearest = int(np.argmin(np.abs(eq_times - ts_times[i])))
        if abs(eq_times[nearest] - ts_times[i]) > eq_tol:
            n_no_equilibrium += 1
            continue
        eq_idx = int(eq_rows[nearest])
        psi_range = sibdry[eq_idx] - simagx[eq_idx]
        qpsi_slice = qpsi[eq_idx]
        psi_n_grid = (psirz[eq_idx] - simagx[eq_idx]) / psi_range

        # Channel psi_n at the measured (R, Z)
        # NaN positions or positions off the grid stay NaN.
        interp = RegularGridInterpolator(
            (r_grid, z_grid), psi_n_grid, bounds_error=False, fill_value=np.nan
        )
        with np.errstate(invalid="ignore"):
            psi_n_ch = interp(np.column_stack([ts_r[ts_idx], ts_z[ts_idx]]))
        rho_tor_norm[i, :] = rho_tor_norm_from_psi_n(
            psi_n_ch, qpsi_slice, sol_extension
        )

    if n_no_equilibrium:
        logger.debug(
            f"No usable equilibrium at {n_no_equilibrium} of {ts_idxs.size} TS slices"
        )
    return ts_times, rho_tor_norm


def channel_rows_at_times(data: xr.DataArray, ts_times: np.ndarray) -> np.ndarray:
    """Take the rows of a (time, ts_channel) variable at the TS slice times.

    Args:
        data: Channel variable carrying a "time" coordinate.
        ts_times: TS slice times [s], as returned by map_ts_channels_to_rho_tor_norm.

    Returns:
        The (n_t, n_ch) rows at those times.
    """
    values = data.transpose("time", "ts_channel").values
    return values[np.isin(data["time"].values, ts_times)]


def ts_channel_fit_rows(
    ds_shot: xr.Dataset, ts_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the TS channel rows at the slice times, converted to the fit units.

    The unprocessed files are SI (Te [eV], ne [m^-3]); the fits and every
    device's cleaning threshold are calibrated in Te [keV] and ne [1e20 m^-3].

    Args:
        ds_shot: One shot's unprocessed dataset, shot dim squeezed out.
        ts_times: TS slice times [s].

    Returns:
        (te_y, te_err, ne_y, ne_err), each (n_t, n_ch) in the fit units.
    """
    rows = {
        name: np.asarray(
            channel_rows_at_times(ds_shot[f"ts_channel_{name}"], ts_times), float
        )
        for name in ("t_e", "t_e_error", "n_e", "n_e_error")
    }
    return (
        rows["t_e"] * 1e-3,
        rows["t_e_error"] * 1e-3,
        rows["n_e"] * 1e-20,
        rows["n_e_error"] * 1e-20,
    )


def cocos_from_signs(
    current, bcentr, simagx, sibdry, qpsi, logger_override=None
) -> int:
    """Identify the COCOS of a shot's reconstructions from the signs of their own data.

    The shot medians of Ip, B0, psi_boundary - psi_axis and q fix sigma_Bp and sigma_rho_theta_phi
    (eqdsk.cocos.identify_cocos), with phi counterclockwise from above and psi per radian.
    C-Mod's EFIT keeps psi increasing outward and q > 0, so its shots are COCOS 1 or 7 with the field direction.
    MAST's EFIT has psi decreasing outward with Ip > 0, B0 < 0 and q > 0, which is COCOS 3.

    Args:
        current: (n_t,) plasma current [A].
        bcentr: (n_t,) vacuum toroidal field [T].
        simagx: (n_t,) poloidal flux at the magnetic axis [Wb/rad].
        sibdry: (n_t,) poloidal flux at the plasma boundary [Wb/rad].
        qpsi: (n_t, n_psi) safety factor.
        logger_override: Logger for the warning, the module logger when None.

    Returns:
        COCOS number (1, 3, 5, or 7).
        1 with a warning when a median is NaN or 0, which means no usable reconstruction.
    """
    if logger_override is None:
        logger_override = logger
    psi_rise = np.asarray(sibdry, dtype=float) - np.asarray(simagx, dtype=float)
    current_median = np.nanmedian(current)
    bcentr_median = np.nanmedian(bcentr)
    psi_rise_median = np.nanmedian(psi_rise)
    q_median = np.nanmedian(qpsi)
    medians = np.array([current_median, bcentr_median, psi_rise_median, q_median])
    if not np.all(np.isfinite(medians) & (medians != 0.0)):
        logger_override.warning(
            "No finite signs of Ip, B0, psi and q to identify the COCOS from. Assuming COCOS 1."
        )
        return 1
    # Only the sign of the boundary-to-axis rise matters, so the median rise stands in for the boundary
    cocos = identify_cocos(
        plasma_current=current_median,
        b_toroidal=bcentr_median,
        psi_at_boundary=psi_rise_median,
        psi_at_mag_axis=0.0,
        q_psi=np.array([q_median]),
        phi_clockwise_from_top=False,
        volt_seconds_per_radian=True,
    )
    return cocos.index


def orient_signal(geqdsk_data, efit_time):
    """Orient every retrieved signal time-first and transpose psirz to (T, r, z).

    No interpolation: each signal is kept on its native per-timeslice EFIT grid.
    Quality flags and reconstructions are not meaningful when interpolated, so
    callers must use time_setting="efit" to keep the tree's own timebase.
    Mutates and returns the dict.

    Returns:
        The mutated geqdsk_data dict.
    """

    def _time_first(data, n_time):
        """Move the axis whose length equals n_time to axis 0.

        Leaves 1D arrays and arrays already time-first unchanged. Used to
        normalise the per-machine MDS layouts (some store profiles/boundary as
        (spatial, T)).

        Returns:
            The array with time on axis 0.
        """
        if data.ndim < 2:
            return data
        if data.shape[0] == n_time:
            return data
        matches = [ax for ax, n in enumerate(data.shape) if n == n_time]
        if matches:
            return np.moveaxis(data, matches[0], 0)
        return data

    n_time = len(efit_time)
    for param, data in geqdsk_data.items():
        data = _time_first(data, n_time)
        # psirz comes back (T, z, r) from MDS (dim_of order reversed vs numpy);
        # the dataset labels dims (idx, r_grid, z_grid), so swap to (T, r, z).
        if param == "psirz" and data.ndim == 3:
            data = data.transpose(0, 2, 1)
        geqdsk_data[param] = data
    return geqdsk_data
