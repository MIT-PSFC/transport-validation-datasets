import numpy as np
import xarray as xr
from loguru import logger


def make_uniform_1kHz_timebase(max_time: float) -> np.ndarray:
    """Create a uniform timebase at 1 kHz up to the specified maximum time.

    This is the timebase used for all datasets.
    Built from an integer millisecond count to avoid problems with float accumulation.

    Args:
        max_time: The maximum time for the timebase [s].

    Returns:
        Times from 0 to max_time in 1 ms steps [s].
    """
    last_ms = int(np.ceil(np.round(max_time * 1000, 6)))
    times = np.round(np.arange(last_ms + 1, dtype=np.float64) * 1e-3, 3).astype(
        "float32"
    )
    return times


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
    rlim=None,
    zlim=None,
):
    """Build an Xarray dataset holding every signal needed to recreate a GEQDSK file.

    FreeQDSK canonical names, COCOS 1.

    Returns:
        Dataset with all GEQDSK signals on dim 'idx', with 'time'/'shot' coords.
    """
    rcentr = r_grid[len(r_grid) // 2]
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

    return ds_geqdsk


def snap_to_grid(ds: xr.Dataset, grid_times: np.ndarray) -> xr.Dataset:
    """Snap an EFIT reconstruction (dim 'idx', 'time'/'shot' coords) onto grid_times.

    Nearest-neighbor snap with tie-break toward later grid points. No interpolation, no fill.

    Each EFIT slice goes to its nearest grid time, with no interpolation of the
    reconstruction. The snap only absorbs sub-step machine jitter in the EFIT clock,
    so ties break toward the later grid point: an EFIT at 9.5 ms on a 1 ms grid lands
    at 10 ms, never feeding the 9 ms slice a future reconstruction. Grid times with no
    EFIT slice (e.g. before the first reconstruction) come back as NaN

    Args:
        ds: EFIT reconstruction with dim 'idx' and 'time'/'shot' coords.
        grid_times: Uniform timebase to snap onto [s].

    Returns:
        The reconstruction on grid_times, NaN at grid times with no EFIT slice.
    """
    grid_times = np.asarray(grid_times)
    efit_times = ds["time"].values
    shot_id = ds["shot"].values[0]

    # Nearest grid point per EFIT slice
    # ties (and float noise within a hair of a half-step) break toward the later point,
    # so the snap only ever absorbs sub-step jitter and never feeds an earlier slice a future reconstruction
    # tol sits well below any real EFIT timing gap but far above float32 round-off at the grid step.
    pos = np.searchsorted(grid_times, efit_times, side="left")
    left = np.clip(pos - 1, 0, len(grid_times) - 1)
    right = np.clip(pos, 0, len(grid_times) - 1)
    tol = 1e-4 * np.median(np.diff(grid_times))
    take_left = (grid_times[right] - efit_times) - (efit_times - grid_times[left]) > tol
    target = np.where(take_left, left, right)

    # snapped is non-decreasing (both arrays sorted), so duplicate slots are adjacent
    # keep the last EFIT slice that lands in each, then NaN-fill the empty grid times
    snapped = grid_times[target]
    keep = np.append(np.diff(snapped) != 0, True)
    ds = ds.drop_vars(["time", "shot"]).assign_coords(idx=snapped).isel(idx=keep)
    ds = ds.reindex(idx=grid_times).reset_index("idx", drop=True)
    return ds.assign_coords(
        time=("idx", grid_times),
        shot=("idx", np.repeat(shot_id, len(grid_times))),
    )


def efit_cocos_from_signs(current, bcentr, logger_override=None) -> int:
    """Identify the EFIT COCOS from the signs of median Ip and B0.

    See https://efit-ai.gitlab.io/efit/files.html

    Returns:
        COCOS number (1, 3, 5, or 7). Falls back to 1 for unexpected sign combinations.
    """
    if logger_override is None:
        logger_override = logger
    sign_ip = np.sign(np.nanmedian(current))
    sign_b0 = np.sign(np.nanmedian(bcentr))
    if sign_ip > 0 and sign_b0 > 0:
        return 1
    if sign_ip < 0 and sign_b0 > 0:
        return 3
    if sign_ip > 0 and sign_b0 < 0:
        return 5
    if sign_ip < 0 and sign_b0 < 0:
        return 7
    logger_override.warning(
        "Unexpected sign combination for current and magnetic field. Assuming COCOS 1."
    )
    return 1


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
        if data is None or data.ndim < 2:
            return data
        if data.shape[0] == n_time:
            return data
        matches = [ax for ax, n in enumerate(data.shape) if n == n_time]
        if matches:
            return np.moveaxis(data, matches[0], 0)
        return data

    n_time = len(efit_time)
    for param, data in geqdsk_data.items():
        if data is None:
            continue
        data = _time_first(data, n_time)
        # psirz comes back (T, z, r) from MDS (dim_of order reversed vs numpy);
        # the dataset labels dims (idx, r_grid, z_grid), so swap to (T, r, z).
        if param == "psirz" and data.ndim == 3:
            data = data.transpose(0, 2, 1)
        geqdsk_data[param] = data
    return geqdsk_data
