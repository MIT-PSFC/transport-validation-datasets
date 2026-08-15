import numpy as np
import xarray as xr
from loguru import logger
from scipy.interpolate import RegularGridInterpolator


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


def _lcfs_crossing_radius(
    r_from_axis: np.ndarray, psi_n_from_axis: np.ndarray
) -> float:
    """Find the midplane radius where psi_n first crosses 1, walking outward.

    Both arrays must be ordered starting at the axis and moving outward.

    Args:
        r_from_axis: Midplane radii, axis outward [m].
        psi_n_from_axis: Normalized poloidal flux at those radii.

    Returns:
        The linearly interpolated crossing radius [m], or NaN if psi_n never
        reaches 1.
    """
    above = psi_n_from_axis >= 1.0
    if not above.any():
        return np.nan
    idx = int(np.argmax(above))
    if idx == 0:
        return float(r_from_axis[0])
    r0, r1 = float(r_from_axis[idx - 1]), float(r_from_axis[idx])
    p0, p1 = float(psi_n_from_axis[idx - 1]), float(psi_n_from_axis[idx])
    if p1 == p0:
        return r1
    return r0 + (1.0 - p0) * (r1 - r0) / (p1 - p0)


def _refine_axis_radius(
    r_grid: np.ndarray, psi_n_mid: np.ndarray, i_axis: int
) -> float:
    """Refine the magnetic axis radius from the midplane psi_n minimum.

    Parabola-refined around the grid minimum, since the EFIT grid is coarse
    (a few cm).

    Args:
        r_grid: Midplane radii [m].
        psi_n_mid: Normalized poloidal flux along the midplane.
        i_axis: Index of the psi_n_mid minimum.

    Returns:
        The refined axis radius [m].
    """
    r_axis = float(r_grid[i_axis])
    if 0 < i_axis < len(r_grid) - 1:
        p_m = psi_n_mid[i_axis - 1]
        p_0 = psi_n_mid[i_axis]
        p_p = psi_n_mid[i_axis + 1]
        curv = p_m - 2 * p_0 + p_p
        if curv > 0:
            r_axis += (
                0.5
                * (p_m - p_p)
                / curv
                * float(r_grid[i_axis + 1] - r_grid[i_axis - 1])
                / 2.0
            )
    return r_axis


def map_ts_channels_to_rho(ds_shot: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Map TS channel (R, Z) positions onto normalized minor radius per slice.

    rho is the normalized outboard midplane minor radius (0 = axis, 1 = LCFS),
    each channel's psi_n (bilinear interpolation of the equilibrium's psirz at the channel position)
    is inverted through the midplane psi_n profile at the magnetic axis height to the outboard
    midplane radius, then normalized by the axis-to-LCFS distance. Only the
    outboard inversion branch is used: mapping through psi_n always lands a
    channel on the outboard side, regardless of which side it was measured on.

    Only times with at least one finite TS value are mapped.
    A slice whose equilibrium is missing or degenerate keeps a NaN rho row,
    the fit-staging min-points gate then skips it.

    Args:
        ds_shot: One shot's unprocessed dataset with standardized names
            (ts_channel_r/z, ts_channel_t_e/n_e, psirz, simagx, sibdry, zmagx,
            r_grid, z_grid).

    Returns:
        (ts_times, rho): the (n_t,) times of the TS slices [s] and the
        (n_t, n_ch) channel rho positions, NaN where the mapping failed.
    """
    if "shot" in ds_shot.dims:
        ds_shot = ds_shot.squeeze("shot", drop=True)

    ts_values = (
        ds_shot["ts_channel_t_e"].notnull() | ds_shot["ts_channel_n_e"].notnull()
    )
    ts_mask = ts_values.any(dim="ts_channel").transpose("time").values
    ts_idx = np.flatnonzero(ts_mask)
    ts_times = ds_shot["time"].values[ts_idx]

    # Slice via named dims: the on-disk dim order of the merged file differs
    # from the retrieval-time order.
    psirz = ds_shot["psirz"].transpose("time", "r_grid", "z_grid").values
    simagx = ds_shot["simagx"].transpose("time").values
    sibdry = ds_shot["sibdry"].transpose("time").values
    zmagx = ds_shot["zmagx"].transpose("time").values
    ts_r = ds_shot["ts_channel_r"].transpose("time", "ts_channel").values
    ts_z = ds_shot["ts_channel_z"].transpose("time", "ts_channel").values
    r_grid = ds_shot["r_grid"].values
    z_grid = ds_shot["z_grid"].values

    rho = np.full((ts_idx.size, ts_r.shape[1]), np.nan)
    n_no_equilibrium = 0
    for row, i_t in enumerate(ts_idx):
        denom = sibdry[i_t] - simagx[i_t]
        psi_slice = psirz[i_t]
        if (
            not np.isfinite(denom)
            or np.abs(denom) < 1e-10
            or not np.isfinite(zmagx[i_t])
            or not np.all(np.isfinite(psi_slice))
        ):
            n_no_equilibrium += 1
            continue
        psi_n_grid = (psi_slice - simagx[i_t]) / denom

        # Channel psi_n at the measured (R, Z)
        # NaN positions or positions off the grid stay NaN.
        interp = RegularGridInterpolator(
            (r_grid, z_grid), psi_n_grid, bounds_error=False, fill_value=np.nan
        )
        with np.errstate(invalid="ignore"):
            psi_n_ch = interp(np.column_stack([ts_r[i_t], ts_z[i_t]]))

        # psi_n along the midplane (z = magnetic axis height)
        psi_n_mid = np.array(
            [
                np.interp(zmagx[i_t], z_grid, psi_n_grid[j, :])
                for j in range(len(r_grid))
            ]
        )
        i_axis = int(np.argmin(psi_n_mid))
        r_axis = _refine_axis_radius(r_grid, psi_n_mid, i_axis)
        r_lcfs_out = _lcfs_crossing_radius(r_grid[i_axis:], psi_n_mid[i_axis:])
        if not np.isfinite(r_lcfs_out) or r_lcfs_out <= r_axis:
            n_no_equilibrium += 1
            continue

        # Invert psi_n to the outboard midplane radius. The outboard branch is
        # monotone increasing inside the LCFS but can fold in the far SOL, so
        # keep only its running-maximum points; channels beyond the last kept
        # psi_n clamp to the grid edge and are cut later by the rho > 1
        # cleaning.
        psi_out = psi_n_mid[i_axis:]
        r_out = r_grid[i_axis:]
        keep = psi_out == np.maximum.accumulate(psi_out)
        r_mid_ch = np.interp(psi_n_ch, psi_out[keep], r_out[keep])
        with np.errstate(invalid="ignore"):
            rho[row, :] = (r_mid_ch - r_axis) / (r_lcfs_out - r_axis)

    if n_no_equilibrium:
        logger.debug(
            f"No usable equilibrium at {n_no_equilibrium} of {ts_idx.size} TS slices"
        )
    return ts_times, rho


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
