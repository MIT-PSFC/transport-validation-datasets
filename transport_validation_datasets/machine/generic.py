import re

import numpy as np
import xarray as xr
from loguru import logger
from scipy.interpolate import RegularGridInterpolator

IMAS_DOCS_URL = "https://imas-data-dictionary.readthedocs.io/en/latest/generated/ids"

# Attributes of the GEQDSK block make_geqdsk_dataset builds, freeqdsk names.
# Units are those of COCOS 1 to 8, the range efit_cocos_from_signs covers,
# where the poloidal flux is per radian; the shot's COCOS number rides on the
# dataset's "cocos" attribute and in the store as the per-shot cocos variable.
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


# How far a TS slice may sit from the equilibrium reconstruction it maps
# through (map_ts_channels_to_rho), in periods of the reconstruction's own
# sampling. The mapping-time mirror of workflow.MAX_HOLD_PERIODS (its own
# constant to avoid a circular import): above 1 to tolerate clock jitter,
# low enough that nothing is borrowed across a real gap.
EQ_MATCH_MAX_PERIODS = 1.5


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


def _lcfs_crossing_radius(
    r_from_axis: np.ndarray, psi_n_from_axis: np.ndarray
) -> float:
    """Find the midplane radius where psi_n first crosses 1, walking outward.

    Both arrays must be ordered starting at the axis and moving outward.

    Args:
        r_from_axis: Midplane radii, axis outward [m].
        psi_n_from_axis: Normalized poloidal flux at those radii.

    Returns:
        The linearly interpolated crossing radius [m], or NaN if psi_n never reaches 1.
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
    r_cross = r0 + (1.0 - p0) * (r1 - r0) / (p1 - p0)
    return r_cross


def _refine_axis_radius(
    r_grid: np.ndarray, psi_n_mid: np.ndarray, i_axis: int
) -> float:
    """Refine the magnetic axis radius from the midplane psi_n minimum.

    When defining rho = (r - r_axis) / (r_lcfs - r_axis), must know where the axis is.
    EFIT grid can be coarse (a few cm), so may get rho errors ~ 5% (worst in the core).
    Here, use a simple 3-point parabola fit to refine the axis radius.

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

    rho is the normalized minor radius (0 = axis, 1 = LCFS),
    each channel's psi_n (bilinear interpolation of the equilibrium's psirz at the channel position)
    is inverted through the midplane psi_n profile at the magnetic axis height to the outboard
    midplane radius, then normalized by the axis-to-LCFS distance.
    Choice to use outboard midplane is arbitrary, could use any line from magnetic axis to the LCFS.
    This one is convenient though because it is one dimension (R) and increases with psi.

    Only times with at least one finite TS value are mapped. The equilibrium
    is not necessarily reconstructed at every one of those grid times
    (C-Mod's EFIT21 is native 1 kHz, but its ANALYSIS tree runs on its own
    ~20 ms clock), so each TS slice maps through the reconstruction nearest
    in time, accepted within EQ_MATCH_MAX_PERIODS of the reconstruction's
    own sampling period so nothing is borrowed across a real gap.
    A slice with no reconstruction in reach, or a degenerate one, keeps a
    NaN rho row; the fit-staging min-points gate then skips it.

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

    ts_valid = ds_shot["ts_channel_t_e"].notnull() | ds_shot["ts_channel_n_e"].notnull()
    ts_mask = ts_valid.any(dim="ts_channel").transpose("time").values
    ts_idxs = np.flatnonzero(ts_mask)  # Indices where TS exists
    ts_times = ds_shot["time"].values[ts_idxs]

    # guarantee load has correct array layout with explicit named dimension transpose
    psirz = ds_shot["psirz"].transpose("time", "r_grid", "z_grid").values
    simagx = ds_shot["simagx"].transpose("time").values
    sibdry = ds_shot["sibdry"].transpose("time").values
    zmagx = ds_shot["zmagx"].transpose("time").values
    ts_r = ds_shot["ts_channel_r"].transpose("time", "ts_channel").values
    ts_z = ds_shot["ts_channel_z"].transpose("time", "ts_channel").values
    r_grid = ds_shot["r_grid"].values
    z_grid = ds_shot["z_grid"].values

    # Reconstruction times, for the nearest-in-time equilibrium match. A
    # single reconstruction has no period of its own, so the grid step
    # stands in (mirroring workflow._hold_onto_grid's lone-sample fallback).
    all_times = ds_shot["time"].values
    eq_rows = np.flatnonzero(np.isfinite(simagx))
    eq_times = all_times[eq_rows]
    if eq_times.size > 1:
        eq_period = float(np.median(np.diff(eq_times)))
    elif all_times.size > 1:
        eq_period = float(np.median(np.diff(all_times)))
    else:
        eq_period = 0.0
    eq_tol = EQ_MATCH_MAX_PERIODS * eq_period

    rho = np.full((ts_idxs.size, ts_r.shape[1]), np.nan)
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

        # get psi_n for the reconstruction this timeslice maps through
        psi_range = sibdry[eq_idx] - simagx[eq_idx]
        psi_slice = psirz[eq_idx]
        if (
            not np.isfinite(psi_range)  # psi range NaN or inf
            or np.abs(psi_range) < 1e-10  # psi range too small to be physical
            or not np.isfinite(zmagx[eq_idx])  # Z magnetic axis NaN or inf
            or not np.all(np.isfinite(psi_slice))  # psi slice has NaN or inf
        ):
            n_no_equilibrium += 1
            continue
        psi_n_grid = (psi_slice - simagx[eq_idx]) / psi_range

        # Channel psi_n at the measured (R, Z)
        # NaN positions or positions off the grid stay NaN.
        interp = RegularGridInterpolator(
            (r_grid, z_grid), psi_n_grid, bounds_error=False, fill_value=np.nan
        )
        with np.errstate(invalid="ignore"):
            psi_n_ch = interp(np.column_stack([ts_r[ts_idx], ts_z[ts_idx]]))

        # psi_n along the midplane (z = magnetic axis height)
        psi_n_mid = np.array(
            [
                np.interp(zmagx[eq_idx], z_grid, psi_n_grid[j, :])
                for j in range(len(r_grid))
            ]
        )
        i_axis = int(np.argmin(psi_n_mid))
        r_axis = _refine_axis_radius(r_grid, psi_n_mid, i_axis)
        r_lcfs_outboard = _lcfs_crossing_radius(r_grid[i_axis:], psi_n_mid[i_axis:])
        if not np.isfinite(r_lcfs_outboard) or r_lcfs_outboard <= r_axis:
            n_no_equilibrium += 1
            continue

        # Map each channel's psi_n to the outboard midplane radius, then normalize to rho.
        # Anchor the table on the refined axis, where psi_n is 0 by definition, and keep
        # only the grid nodes outboard of it (otherwise might get negative rho)
        outboard = r_grid[i_axis:] > r_axis
        psi_outboard = np.concatenate([[0.0], psi_n_mid[i_axis:][outboard]])
        r_outboard = np.concatenate([[r_axis], r_grid[i_axis:][outboard]])
        if r_outboard.size < 2:
            n_no_equilibrium += 1
            continue
        keep = psi_outboard == np.maximum.accumulate(
            psi_outboard
        )  # Ensure monotonicity for interp
        r_mid_ch = np.interp(psi_n_ch, psi_outboard[keep], r_outboard[keep])
        with np.errstate(invalid="ignore"):
            rho[i, :] = (r_mid_ch - r_axis) / (r_lcfs_outboard - r_axis)

    if n_no_equilibrium:
        logger.debug(
            f"No usable equilibrium at {n_no_equilibrium} of {ts_idxs.size} TS slices"
        )
    return ts_times, rho


def channel_rows_at_times(data: xr.DataArray, ts_times: np.ndarray) -> np.ndarray:
    """Take the rows of a (time, ts_channel) variable at the TS slice times.

    Args:
        data: Channel variable carrying a "time" coordinate.
        ts_times: TS slice times [s], as returned by map_ts_channels_to_rho.

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
