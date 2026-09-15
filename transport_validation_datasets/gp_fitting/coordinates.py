"""Radial coordinate transforms and Jacobians for GP profile fitting.

Needs only numpy + scipy -- both are on the GP fitting cluster venv (see
bootstrap_remote.sh: `uv pip install numpy scipy mkgp`), same as
akho/fit_functions.py's own `scipy.optimize.curve_fit` dependency. What is
NOT on that venv is xarray/loguru/disruption_py, so this module -- unlike
machine/generic.py, which needs the full extraction stack to fetch and merge
raw signals -- deliberately does not import from `machine`, even though the
axis-refinement/LCFS-crossing helpers below are conceptually about
equilibrium geometry: `machine.generic` imports these from here instead, to
keep a single implementation without pulling xarray/loguru into a worker's
import chain. That is the actual constraint this module is shaped around,
not scipy avoidance.

By the time a worker sees a shot, `psi_norm` (and `rho`, computed
separately) are already channel-level arrays on `ShotFitInput` -- staged
once via `map_ts_channels_to_flux_coordinates`/`psi_norm_at_positions`
against the full equilibrium (`psirz`, `r_grid`, `z_grid`, ...), which a
worker never touches directly. `coordinates_from_psi_norm` is what a worker
calls, with `psi_norm` as the pivot and no equilibrium in hand
(`midplane_map=None`, since `rho` arrives pre-staged); `psi_norm_at_positions`
and `build_midplane_flux_map` are for staging-time/diagnostic callers that
do have the raw equilibrium fields (plots, IMAS export, ad hoc analysis).

Every one of `psi_norm`, `sqrt(psi_norm)`, `phi_norm`, `sqrt(phi_norm)` (and
`rho`, when a `MidplaneFluxMap` is supplied) is a function of `psi_norm`
alone for a given equilibrium time slice, so every Jacobian here is d(.)/
d(psi_norm) -- `transform_gradient` combines any two of them by the chain
rule to re-express a GP fit's derivative from one coordinate into another.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Floor used wherever a Jacobian's analytic denominator is a coordinate value
# that can be exactly 0 (the magnetic axis, where d(sqrt(x))/dx diverges).
# Keeps every returned Jacobian finite instead of inf/NaN, at the cost of
# capping how steep it can report right at the axis.
_JACOBIAN_EPS = 1.0e-6

# Fallback for d(phi_norm)/d(psi_norm) when qpsi is entirely missing for a
# slice (MAST's qpsi is a best-effort level-1 signal, see
# machine/mast/mast_dataset.py's `_equilibrium_qpsi`). phi_norm itself stays
# NaN in that case -- there is no way to recover a missing q profile -- but
# its Jacobian defaults to 1.0 (treats phi_norm ~= psi_norm locally) so a
# caller converting an otherwise-valid gradient does not get NaN injected by
# the Jacobian alone. This is a modeling choice, not a measurement: flag it
# if it is hit often for a given machine/shot range.
_MISSING_Q_JACOBIAN_FALLBACK = 1.0


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


@dataclass
class MidplaneFluxMap:
    """One equilibrium time slice's outboard-midplane psi_norm -> radius map.

    Built at staging time by `build_midplane_flux_map` (needs the raw psirz
    equilibrium, not available to a worker); a worker that already has
    staged `rho`/`psi_norm` per channel never needs to build one of these
    itself, so `coordinates_from_psi_norm` treats it as optional and skips
    `rho` (NaN) when it is not supplied.
    """

    r_axis: float
    r_lcfs_outboard: float
    r_outboard: np.ndarray  # (n,) midplane radii, axis outward, monotonic in psi_norm [m]
    psi_norm_outboard: np.ndarray  # (n,) psi_norm at those radii, monotonic increasing


def build_midplane_flux_map(
    psirz_slice: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    simagx: float,
    sibdry: float,
    zmagx: float,
) -> MidplaneFluxMap | None:
    """Build one time slice's outboard-midplane flux map.

    Mirrors the per-slice logic in `machine.generic.map_ts_channels_to_rho`
    exactly (down to sharing its axis-refinement/LCFS-crossing helpers
    above), so `rho` computed through this map agrees with the `rho` GP
    fitting is staged on -- see tests/test_flux_coordinates.py's
    cross-consistency check against `map_ts_channels_to_rho` itself.

    Args:
        psirz_slice: (n_r, n_z) poloidal flux at this time [COCOS units].
        r_grid: Midplane R grid the flux map is defined on [m].
        z_grid: Z grid the flux map is defined on [m].
        simagx: Poloidal flux at the magnetic axis, this time.
        sibdry: Poloidal flux at the LCFS, this time.
        zmagx: Magnetic axis height, this time [m].

    Returns:
        The flux map, or None if the equilibrium is missing/degenerate at
        this time (mirrors `map_ts_channels_to_rho`'s own checks) -- callers
        should treat every coordinate as NaN for such a slice; there is no
        reasonable default for a wholly missing equilibrium.
    """
    psi_range = sibdry - simagx
    if (
        not np.isfinite(psi_range)
        or np.abs(psi_range) < 1e-10
        or not np.isfinite(zmagx)
        or not np.all(np.isfinite(psirz_slice))
    ):
        return None
    psi_n_grid = (psirz_slice - simagx) / psi_range

    psi_n_mid = np.array(
        [np.interp(zmagx, z_grid, psi_n_grid[j, :]) for j in range(len(r_grid))]
    )
    i_axis = int(np.argmin(psi_n_mid))
    r_axis = _refine_axis_radius(r_grid, psi_n_mid, i_axis)
    r_lcfs_outboard = _lcfs_crossing_radius(r_grid[i_axis:], psi_n_mid[i_axis:])
    if not np.isfinite(r_lcfs_outboard) or r_lcfs_outboard <= r_axis:
        return None

    psi_outboard = psi_n_mid[i_axis:]
    r_outboard = r_grid[i_axis:]
    keep = psi_outboard == np.maximum.accumulate(psi_outboard)
    return MidplaneFluxMap(
        r_axis=r_axis,
        r_lcfs_outboard=r_lcfs_outboard,
        r_outboard=r_outboard[keep],
        psi_norm_outboard=psi_outboard[keep],
    )


def psi_norm_at_positions(
    psirz_slice: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    simagx: float,
    sibdry: float,
    r: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Interpolate psi_norm at arbitrary (R, Z) points for one time slice.

    Args:
        psirz_slice: (n_r, n_z) poloidal flux at this time.
        r_grid: R grid psirz_slice is defined on [m].
        z_grid: Z grid psirz_slice is defined on [m].
        simagx: Poloidal flux at the magnetic axis, this time.
        sibdry: Poloidal flux at the LCFS, this time.
        r: Query major radii [m], any shape.
        z: Query heights [m], same shape as `r`.

    Returns:
        psi_norm at each (r, z), shaped like `r`. NaN off-grid or where the
        equilibrium is degenerate this time.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    psi_range = sibdry - simagx
    if not np.isfinite(psi_range) or np.abs(psi_range) < 1e-10:
        return np.full(r.shape, np.nan)
    psi_n_grid = (np.asarray(psirz_slice, dtype=float) - simagx) / psi_range
    interp = RegularGridInterpolator(
        (r_grid, z_grid), psi_n_grid, bounds_error=False, fill_value=np.nan
    )
    with np.errstate(invalid="ignore"):
        return interp(np.column_stack([r.ravel(), z.ravel()])).reshape(r.shape)


def _toroidal_flux_unnorm(qpsi: np.ndarray) -> np.ndarray:
    """Cumulative-trapezoid integral of qpsi over its own uniform [0, 1] psi_norm grid.

    Returns:
        (n_psi,) unnormalized toroidal flux at the same grid points.
    """
    psi_norm_grid = np.linspace(0.0, 1.0, qpsi.size)
    return np.concatenate(
        [[0.0], np.cumsum(0.5 * (qpsi[1:] + qpsi[:-1]) * np.diff(psi_norm_grid))]
    )


def toroidal_flux_norm_profile(qpsi: np.ndarray) -> np.ndarray:
    """Normalized toroidal flux phi_norm(psi_norm) on qpsi's own uniform psi_norm grid.

    phi_norm = (integral of q from 0 to psi_norm) / (integral of q from 0 to 1),
    via cumulative trapezoidal integration over a uniform [0, 1] psi_norm grid
    of the same length as qpsi -- the GEQDSK 1D-profile convention every
    machine backend stages qpsi on (see machine/generic.py's
    `make_geqdsk_dataset`, dim `psi_idx`).

    Args:
        qpsi: (n_psi,) safety factor on the uniform psi_norm grid [0, 1]. NaN
            (qpsi entirely missing, e.g. MAST's best-effort level-1 signal --
            see mast_dataset.py's `_equilibrium_qpsi`) propagates to an
            all-NaN profile; there is no reasonable default for a genuinely
            missing q profile.

    Returns:
        (n_psi,) phi_norm at the same grid points, NaN if qpsi is missing or
        degenerate (e.g. integrates to exactly 0).
    """
    qpsi = np.asarray(qpsi, dtype=float)
    if qpsi.size == 0 or not np.all(np.isfinite(qpsi)):
        return np.full_like(qpsi, np.nan)
    phi_unnorm = _toroidal_flux_unnorm(qpsi)
    total = phi_unnorm[-1]
    if not np.isfinite(total) or total == 0:
        return np.full_like(qpsi, np.nan)
    return phi_unnorm / total


@dataclass
class CoordinateValues:
    """Every radial coordinate at one set of query points, one time slice."""

    psi_norm: np.ndarray
    sqrt_psi_norm: np.ndarray
    phi_norm: np.ndarray
    sqrt_phi_norm: np.ndarray
    rho: np.ndarray


@dataclass
class CoordinateJacobians:
    """d(coordinate)/d(psi_norm) at the same points as a `CoordinateValues`.

    The common pivot every pairwise Jacobian is built from via the chain
    rule -- see `transform_gradient`.
    """

    psi_norm: np.ndarray  # identically 1.0
    sqrt_psi_norm: np.ndarray
    phi_norm: np.ndarray
    sqrt_phi_norm: np.ndarray
    rho: np.ndarray


def coordinates_from_psi_norm(
    psi_norm: np.ndarray,
    midplane_map: MidplaneFluxMap | None,
    qpsi: np.ndarray | None,
) -> tuple[CoordinateValues, CoordinateJacobians]:
    """Compute every coordinate and its d(.)/d(psi_norm) Jacobian at given psi_norm points.

    Args:
        psi_norm: Query points' normalized poloidal flux, any shape.
        midplane_map: This time slice's outboard-midplane flux map (see
            `build_midplane_flux_map`), or None if
            unavailable (the usual case inside a worker, which already has
            `rho` staged separately) or if the equilibrium was missing/
            degenerate this slice -- rho and its Jacobian come back all-NaN
            either way; there is no reasonable default for a missing map.
        qpsi: This time slice's (n_psi,) safety factor profile, or None/
            all-NaN if unavailable -- phi_norm/sqrt_phi_norm come back NaN,
            but their Jacobians fall back to `_MISSING_Q_JACOBIAN_FALLBACK`
            (see its module-level docstring).

    Returns:
        (values, jacobians), each field shaped like `psi_norm`. NaN
        coordinate inputs (e.g. an invalid TS channel) stay NaN throughout;
        only well-defined edge cases (the axis singularity, the SOL beyond
        the flux map's domain, a missing q profile) get a documented
        fallback instead of NaN/inf.
    """
    psi_norm = np.asarray(psi_norm, dtype=float)
    psi_floor = np.maximum(psi_norm, _JACOBIAN_EPS)

    sqrt_psi_norm = np.sqrt(np.maximum(psi_norm, 0.0))
    jac_sqrt_psi_norm = 0.5 / np.sqrt(psi_floor)

    if midplane_map is not None:
        span = midplane_map.r_lcfs_outboard - midplane_map.r_axis
        rho_outboard = (midplane_map.r_outboard - midplane_map.r_axis) / span
        rho = np.interp(psi_norm, midplane_map.psi_norm_outboard, rho_outboard)
        # d(rho)/d(psi_norm) along the midplane map, held constant beyond its
        # domain at both ends via the edge-clamped lookup below -- a
        # reasonable default for the fit's SOL extension (rho out to ~1.1)
        # rather than extrapolating a raw finite-difference slope.
        with np.errstate(invalid="ignore", divide="ignore"):
            d_rho_d_psi_grid = np.gradient(rho_outboard, midplane_map.psi_norm_outboard)
        jac_rho = np.interp(psi_norm, midplane_map.psi_norm_outboard, d_rho_d_psi_grid)
    else:
        rho = np.full_like(psi_norm, np.nan)
        jac_rho = np.full_like(psi_norm, np.nan)

    qpsi_arr = None if qpsi is None else np.asarray(qpsi, dtype=float)
    if qpsi_arr is not None and qpsi_arr.size > 0 and np.all(np.isfinite(qpsi_arr)):
        psi_norm_grid = np.linspace(0.0, 1.0, qpsi_arr.size)
        phi_unnorm = _toroidal_flux_unnorm(qpsi_arr)
        total_q = phi_unnorm[-1]
        phi_norm = np.interp(psi_norm, psi_norm_grid, phi_unnorm / total_q)
        q_at = np.interp(psi_norm, psi_norm_grid, qpsi_arr)
        jac_phi_norm = q_at / total_q
    else:
        phi_norm = np.full_like(psi_norm, np.nan)
        jac_phi_norm = np.full_like(psi_norm, _MISSING_Q_JACOBIAN_FALLBACK)

    phi_floor = np.maximum(phi_norm, _JACOBIAN_EPS)
    sqrt_phi_norm = np.sqrt(np.maximum(phi_norm, 0.0))
    with np.errstate(invalid="ignore"):
        jac_sqrt_phi_norm = np.where(
            np.isfinite(phi_norm),
            0.5 * jac_phi_norm / np.sqrt(phi_floor),
            _MISSING_Q_JACOBIAN_FALLBACK,
        )

    values = CoordinateValues(
        psi_norm=psi_norm,
        sqrt_psi_norm=sqrt_psi_norm,
        phi_norm=phi_norm,
        sqrt_phi_norm=sqrt_phi_norm,
        rho=rho,
    )
    jacobians = CoordinateJacobians(
        psi_norm=np.ones_like(psi_norm),
        sqrt_psi_norm=jac_sqrt_psi_norm,
        phi_norm=jac_phi_norm,
        sqrt_phi_norm=jac_sqrt_phi_norm,
        rho=jac_rho,
    )
    return values, jacobians


def transform_gradient(
    grad_wrt_from: np.ndarray,
    jac_from: np.ndarray,
    jac_to: np.ndarray,
    *,
    jac_to_floor: float = _JACOBIAN_EPS,
) -> np.ndarray:
    """Re-express a derivative from one coordinate into another via the chain rule.

    Both Jacobians must be d(coordinate)/d(psi_norm) at the same points (see
    `CoordinateJacobians`/`coordinates_from_psi_norm`) -- psi_norm is the
    common pivot every pairwise conversion goes through:

        d(quantity)/d(to) = d(quantity)/d(from) * d(from)/d(to)
                           = grad_wrt_from * (jac_from / jac_to)

    Args:
        grad_wrt_from: (n,) derivative of some quantity w.r.t. the `from` coordinate
            (e.g. a GP fit's d(te)/d(rho)).
        jac_from: (n,) d(from)/d(psi_norm) at the same points.
        jac_to: (n,) d(to)/d(psi_norm) at the same points.
        jac_to_floor: Minimum |jac_to| before it is treated as locally flat
            and floored (sign-preserved) instead of divided by directly.

    Returns:
        (n,) derivative of the same quantity w.r.t. the `to` coordinate. NaN
        only where an input was already NaN (a genuinely missing point --
        see module docstring); a merely-tiny jac_to is floored, not
        propagated as NaN/inf.
    """
    grad_wrt_from = np.asarray(grad_wrt_from, dtype=float)
    jac_from = np.asarray(jac_from, dtype=float)
    jac_to = np.asarray(jac_to, dtype=float)
    sign = np.where(jac_to < 0, -1.0, 1.0)
    jac_to_safe = np.where(np.abs(jac_to) < jac_to_floor, sign * jac_to_floor, jac_to)
    return grad_wrt_from * jac_from / jac_to_safe
