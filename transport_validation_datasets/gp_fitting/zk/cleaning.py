"""Channel cleaning for the zk method: one entry point, fixed order.

clean_channels runs every screen in sequence:
1. drop NaN points
2. drop error-bar outliers (isolated huge-error channels)
3. drop isolated value spikes (neighbor-disagreement test)
4. compute the per-slice scale from the cleaned survivors and normalize
5. rough reference fit on the normalized data
6. leave-one-out outlier removal judged against that reference

Each screen keeps its own threshold and logic - each encodes a documented,
audited failure class - but they run exactly once, in one place. The scale is
computed after the spike filter so a single misfired high-value channel
cannot set the scale itself, squashing the rest of the real profile toward
~0 before it ever reaches outlier removal (and making that channel look like
the profile's own peak instead of the spike it is).
"""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import FitBounds
from transport_validation_datasets.gp_fitting.zk.gp import VALUE_BC, run_gp
from transport_validation_datasets.gp_fitting.zk.kernel import build_kernel

# An isolated channel whose error bar is many times its rho-neighbors' poisons
# the heteroscedastic noise model: the HSGP error kernel smooths error bars in
# rho, so one huge-error channel inflates the effective noise of every channel
# near it and the fit goes slack across that region.
# The comparison is on ABSOLUTE errors: a relative-to-value rule flags low-value SOL
# points (large relative error is normal there), while the absolute ratio
# leaves them alone and MAST's uniform fractional errors almost never trip it
_ERR_OUTLIER_FACTOR = 5.0
_ERR_OUTLIER_HALFWIDTH = 0.1
_ERR_OUTLIER_MIN_NEIGHBORS = 3

# Optimizer restarts for the rough reference fit. Its result is only an
# outlier-judging reference (discarded afterwards), so it runs a reduced
# restart budget and no pinned-hyperparameter retries.
_ROUGH_NRESTARTS = 3


def clean_channels(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    fit_bounds: FitBounds,
    scale_per_slice: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Clean one slice's channels and normalize them for fitting.

    Runs the screens in the fixed order described in the module docstring.

    Args:
        x: Channel rho positions.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        fit_bounds: The variable's staged bound knobs (for the rough fit).
        scale_per_slice: Normalize by the cleaned slice max before fitting, to
            prevent amplitude collapse when channels do not cover the full
            radial range.

    Returns:
        (x, y, err, scale) with y and err divided by scale, or None when no
        valid data remains or the scale is degenerate.
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if not valid.any():
        return None
    x, y, err = x[valid], y[valid], err[valid]

    x, y, err = _remove_error_outliers(x, y, err)
    x, y, err = _remove_local_outliers(x, y, err)

    scale = 1.0
    if scale_per_slice:
        with np.errstate(all="ignore"):
            scale = float(np.nanmax(y)) if y.size else np.nan
        if not np.isfinite(scale) or scale < 1e-6:
            return None
    y = np.asarray(y, dtype=float) / scale
    err = np.asarray(err, dtype=float) / scale
    x = np.asarray(x, dtype=float)

    rough_hyps = _rough_hyperparameters(x, y, err, fit_bounds)
    x, y, err = _remove_loo_outliers(x, y, err, ref_hyperparams=rough_hyps)
    return x, y, err, scale


def _remove_local_outliers(x, y, err, sigma_neighbor=2.0, sigma_local=3.0):
    """Drop points that disagree with both immediate rho-neighbors.

    A point is dropped only when its neighbors agree with each other:
    independent of any GP fit or hyperparameters, unlike the LOO removal. A GP
    reference fit (however it is built - generic or self-tuned) can be
    flexible enough to bend down and absorb a single bad point along with its
    genuinely-consistent neighbors, which is exactly what let a near-zero
    misfired channel escape the LOO removal on a C-Mod Te slice (a short core
    length scale dove down to chase it instead of the reference flagging it).
    Comparing a point only to its immediate left/right neighbors in rho
    catches an isolated single-channel spike regardless of how flexible the
    eventual fit is allowed to be.

    A point (not the first or last, by rho) is dropped when its neighbors
    agree with each other (within sigma_neighbor combined sigma) but it
    disagrees with their average (by more than sigma_local combined sigma). A
    genuine trend - where the neighbors themselves disagree - never trips
    this, since the neighbor-agreement precondition fails first.

    Args:
        x: Channel rho positions (NaN-free).
        y: Channel values.
        err: Channel errors.
        sigma_neighbor: Combined-sigma window for neighbor agreement.
        sigma_local: Combined-sigma threshold for the point's disagreement.

    Returns:
        (x, y, err) with the spikes removed, sorted by rho when any were.
    """
    n = x.size
    if n < 3:
        return x, y, err
    order = np.argsort(x)
    xs, ys, es = x[order], y[order], err[order]

    y_left, y_right = ys[:-2], ys[2:]
    e_left, e_right = es[:-2], es[2:]
    y_mid, e_mid = ys[1:-1], es[1:-1]

    neighbors_agree = np.abs(y_left - y_right) <= sigma_neighbor * np.sqrt(
        e_left**2 + e_right**2
    )
    neighbor_mean = 0.5 * (y_left + y_right)
    neighbor_mean_err = 0.5 * np.sqrt(e_left**2 + e_right**2)
    point_disagrees = np.abs(y_mid - neighbor_mean) > sigma_local * np.sqrt(
        e_mid**2 + neighbor_mean_err**2
    )

    drop = np.zeros(n, dtype=bool)
    drop[1:-1] = neighbors_agree & point_disagrees
    if not drop.any():
        return x, y, err
    keep = ~drop
    return xs[keep], ys[keep], es[keep]


def _remove_error_outliers(x, y, err):
    """Drop points whose error bar dwarfs the local error level.

    A point is dropped when its error exceeds _ERR_OUTLIER_FACTOR times the
    median error of its rho-neighbors (within _ERR_OUTLIER_HALFWIDTH, and only
    when at least _ERR_OUTLIER_MIN_NEIGHBORS are there to define a local error
    level). See the constants' block comment for the calibration.

    Args:
        x: Channel rho positions (NaN-free).
        y: Channel values.
        err: Channel errors.

    Returns:
        (x, y, err) with the error outliers removed.
    """
    n = x.size
    if n < _ERR_OUTLIER_MIN_NEIGHBORS + 1:
        return x, y, err
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        m = np.abs(x - x[i]) <= _ERR_OUTLIER_HALFWIDTH
        m[i] = False
        if int(m.sum()) >= _ERR_OUTLIER_MIN_NEIGHBORS and err[i] > (
            _ERR_OUTLIER_FACTOR * np.median(err[m])
        ):
            bad[i] = True
    if not bad.any():
        return x, y, err
    keep = ~bad
    return x[keep], y[keep], err[keep]


def _rough_hyperparameters(x, y, err, fit_bounds: FitBounds) -> np.ndarray | None:
    """Run one reduced optimize pass on (possibly still contaminated) data.

    Used only to get a locally-representative reference for the LOO outlier
    removal; never returned to callers as a real fit result.

    Args:
        x: Channel rho positions.
        y: Channel values (normalized).
        err: Channel errors (normalized).
        fit_bounds: The variable's staged bound knobs.

    Returns:
        Fitted [var, l1, l2, lw, x0], or None if the fit failed.
    """
    gp = run_gp(x, y, err, x, fit_bounds, nrestarts=_ROUGH_NRESTARTS, hyp_retries=0)
    if gp is None:
        return None
    return np.asarray(gp.get_gp_kernel_details()[1], dtype=float)


def _loo_standardized_residuals(x, y, err, hyperparams) -> np.ndarray | None:
    """Compute the leave-one-out standardized residual at every data point.

    Uses the closed-form GP LOO identities (Rasmussen and Williams 5.12): for
    C = (K + diag(err^2))^-1 and alpha = C @ y, the LOO residual at point i is
    alpha_i / C_ii and its predictive std is 1/sqrt(C_ii), so the standardized
    residual is simply alpha_i / sqrt(C_ii). Every point is thus predicted
    from all the others (never from itself) in a single matrix solve, with no
    per-point or reference GP fit.

    The edge value BCs (VALUE_BC) are appended as fixed training points so
    edge channels are judged against the same "pull to zero past the
    separatrix" constraint the real fit sees; residuals are returned for the
    real data points only.

    Args:
        x: Channel rho positions.
        y: Channel values.
        err: Channel errors.
        hyperparams: Kernel hyperparameters (length scales etc.); None uses
            the generic start values.

    Returns:
        (n,) standardized residuals, or None if the covariance solve fails or
        is not positive definite.
    """
    xx = np.concatenate([x, VALUE_BC[:, 0]])
    yy = np.concatenate([y, VALUE_BC[:, 1]])
    ee = np.concatenate([err, VALUE_BC[:, 2]])

    kernel = build_kernel(hyperparams)
    try:
        K = np.asarray(kernel(xx, xx, der=0), dtype=float)
        K[np.diag_indices_from(K)] += ee**2
        C = np.linalg.inv(K)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    c_diag = np.diag(C)
    if np.any(c_diag <= 0) or not np.all(np.isfinite(c_diag)):
        return None
    z = (C @ yy) / np.sqrt(c_diag)
    return z[: x.size]


def _locally_corroborated(x, y, err, i, sigma_corr) -> bool:
    """Check whether point i agrees with at least one immediate rho-neighbor.

    This is what separates a genuinely high, steep core - a run of points that
    each agree with the next - from an isolated bad channel. The global LOO
    reference (_loo_standardized_residuals) predicts every point from a single
    smooth kernel, so a legitimately steep core reads as a cluster of large
    residuals and gets culled along with the real spikes. A point whose
    neighbor sits at the same value is corroborated by real data at that rho,
    so it is protected from the LOO drop; an isolated spike (high or low),
    disagreeing with both neighbors, is not. Uses immediate sorted neighbors
    like _remove_local_outliers, but here one agreeing neighbor is enough
    (that check needs both neighbors to agree with each other, which a steep
    core fails).

    Args:
        x: Channel rho positions.
        y: Channel values.
        err: Channel errors.
        i: Index of the point to check.
        sigma_corr: Combined-sigma window for neighbor agreement.

    Returns:
        True if an immediate neighbor corroborates the point.
    """
    order = np.argsort(x)
    pos = int(np.flatnonzero(order == i)[0])
    for nb in (pos - 1, pos + 1):
        if 0 <= nb < x.size:
            j = int(order[nb])
            if np.abs(y[i] - y[j]) <= sigma_corr * np.sqrt(err[i] ** 2 + err[j] ** 2):
                return True
    return False


def _remove_loo_outliers(
    x, y, err, sigma=3.0, sigma_corr=2.0, max_drop_frac=0.3, ref_hyperparams=None
):
    """Drop uncorroborated LOO outliers, worst first, one at a time.

    Judging each point by its LOO residual (_loo_standardized_residuals) is
    what makes this robust: a bad channel cannot pull the reference toward
    itself to hide. But one bad point also inflates its neighbors' residuals,
    so flagging everything over sigma in one pass over-drops. Dropping only
    the single worst point and recomputing lets the swamped neighbors fall
    back below sigma.

    The LOO reference is a single smooth kernel, so a genuinely steep, high
    core reads as a run of large residuals and the plain rule culls the whole
    core. The corroboration gate (_locally_corroborated) fixes that: the worst
    over-sigma point is dropped only if it also disagrees with both immediate
    rho-neighbors, so consistently-high core points protect each other while
    an isolated spike is still removed. When the worst point is corroborated
    the next-worst uncorroborated one is taken instead; if every remaining
    over-sigma point is corroborated, stop.

    Args:
        x: Channel rho positions.
        y: Channel values.
        err: Channel errors.
        sigma: LOO residual threshold; 3.0 (rather than a stricter 2.0)
            tolerates reference/data mismatch on genuinely steep slices.
        sigma_corr: Combined-sigma window for neighbor corroboration.
        max_drop_frac: Stop after dropping this fraction of the points (and
            never below 3): a slice needing more than that is either genuinely
            bad or a real pedestal the fixed length scale cannot follow, and
            gutting it further only produces worse fits.
        ref_hyperparams: LOO kernel hyperparameters (the slice's own
            rough-optimized shape); falls back to the generic start values if
            None. Held fixed across iterations - refitting each pass would be
            the old cost back.

    Returns:
        (x, y, err) with the outliers removed.
    """
    n0 = x.size
    if n0 < 3:
        return x, y, err

    max_drop = max(1, round(max_drop_frac * n0))
    n_dropped = 0
    while x.size > 3 and n_dropped < max_drop:
        z = _loo_standardized_residuals(x, y, err, ref_hyperparams)
        if z is None:
            break
        to_drop = None
        for i in np.argsort(np.abs(z))[::-1]:
            if np.abs(z[i]) <= sigma:
                break  # remaining points are all below sigma
            if not _locally_corroborated(x, y, err, int(i), sigma_corr):
                to_drop = int(i)
                break
        if to_drop is None:
            break  # every over-sigma point is corroborated by a neighbor
        keep = np.ones(x.size, dtype=bool)
        keep[to_drop] = False
        x, y, err = x[keep], y[keep], err[keep]
        n_dropped += 1

    return x, y, err
