"""Channel cleaning for the zk method: one entry point, fixed order.

The GP-free screens (error-bar outliers, isolated spikes) already ran when the batch was staged,
see transport_validation_datasets.cleaning, so every method sees the same data.
clean_channels runs what needs a GP, in sequence:
1. drop NaN points
2. compute the per-slice scale from the staged survivors and normalize
3. rough reference fit on the normalized data
4. leave-one-out outlier removal judged against that reference

The scale comes from data the staging screens already cleaned,
so a single misfired high-value channel cannot set it,
squashing the rest of the real profile toward ~0 before it ever reaches outlier removal.
"""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import FitAnchors, FitBounds
from transport_validation_datasets.gp_fitting.zk.gp import run_gp
from transport_validation_datasets.gp_fitting.zk.kernel import build_kernel

# Optimizer restarts for the rough reference fit. Its result is only an
# outlier-judging reference (discarded afterwards), so it runs a reduced
# restart budget and no pinned-hyperparameter retries.
_ROUGH_NRESTARTS = 3


def clean_channels(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    fit_bounds: FitBounds,
    anchors: FitAnchors,
    pedestal_rho: float,
    scale_per_slice: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, FitAnchors] | None:
    """Clean one slice's channels and normalize them for fitting.

    Runs the screens in the fixed order described in the module docstring.

    Args:
        x: Channel rho positions.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        fit_bounds: The variable's staged bound knobs (for the rough fit).
        anchors: The variable's anchors, in the data's own units.
        pedestal_rho: The kernel's length-scale transition center, for the rough fit and the leave-one-out.
        scale_per_slice: Normalize by the cleaned slice max before fitting, to
            prevent amplitude collapse when channels do not cover the full
            radial range.

    Returns:
        (x, y, err, scale, anchors) with y, err, and the anchors divided by scale,
        or None when no valid data remains or the scale is degenerate.
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if not valid.any():
        return None
    x, y, err = x[valid], y[valid], err[valid]

    scale = 1.0
    if scale_per_slice:
        with np.errstate(all="ignore"):
            scale = float(np.nanmax(y)) if y.size else np.nan
        if not np.isfinite(scale) or scale < 1e-6:
            return None
    y = np.asarray(y, dtype=float) / scale
    err = np.asarray(err, dtype=float) / scale
    x = np.asarray(x, dtype=float)

    anchors = anchors.scaled(scale)
    rough_hyps = _rough_hyperparameters(x, y, err, fit_bounds, anchors, pedestal_rho)
    x, y, err = _remove_loo_outliers(
        x, y, err, anchors.value, pedestal_rho, ref_hyperparams=rough_hyps
    )
    return x, y, err, scale, anchors


def _rough_hyperparameters(
    x, y, err, fit_bounds: FitBounds, anchors: FitAnchors, pedestal_rho: float
) -> np.ndarray | None:
    """Run one reduced optimize pass on (possibly still contaminated) data.

    Used only to get a locally-representative reference for the LOO outlier
    removal; never returned to callers as a real fit result.

    Args:
        x: Channel rho positions.
        y: Channel values (normalized).
        err: Channel errors (normalized).
        fit_bounds: The variable's staged bound knobs.
        anchors: The variable's anchors (normalized).
        pedestal_rho: The kernel's length-scale transition center.

    Returns:
        Fitted [var, l1, l2, lw], or None if the fit failed.
    """
    gp = run_gp(
        x,
        y,
        err,
        x,
        fit_bounds,
        anchors,
        pedestal_rho,
        nrestarts=_ROUGH_NRESTARTS,
        hyp_retries=0,
    )
    if gp is None:
        return None
    return np.asarray(gp.get_gp_kernel().hyperparameters, dtype=float)


def _loo_standardized_residuals(
    x, y, err, value_anchors, pedestal_rho, hyperparams
) -> np.ndarray | None:
    """Compute the leave-one-out standardized residual at every data point.

    Uses the closed-form GP LOO identities (Rasmussen and Williams 5.12): for
    C = (K + diag(err^2))^-1 and alpha = C @ y, the LOO residual at point i is
    alpha_i / C_ii and its predictive std is 1/sqrt(C_ii), so the standardized
    residual is simply alpha_i / sqrt(C_ii). Every point is thus predicted
    from all the others (never from itself) in a single matrix solve, with no
    per-point or reference GP fit.

    The value anchors are appended as fixed training points so
    edge channels are judged against the same "pull to zero past the
    separatrix" constraint the real fit sees; residuals are returned for the
    real data points only.

    Args:
        x: Channel rho positions.
        y: Channel values.
        err: Channel errors.
        value_anchors: (n_a, 3) value anchor rows (normalized).
        pedestal_rho: The kernel's length-scale transition center.
        hyperparams: Kernel hyperparameters (length scales etc.); None uses
            the generic start values.

    Returns:
        (n,) standardized residuals, or None if the covariance solve fails or
        is not positive definite.
    """
    xx = np.concatenate([x, value_anchors[:, 0]])
    yy = np.concatenate([y, value_anchors[:, 1]])
    ee = np.concatenate([err, value_anchors[:, 2]])

    kernel = build_kernel(pedestal_rho, hyperparams)
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
    like the staging spike screen (cleaning._isolated_spikes), but here one agreeing neighbor is enough
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
    x,
    y,
    err,
    value_anchors,
    pedestal_rho,
    sigma=3.0,
    sigma_corr=2.0,
    max_drop_frac=0.3,
    ref_hyperparams=None,
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
        value_anchors: (n_a, 3) value anchor rows (normalized).
        pedestal_rho: The kernel's length-scale transition center.
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
        z = _loo_standardized_residuals(
            x, y, err, value_anchors, pedestal_rho, ref_hyperparams
        )
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
