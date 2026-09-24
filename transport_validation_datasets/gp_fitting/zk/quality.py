"""Nonphysical-fit checks for the zk method.

Te and ne fall from the core, so a fit that peaks off-axis is suspect.

fit_ignores_data flags a fit that sits below its innermost channels.

The worker retries a flagged fit and culls it once the repairs run out (worker_zk._fit_variable).
"""

import numpy as np

_EDGE_RHO = 0.9
_EDGE_MARGIN = 1.1
_ENVELOPE_MARGIN = 1.2
# Half-width of the envelope window, and of the channels the worker drops around a peak
REPAIR_HALFWIDTH = 0.1

# A fit whose innermost _FIT_BIAS_CORE_N channels sit on average more than _FIT_BIAS_CORE_SIGMA above it
# is a core amplitude collapse,
# where the likelihood prefers a small variance that hugs the prior below a sparse, noisy core.
# Healthy fits stay below ~1.7 and collapses sit at 2.6 and above.
# One-sided, since a fit above a low bad channel is doing its job,
# and an overshoot is the envelope check's to judge.
# Core-only, since the same test along the whole profile culls healthy fits.
# No channel subset repairs a core the fit refuses to reach,
# so a flagged fit gets the retries but never the channel drop.
_FIT_BIAS_CORE_N = 4
_FIT_BIAS_CORE_SIGMA = 2.5

# Monotonic-edge constraint, applied by gp.fit_profile.
# The GP can ring up into a bump past rho ~1.0,
# between the outermost channel and the first value anchor (1.3 by default),
# where the short edge length scale wiggles freely.
# Such a bump rarely beats the whole interior by _EDGE_MARGIN, so nonphysical_peak misses it.
# Wherever the fit's gradient on MONO_CHECK_RHO exceeds MONO_GRAD_TOL,
# fit_profile adds a zero-gradient observation with error MONO_GRAD_ERR and refits at the same hyperparameters.
# It repeats up to MONO_MAX_PASSES times, since a refit can push the bump sideways.
# The constraint is soft, so a sharp bump flattens toward a plateau rather than to zero slope.
# A rise the channels support is left alone (rise_is_data_supported),
# since hollow ne genuinely rises through rho 0.6-0.9.
# Tolerance and error are in scale_per_slice-normalized units.
MONO_CHECK_RHO = np.concatenate([np.linspace(0.6, 0.85, 6), np.linspace(0.9, 1.09, 20)])
MONO_GRAD_TOL = 0.01
MONO_GRAD_ERR = 0.05
MONO_MAX_PASSES = 3
# Data-support gate, see rise_is_data_supported.
# Five channels separates the flank of a hollow profile, which spans many (~20 per window on MAST),
# from the 2-3 channel pedestal-shoulder bump the constraint exists for.
_MONO_SUPPORT_HALFWIDTH = 0.08
_MONO_SUPPORT_MIN_POINTS = 5
_MONO_SUPPORT_TSTAT = 1.0


def rise_is_data_supported(data_x, data_y, err_y, rho) -> bool:
    """Check whether the channels around rho show a significant rise.

    Fits a weighted least-squares slope to the channels within _MONO_SUPPORT_HALFWIDTH of rho.
    The rise is data-supported when the slope is positive with a t-statistic above _MONO_SUPPORT_TSTAT.
    With fewer than _MONO_SUPPORT_MIN_POINTS channels the fit is extrapolating or bridging a gap,
    so nothing supports the rise.

    Args:
        data_x: Channel rho positions.
        data_y: Channel values.
        err_y: Channel errors.
        rho: Where the fit rises.

    Returns:
        True if the channels support a positive slope at rho.
    """
    m = (
        np.isfinite(data_x)
        & np.isfinite(data_y)
        & np.isfinite(err_y)
        & (np.abs(data_x - rho) <= _MONO_SUPPORT_HALFWIDTH)
    )
    if int(m.sum()) < _MONO_SUPPORT_MIN_POINTS:
        return False
    x, y, e = data_x[m], data_y[m], err_y[m]
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    sxx = float(np.sum(w * (x - xm) ** 2))
    if sxx <= 0:
        return False  # all channels at one rho, no slope information
    slope = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
    slope_err = float(np.sqrt(1.0 / sxx))
    return slope > 0 and slope / max(slope_err, 1e-12) > _MONO_SUPPORT_TSTAT


def fit_ignores_data(data_x, data_y, err_y, x_star, y_fit) -> bool:
    """Check whether the fit sits below its innermost channels.

    Compares the mean of z = (y - fit) / err over the innermost
    _FIT_BIAS_CORE_N channels with _FIT_BIAS_CORE_SIGMA.
    See the block comment above _FIT_BIAS_CORE_N for why the test is one-sided and core-only.

    Args:
        data_x: Channel rho positions.
        data_y: Channel values.
        err_y: Channel errors.
        x_star: rho grid of the fit.
        y_fit: Fitted profile on x_star.

    Returns:
        True if the fit is a core amplitude collapse.
    """
    valid = np.isfinite(data_x) & np.isfinite(data_y) & np.isfinite(err_y)
    if not valid.any():
        return False
    order = np.argsort(data_x[valid])
    xs = np.asarray(data_x, dtype=float)[valid][order]
    ys = np.asarray(data_y, dtype=float)[valid][order]
    es = np.asarray(err_y, dtype=float)[valid][order]
    z = (ys - np.interp(xs, np.asarray(x_star, dtype=float), y_fit)) / es
    n_core = min(_FIT_BIAS_CORE_N, z.size)
    return float(np.mean(z[:n_core])) > _FIT_BIAS_CORE_SIGMA


def data_envelope(data_x, data_y, data_err, rho0) -> float:
    """Compute the upper envelope of the channel scatter near rho0.

    The envelope is max(y + 2 err) over the channels within REPAIR_HALFWIDTH of rho0
    and the nearest channel on each side.
    The nearest channels keep a fit bridging a data gap from reading as overshoot,
    since a fit descending a steep pedestal sits below its inner neighbor
    even when that neighbor is outside the window.

    Args:
        data_x: Channel rho positions.
        data_y: Channel values.
        data_err: Channel errors.
        rho0: Location to evaluate the envelope at.

    Returns:
        The envelope value, or NaN if there is no finite data at all.
    """
    mask = np.isfinite(data_x) & np.isfinite(data_y) & np.isfinite(data_err)
    if not mask.any():
        return np.nan
    x_valid = data_x[mask]
    top_valid = data_y[mask] + 2.0 * data_err[mask]
    keep = np.abs(x_valid - rho0) <= REPAIR_HALFWIDTH
    inner = x_valid < rho0
    outer = x_valid > rho0
    if inner.any():
        keep[inner & (x_valid == x_valid[inner].max())] = True
    if outer.any():
        keep[outer & (x_valid == x_valid[outer].min())] = True
    return float(np.max(top_valid[keep])) if keep.any() else np.nan


def nonphysical_peak(y, x_star, data_x, data_y, data_err) -> float | None:
    """Locate the nonphysical off-axis peak of a fitted slice, if any.

    Edge bump:
    The fit at rho >= _EDGE_RHO beats the whole interior by more than _EDGE_MARGIN.
    The margin spares flat profiles whose maximum lands in the edge by noise.
    It fires even when channels support the bump,
    since an edge above the whole interior means miscalibrated edge channels.

    Data overshoot:
    The fit beats the local data envelope (data_envelope) by more than _ENVELOPE_MARGIN,
    which is the GP ringing above its own data.
    It is checked from the innermost to the outermost channel, where the fit interpolates.
    Inside the innermost channel a peaked profile may rightly keep rising toward the axis.
    Past the outermost channel fit and envelope are both near zero,
    so their ratio would flag harmless SOL ringing that the anchors and the monotonic-edge constraint govern.
    Off-axis humps the data supports, such as hollow ne, pass.

    Args:
        y: Fitted profile on x_star.
        x_star: rho grid of the fit.
        data_x: Channel rho positions.
        data_y: Channel values.
        data_err: Channel errors.

    Returns:
        rho of the peak, or None if the slice is healthy.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x_star, dtype=float).ravel()
    if not np.isfinite(y).any():
        return None
    edge, interior = x >= _EDGE_RHO, x < _EDGE_RHO
    with np.errstate(invalid="ignore"):
        if (
            np.isfinite(y[edge]).any()
            and np.isfinite(y[interior]).any()
            and np.nanmax(y[edge]) > _EDGE_MARGIN * np.nanmax(y[interior])
        ):
            return float(x[edge][np.nanargmax(y[edge])])
        ch_finite = np.isfinite(data_x) & np.isfinite(data_y)
        innermost = float(np.min(data_x[ch_finite])) if ch_finite.any() else np.inf
        outermost = float(np.max(data_x[ch_finite])) if ch_finite.any() else -np.inf
        worst_rho, worst = None, _ENVELOPE_MARGIN
        in_span = (x >= innermost) & (x <= outermost)
        for i in np.flatnonzero(in_span & np.isfinite(y) & (y > 0)):
            env = data_envelope(data_x, data_y, data_err, x[i])
            if np.isfinite(env) and env > 0 and y[i] / env > worst:
                worst_rho, worst = float(x[i]), y[i] / env
    return worst_rho
