"""Nonphysical-fit detection for the zk method (numpy only, no mkgp).

Te and ne fall monotonically from the core, so a fitted slice that peaks
off-axis is suspect. Two triggers:
  Edge bump: the fit at rho >= _EDGE_RHO exceeds everything interior to it by
    more than _EDGE_MARGIN. The margin keeps flat profiles whose global max
    lands in the edge region by noise (a zero-margin rule culled those). This
    fires whether or not the scatter supports the bump: a data-supported edge
    bump above the whole interior means miscalibrated edge channels (C-Mod
    edge-vs-core TS cross-calibration blocks), and the repair drops them.
  Data overshoot: the fit exceeds the local scatter envelope (data_envelope)
    by more than _ENVELOPE_MARGIN - the GP ringing above its own data (e.g. a
    14 keV spike off a 5 keV stray point). Only the true extrapolation region
    inside the innermost finite channel is exempt: a peaked profile
    legitimately rises above its innermost channel toward the axis, but a
    hump BETWEEN channels (rho bracketed by data on both sides) is
    interpolation and has no business beating the envelope - a blanket
    rho < 0.4 exemption once let a pinned te fit invent a 6.4 keV peak at
    rho 0.26 between 4 keV channels (C-Mod 1160503008 t=1.311).
    Data-supported off-axis humps (hollow ramp-up ne) pass: real physics, not
    an artifact.
A flagged slice is repaired by refitting without the channels under the peak
(see the worker's _fit_variable; a pinned te fit is first retried unpinned);
it is culled only if the repairs are exhausted.
"""

import numpy as np

_EDGE_RHO = 0.9
_EDGE_MARGIN = 1.1
_ENVELOPE_MARGIN = 1.2
REPAIR_HALFWIDTH = 0.1

# A fit whose innermost channels sit >= _FIT_BIAS_CORE_SIGMA ABOVE it is a
# core amplitude collapse: LML can prefer a tiny variance that hugs the prior
# below a sparse noisy core cluster (C-Mod 1160503008 t=0.911, var=0.12 with
# the 2.5-2.8 keV core cluster 2.6 sigma above the fit). The check is
# one-sided (fit-below-data only: a fit riding above a garbage-low channel
# subset - outlier-removed miscalibrated blocks the interferometer
# contradicts, e.g. the C-Mod 1160920xxx run day - is the fit doing its job,
# and the fit-above-data direction belongs to nonphysical_peak's envelope
# check) and CORE-ONLY: a general sliding-window version was tried and culled
# ~3% of healthy C-Mod ne slices whose dense tight-error runs sit 2 sigma off
# for benign reasons, versus 0.01-0.16% for this innermost-channel form
# (full-rebuild calibration: innermost-4 bias p99 is 1.1-1.7 per device/var,
# the collapse class sits at 2.6+). Pinned fits retry unpinned; otherwise the
# slice is culled - no channel subset repairs a core the fit refuses to reach.
_FIT_BIAS_CORE_N = 4
_FIT_BIAS_CORE_SIGMA = 2.5

# Monotonic-edge constraint (virtual zero-slope observations). Te and ne fall
# monotonically toward the edge, but the GP can ring up into a small bump
# around rho ~1.0, between the outermost channel and the value BCs at 1.1+
# where the short edge length scale wiggles freely (nonphysical_peak only
# catches bumps beating the whole interior by _EDGE_MARGIN, so a pedestal-top
# bump passes). fit_profile checks the posterior gradient on MONO_CHECK_RHO
# and, wherever it exceeds MONO_GRAD_TOL, adds a virtual gradient observation
# (rho, 0, MONO_GRAD_ERR) and refits at the same hyperparameters, up to
# MONO_MAX_PASSES times (a refit can push the bump sideways into an
# unconstrained neighbor). Observations are added only where violated AND
# where the channel data itself does not support a rise
# (rise_is_data_supported): hollow MAST ne genuinely rises through rho
# 0.6-0.9, and constraining a data-backed rise flattened both the valley and
# the off-axis peak of every hollow profile (icddps2 audit, 2026-07: e.g.
# shot 30097 t=0.245 s). With the gate, only rises the data does not
# corroborate (ringing between the outermost channel and the value BCs, or
# bumps inside data gaps) are suppressed. The constraint is soft
# (MONO_GRAD_ERR is the virtual observation's error bar), so a sharp bump
# flattens toward a plateau rather than to exactly zero slope. Tolerance and
# error are in scale_per_slice-normalized units, like every other constant
# here.
MONO_CHECK_RHO = np.concatenate([np.linspace(0.6, 0.85, 6), np.linspace(0.9, 1.09, 20)])
MONO_GRAD_TOL = 0.01
MONO_GRAD_ERR = 0.05
MONO_MAX_PASSES = 3
# Data-support gate for the mono constraint: window half-width around the
# violation point, minimum channels in the window, and the t-statistic the
# local weighted-least-squares slope must exceed for the rise to count as
# data-supported (and thus be left alone). MIN_POINTS = 5 encodes the breadth
# distinction: a genuine hollow-profile flank spans many channels (MAST has
# ~20 per window), while a narrow 2-3 channel bump on the pedestal shoulder -
# the artifact this constraint exists for - cannot muster 5, so sparse or
# narrow features keep the old always-constrain behavior.
_MONO_SUPPORT_HALFWIDTH = 0.08
_MONO_SUPPORT_MIN_POINTS = 5
_MONO_SUPPORT_TSTAT = 1.0


def rise_is_data_supported(data_x, data_y, err_y, rho) -> bool:
    """Check whether the channel data around rho shows a significant rise.

    Weighted least-squares slope over channels within _MONO_SUPPORT_HALFWIDTH
    of rho; the rise counts as data-supported when the slope is positive with
    t-statistic above _MONO_SUPPORT_TSTAT. Fewer than _MONO_SUPPORT_MIN_POINTS
    channels in the window means there is no data to support anything (the fit
    is extrapolating or interpolating a gap), so the mono constraint applies.

    Args:
        data_x: Channel rho positions.
        data_y: Channel values.
        err_y: Channel errors.
        rho: Location of the fit's rising gradient.

    Returns:
        True if the local data itself supports a positive slope at rho.
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
    """Check whether the fit sits systematically BELOW its innermost channels.

    One-sided test on the standardized residuals z = (y - fit) / err of the
    innermost _FIT_BIAS_CORE_N channels (in rho order) against
    _FIT_BIAS_CORE_SIGMA. See the block comment for why this is deliberately
    core-only and one-sided.

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

    max(y + 2 err) over channels within REPAIR_HALFWIDTH of rho0, plus the
    nearest finite channel on each side. Including the nearest neighbors keeps
    interpolation across a data gap from reading as overshoot: a fit
    descending a steep pedestal sits below its inner neighbor, which belongs
    in the envelope even when it falls outside the fixed window.

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

    See the trigger definitions in the module docstring.

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
        worst_rho, worst = None, _ENVELOPE_MARGIN
        for i in np.flatnonzero((x >= innermost) & np.isfinite(y) & (y > 0)):
            env = data_envelope(data_x, data_y, data_err, x[i])
            if np.isfinite(env) and env > 0 and y[i] / env > worst:
                worst_rho, worst = float(x[i]), y[i] / env
    return worst_rho
