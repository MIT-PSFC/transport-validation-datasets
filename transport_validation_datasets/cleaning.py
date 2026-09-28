"""Channel screens shared by every fit method, run when the fit batches are staged.

clean_fit_rows judges each Thomson sample on its own, before time windows restrict or pool the rows.
Judged in a pooled window, a channel that misfires in several samples
finds its own other readings as neighbours and survives.
The screens compare a channel only with its radial neighbours in the same sample and are scale-free,
so they run on the staged units, before any worker normalizes.

1. flag error-bar outliers (isolated huge-error channels), then spikes (neighbour-disagreement test),
   in Te and in ne, each judged on its own readings
2. drop both the Te and the ne reading of every channel flagged in either,
   since Thomson scattering measures the two together from one spectrum

drop_rows_without_core then judges each row as it is staged, after the windows,
since it asks what one fit will see: a variable with no channel in the core is not fit at all.
"""

from dataclasses import replace

import numpy as np
from loguru import logger

from transport_validation_datasets.gp_fitting.batch_io import ShotFitInput

# An isolated channel whose error bar is many times its neighbours' poisons
# a heteroscedastic noise model: akho's error kernel smooths error bars in rho_tor_norm,
# so one huge-error channel inflates the effective noise of every channel near it
# and the fit goes slack across that region.
# The comparison is on ABSOLUTE errors: a relative-to-value rule flags low-value SOL
# points (large relative error is normal there), while the absolute ratio
# leaves them alone and MAST's uniform fractional errors almost never trip it
_ERR_OUTLIER_FACTOR = 5.0
_ERR_OUTLIER_HALFWIDTH = 0.1
_ERR_OUTLIER_MIN_NEIGHBORS = 3

# Combined-sigma windows of the spike screen, see _isolated_spikes.
_SPIKE_SIGMA_NEIGHBOR = 2.0
_SPIKE_SIGMA_LOCAL = 3.0

# A row whose innermost valid channel of a variable sits outside this has nothing to pin the core,
# and the fit there is an extrapolation (akho reached ne 25e20 m^-3 on C-Mod 1030516024).
# Fewer, well-constrained profiles are worth more than many extrapolated ones.
# On a 92-shot C-Mod sample 0.15 skips ~1% of the rows.
# On a 97-shot MAST sample it skips ~half, since the chord passes above the axis.
CORE_COVERAGE_RHO_TOR_NORM = 0.15

# A reading under DIP_RATIO times both its rho neighbours is a dead or misfired channel
# The spike screen above misses these, because it needs the neighbours to agree with each other,
# which a pedestal never does, and a near-zero C-Mod Te reading on its 15 eV error floor
# then outweighs its neighbours 10-50x and can pull the fitted pedestal to zero.
DIP_RATIO = 0.35
DIP_RHO_TOR_NORM_MAX = 0.97


def clean_fit_rows(fit_input: ShotFitInput, shot: int) -> ShotFitInput:
    """Run the shared screens on every Thomson sample of one shot, for Te and ne.

    Thomson scattering measures Te and ne together, from the spectrum of one laser pulse in one channel,
    so a channel flagged in either variable loses both readings in that sample.

    Args:
        fit_input: The shot's per-sample fit input, as prepare_fit_input built it.
        shot: Shot number, for the log line.

    Returns:
        The fit input with the dropped readings NaN in te_y and ne_y.
    """
    te_error_outliers, te_spikes = _screen_rows(
        fit_input.x, fit_input.te_y, fit_input.te_err
    )
    ne_error_outliers, ne_spikes = _screen_rows(
        fit_input.x, fit_input.ne_y, fit_input.ne_err
    )
    te_flagged = te_error_outliers | te_spikes
    ne_flagged = ne_error_outliers | ne_spikes
    te_y, ne_y, n_te_taken, n_ne_taken = drop_in_both(
        fit_input.te_y, fit_input.ne_y, te_flagged, ne_flagged
    )
    if te_flagged.any() or ne_flagged.any():
        logger.info(
            f"Shot {shot}: cleaning dropped "
            f"te {int(te_error_outliers.sum())} error outliers and {int(te_spikes.sum())} spikes, "
            f"ne {int(ne_error_outliers.sum())} error outliers and {int(ne_spikes.sum())} spikes, "
            f"and with them {n_te_taken} te and {n_ne_taken} ne readings of the same channels"
        )
    return replace(fit_input, te_y=te_y, ne_y=ne_y)


def drop_in_both(
    te_y: np.ndarray, ne_y: np.ndarray, te_dropped: np.ndarray, ne_dropped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Drop both readings of every channel a screen dropped in either Te or ne.

    Thomson scattering measures Te and ne together, from the spectrum of one laser pulse in one channel,
    so a reading a screen rejects in one variable condemns the other in the same sample.

    Args:
        te_y: (n_t, n_ch) Te readings.
        ne_y: (n_t, n_ch) ne readings.
        te_dropped: (n_t, n_ch) mask of the Te readings a screen dropped.
        ne_dropped: (n_t, n_ch) mask of the ne readings a screen dropped.

    Returns:
        (te_y, ne_y, n_te_taken, n_ne_taken): the readings with every dropped channel NaN in both,
        and how many readings of each variable the other variable's drops took along.
    """
    channel_dropped = te_dropped | ne_dropped
    te_taken = ne_dropped & ~te_dropped & np.isfinite(te_y)
    ne_taken = te_dropped & ~ne_dropped & np.isfinite(ne_y)
    te_y_out = np.where(channel_dropped, np.nan, te_y)
    ne_y_out = np.where(channel_dropped, np.nan, ne_y)
    return te_y_out, ne_y_out, int(te_taken.sum()), int(ne_taken.sum())


def _screen_rows(
    x_rows: np.ndarray, y_rows: np.ndarray, err_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run the error-outlier and spike screens on every row of one variable.

    Args:
        x_rows: (n_t, n_ch) channel rho_tor_norm positions.
        y_rows: (n_t, n_ch) channel values, NaN where invalid.
        err_rows: (n_t, n_ch) channel errors.

    Returns:
        (error_outliers, spikes): the (n_t, n_ch) masks of the points each screen flagged.
    """
    error_outliers = np.zeros(y_rows.shape, dtype=bool)
    spikes = np.zeros(y_rows.shape, dtype=bool)
    for row in range(y_rows.shape[0]):
        x = x_rows[row]
        y = y_rows[row]
        err = err_rows[row]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
        idx = np.flatnonzero(valid)
        row_error_outliers = _error_outliers(x[idx], err[idx])
        error_outliers[row, idx[row_error_outliers]] = True
        # The spike screen judges the survivors of the error screen
        idx = idx[~row_error_outliers]
        row_spikes = _isolated_spikes(x[idx], y[idx], err[idx])
        spikes[row, idx[row_spikes]] = True
    return error_outliers, spikes


def _isolated_spikes(x: np.ndarray, y: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Find points that disagree with both immediate radial neighbours.

    A point is flagged only when its neighbours agree with each other,
    independent of any GP fit or hyperparameters.
    A GP reference fit can be flexible enough to bend down and absorb a single bad point
    along with its genuinely consistent neighbours,
    which is exactly what let a near-zero misfired channel escape zk's leave-one-out removal on a C-Mod Te slice.
    Comparing a point only to its immediate neighbours in rho_tor_norm
    catches an isolated single-channel spike however flexible the eventual fit is.

    A point (not the first or last, by rho_tor_norm) is flagged when its neighbours
    agree with each other (within _SPIKE_SIGMA_NEIGHBOR combined sigma)
    but it disagrees with their average (by more than _SPIKE_SIGMA_LOCAL combined sigma).
    A genuine trend, where the neighbours themselves disagree, never trips this.

    Args:
        x: (n,) channel rho_tor_norm positions, NaN-free.
        y: (n,) channel values.
        err: (n,) channel errors.

    Returns:
        (n,) mask of the spikes, in the input order.
    """
    spike = np.zeros(x.size, dtype=bool)
    if x.size < 3:
        return spike
    order = np.argsort(x)
    ys, es = y[order], err[order]

    y_left, y_right = ys[:-2], ys[2:]
    e_left, e_right = es[:-2], es[2:]
    y_mid, e_mid = ys[1:-1], es[1:-1]

    neighbor_sigma = np.sqrt(e_left**2 + e_right**2)
    neighbors_agree = np.abs(y_left - y_right) <= _SPIKE_SIGMA_NEIGHBOR * neighbor_sigma
    neighbor_mean = 0.5 * (y_left + y_right)
    neighbor_mean_err = 0.5 * neighbor_sigma
    local_sigma = np.sqrt(e_mid**2 + neighbor_mean_err**2)
    point_disagrees = np.abs(y_mid - neighbor_mean) > _SPIKE_SIGMA_LOCAL * local_sigma

    spike[order[1:-1]] = neighbors_agree & point_disagrees
    return spike


def _error_outliers(x: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Find points whose error bar dwarfs the local error level.

    A point is flagged when its error exceeds _ERR_OUTLIER_FACTOR times the
    median error of its neighbours (within _ERR_OUTLIER_HALFWIDTH in rho_tor_norm),
    and only when at least _ERR_OUTLIER_MIN_NEIGHBORS are there to define a local error level.
    See the constants' block comment for the calibration.

    Args:
        x: (n,) channel rho_tor_norm positions, NaN-free.
        err: (n,) channel errors.

    Returns:
        (n,) mask of the error outliers.
    """
    outlier = np.zeros(x.size, dtype=bool)
    if x.size < _ERR_OUTLIER_MIN_NEIGHBORS + 1:
        return outlier
    for i in range(x.size):
        near = np.abs(x - x[i]) <= _ERR_OUTLIER_HALFWIDTH
        near[i] = False
        if int(near.sum()) < _ERR_OUTLIER_MIN_NEIGHBORS:
            continue
        local_err = np.median(err[near])
        outlier[i] = err[i] > _ERR_OUTLIER_FACTOR * local_err
    return outlier


def relative_dips(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    ratio: float = DIP_RATIO,
    rho_max: float = DIP_RHO_TOR_NORM_MAX,
) -> np.ndarray:
    """Find readings under ratio times both immediate rho neighbours, inside rho_max.

    The innermost reading is judged against the next two and the outermost is never judged.
    Scale-free and GP-free, so it runs on any units.

    Args:
        x_rows: (n_t, n_ch) channel rho_tor_norm positions.
        y_rows: (n_t, n_ch) channel values, NaN where invalid.
        ratio: A reading under this fraction of both neighbours is a dip.
        rho_max: Readings at or past this are not judged.

    Returns:
        (n_t, n_ch) mask of the dips.
    """
    dips = np.zeros(y_rows.shape, dtype=bool)
    for row in range(y_rows.shape[0]):
        valid = np.isfinite(x_rows[row]) & np.isfinite(y_rows[row])
        idx = np.flatnonzero(valid)
        if idx.size < 3:
            continue
        order = idx[np.argsort(x_rows[row][idx])]
        x = x_rows[row][order]
        y = y_rows[row][order]
        for k in range(order.size - 1):
            if x[k] >= rho_max:
                break
            neighbours = (1, 2) if k == 0 else (k - 1, k + 1)
            neighbour_min = min(y[j] for j in neighbours)
            dips[row, order[k]] = y[k] < ratio * neighbour_min
    return dips


def drop_rows_without_core(fit_input: ShotFitInput, shot: int) -> ShotFitInput:
    """Empty every row of a variable whose valid channels all sit outside CORE_COVERAGE_RHO_TOR_NORM.

    The worker then skips that variable of that row for too few points.

    Args:
        fit_input: The shot's fit input as it is staged, after the windows.
        shot: Shot number, for the log line.

    Returns:
        The fit input with those rows NaN in te_y or ne_y.
    """
    emptied = {}
    counts = {}
    for var in ("te", "ne"):
        y_rows = np.array(getattr(fit_input, f"{var}_y"), dtype=float)
        err_rows = getattr(fit_input, f"{var}_err")
        valid = np.isfinite(fit_input.x) & np.isfinite(y_rows) & np.isfinite(err_rows)
        x_valid = np.where(valid, fit_input.x, np.inf)
        innermost = x_valid.min(axis=1)
        # A row with no valid channel at all is left for the worker's own skip
        uncovered = valid.any(axis=1) & (innermost > CORE_COVERAGE_RHO_TOR_NORM)
        y_rows[uncovered] = np.nan
        emptied[f"{var}_y"] = y_rows
        counts[var] = int(uncovered.sum())
    if any(counts.values()):
        logger.info(
            f"Shot {shot}: no channel inside rho_tor_norm {CORE_COVERAGE_RHO_TOR_NORM} "
            f"in {counts['te']} te and {counts['ne']} ne rows, not fitting them"
        )
    return replace(fit_input, **emptied)
