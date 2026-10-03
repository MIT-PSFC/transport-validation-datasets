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

A chord that crosses the magnetic axis samples each flux surface twice, on two branches
(MAST inboard and outboard, TCV below and above the axis).
low_side_channels splits them and branch_disagreement_errors inflates the errors where they disagree,
which a device calls from its prepare_fit_input.
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

# The two branches of a chord can disagree:
# kinetic profiles in spherical tokamaks are not necessarily flux functions,
# and a reconstruction pins the flux surfaces away from the chord's crossing less precisely.
# Where they do, each channel's error gets half the local disagreement added,
# so both branches are consistent with a profile between them.
# The disagreement at a channel is its value minus the other branch interpolated to its rho,
# only between two channels of the other branch at most BRANCH_MAX_GAP apart.
# Each channel takes the median |disagreement| of the channels of both branches within BRANCH_SMOOTH_HALFWIDTH
# (at least BRANCH_MIN_CHANNELS of them), which keeps one spike from inflating its neighbours.
# Pooling both branches inflates channels at the same rho alike,
# where per branch the sparser one could keep its raw errors and steer the fit (MAST 24403 t=0.342).
# Channels outside the overlap keep their raw errors.
BRANCH_MAX_GAP = 0.08
BRANCH_SMOOTH_HALFWIDTH = 0.05
BRANCH_MIN_CHANNELS = 3

# A channel whose readings sit far under their rho neighbours for most of a shot is broken for the shot,
# a miscalibrated polychromator or a misaligned scattering volume, see persistently_low_channels.
# Each reading is compared with the median of the other channels within PERSISTENT_HALFWIDTH, both branches pooled,
# at rho_tor_norm under PERSISTENT_RHO_MAX, where the profile is smooth enough on that scale.
# Calibrated on 45 TCV shots: 64529 loses six upper ne channels reading 0.06-0.57 of their neighbours in 88-100 percent of slices
# (its Te is fine, so the DEFUSE ne fits rang through them),
# and 11 other channel readings go in 9 shots, among them Te channel 13 of 60122 and 60181 and ne channel 9 of 74134.
# Only low channels are judged: a high test flags good channels whose neighbourhood a low one drags down.
PERSISTENT_HALFWIDTH = 0.05
PERSISTENT_RHO_MAX = 0.95
PERSISTENT_LOW_RATIO = 0.6
PERSISTENT_MIN_FRACTION = 0.5
PERSISTENT_MIN_SLICES = 10
PERSISTENT_MIN_NEIGHBOURS = 3


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


def low_side_channels(
    rho_tor_norm: np.ndarray, chord_position: np.ndarray
) -> np.ndarray:
    """Mark the channels on the low side of the chord's crossing of the axis, slice by slice.

    The branches split at the channel of lowest rho_tor_norm,
    where the chord passes closest to the magnetic axis of the reconstruction the channels were mapped through.
    The low side is where the channel position along the chord is below that channel's,
    the inboard branch for a major radius, the branch below the axis for a height.

    Args:
        rho_tor_norm: (n_t, n_ch) channel positions, NaN where unmapped.
        chord_position: (n_t, n_ch) channel positions along the chord [m].

    Returns:
        (n_t, n_ch) mask of the low side channels, False in slices with no mapped channel.
    """
    mapped = np.isfinite(rho_tor_norm).any(axis=1)
    rho_filled = np.where(np.isfinite(rho_tor_norm), rho_tor_norm, np.inf)
    i_axis = np.argmin(rho_filled, axis=1)
    position_axis = np.take_along_axis(chord_position, i_axis[:, None], axis=1)
    return mapped[:, None] & (chord_position < position_axis)


def branch_disagreement_errors(
    rho_tor_norm: np.ndarray, y: np.ndarray, err: np.ndarray, low_side: np.ndarray
) -> np.ndarray:
    """Inflate the errors by half the local disagreement between the two branches of a chord.

    See the BRANCH_* constants for the calibration.
    A channel with fewer than BRANCH_MIN_CHANNELS estimates within BRANCH_SMOOTH_HALFWIDTH keeps its error.

    Args:
        rho_tor_norm: (n_t, n_ch) channel positions, NaN where unmapped.
        y: (n_t, n_ch) channel values, NaN where invalid.
        err: (n_t, n_ch) channel errors.
        low_side: (n_t, n_ch) mask of one branch (low_side_channels).

    Returns:
        The (n_t, n_ch) inflated errors.
    """
    err_out = np.array(err, dtype=float)
    for i_time in range(y.shape[0]):
        rho = rho_tor_norm[i_time]
        valid = np.isfinite(rho) & np.isfinite(y[i_time])
        is_low_side = low_side[i_time]
        delta = np.full(rho.shape, np.nan)
        for side in (True, False):
            this = np.flatnonzero(valid & (is_low_side == side))
            other = valid & (is_low_side != side)
            if other.sum() < 2 or this.size == 0:
                continue
            order = np.argsort(rho[other])
            rho_other = rho[other][order]
            y_other = y[i_time][other][order]
            right = np.searchsorted(rho_other, rho[this])
            right_clipped = np.clip(right, 1, rho_other.size - 1)
            gap = rho_other[right_clipped] - rho_other[right_clipped - 1]
            bracketed = (right > 0) & (right < rho_other.size) & (gap <= BRANCH_MAX_GAP)
            y_other_at_this = np.interp(rho[this], rho_other, y_other)
            delta_this = y[i_time][this] - y_other_at_this
            delta[this[bracketed]] = delta_this[bracketed]
        abs_delta = np.abs(delta)
        has_delta = np.isfinite(abs_delta)
        disagreement = np.full(rho.shape, np.nan)
        for k in np.flatnonzero(valid):
            near = np.abs(rho - rho[k]) <= BRANCH_SMOOTH_HALFWIDTH
            window = has_delta & near
            if window.sum() >= BRANCH_MIN_CHANNELS:
                disagreement[k] = np.median(abs_delta[window])
        inflate = np.isfinite(disagreement)
        err_out[i_time, inflate] = np.hypot(
            err[i_time, inflate], 0.5 * disagreement[inflate]
        )
    return err_out


def persistently_low_channels(x_rows: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """Find the channels that read far under their rho neighbours in most slices of a shot.

    In each slice a reading inside PERSISTENT_RHO_MAX is divided by the median of the other readings
    within PERSISTENT_HALFWIDTH of it in rho (at least PERSISTENT_MIN_NEIGHBOURS of them).
    A channel judged in at least PERSISTENT_MIN_SLICES slices is low
    when that ratio is under PERSISTENT_LOW_RATIO in more than PERSISTENT_MIN_FRACTION of them.
    One variable at a time, since a broken ne calibration leaves Te alone.

    Args:
        x_rows: (n_t, n_ch) channel rho_tor_norm positions.
        y_rows: (n_t, n_ch) channel values, NaN where invalid.

    Returns:
        (n_ch,) mask of the persistently low channels.
    """
    n_channels = y_rows.shape[1]
    low_counts = np.zeros(n_channels, dtype=int)
    judged_counts = np.zeros(n_channels, dtype=int)
    for row in range(y_rows.shape[0]):
        x_row = x_rows[row]
        y_row = y_rows[row]
        with np.errstate(invalid="ignore"):
            mask_judged = (
                np.isfinite(x_row) & np.isfinite(y_row) & (x_row < PERSISTENT_RHO_MAX)
            )
        idx_judged = np.flatnonzero(mask_judged)
        for channel in idx_judged:
            mask_near = (
                np.abs(x_row[idx_judged] - x_row[channel]) <= PERSISTENT_HALFWIDTH
            )
            idx_near = idx_judged[mask_near & (idx_judged != channel)]
            if idx_near.size < PERSISTENT_MIN_NEIGHBOURS:
                continue
            neighbour_median = np.median(y_row[idx_near])
            judged_counts[channel] += 1
            low_counts[channel] += (
                y_row[channel] < PERSISTENT_LOW_RATIO * neighbour_median
            )
    mask_enough = judged_counts >= PERSISTENT_MIN_SLICES
    low_fraction = low_counts / np.maximum(judged_counts, 1)
    return mask_enough & (low_fraction > PERSISTENT_MIN_FRACTION)
