"""GP profile fitting with mkgp (Gibbs kernel, tanh-warped length scale).

Each (shot, time slice) is fit independently, Te and ne each on their own,
with the kernel's length-scale transition at the configured pedestal location.
Each variable's fit runs the cleaning pipeline (cleaning.py),
the GP fit with edge boundary conditions and monotonic-edge repair (gp.py),
and the nonphysical-fit checks with up to three repairs (quality.py, see _fit_variable).

Runs standalone on the cluster like so:
`python -m transport_validation_datasets.gp_fitting.worker_zk input.npz output.npz --num-workers N`

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
"""

import os
from dataclasses import replace

# Limit BLAS threads before numpy loads so slice-level multiprocessing
# (fit_batch num_workers) does not oversubscribe cores
# mkgp is single-threaded, so one thread per worker is right.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from transport_validation_datasets.gp_fitting.batch_io import (  # noqa: E402
    STATUS_CULLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REPAIRED,
    STATUS_SKIPPED,
    FitAnchors,
    FitBatch,
    FitBounds,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.worker_base import (  # noqa: E402
    SliceResult,
    SliceTask,
    VariableFit,
    map_slices,
    run_worker_cli,
)
from transport_validation_datasets.gp_fitting.zk.cleaning import (  # noqa: E402
    clean_channels,
)
from transport_validation_datasets.gp_fitting.zk.gp import (  # noqa: E402
    fit_profile,
)
from transport_validation_datasets.gp_fitting.zk.quality import (  # noqa: E402
    REPAIR_HALFWIDTH,
    fit_ignores_data,
    nonphysical_peak,
)

# Columns of the hyps diagnostic arrays in this method's outputs.
HYP_NAMES = ("var", "l1", "l2", "lw")

# Core length-scale floor of the smooth-rescue attempt (see _fit_variable).
# High enough to force the optimizer out of a short-l1 basin, below the 0.9
# ceiling so the optimizer still has a range to search.
SMOOTH_RESCUE_L1_MIN = 0.7


def _no_fit(status: int) -> VariableFit:
    """Build the all-None VariableFit for a slice that produced no fit.

    Args:
        status: STATUS_* code explaining why.

    Returns:
        VariableFit with every array None.
    """
    return VariableFit(
        fit=None, std=None, grad=None, grad_std=None, hyps=None, status=status
    )


def _attempt_fit(
    x, y, err, x_star, scale_per_slice, bounds, anchors, pedestal_rho, seed_salt=0
):
    """Clean, fit, and rescale one variable of one slice, once.

    Args:
        x: Channel rho positions.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        x_star: Target rho grid.
        scale_per_slice: Normalize by the cleaned slice max before fitting.
        bounds: The variable's staged bound knobs.
        anchors: The variable's anchors, in the data's own units.
        pedestal_rho: The kernel's length-scale transition center.
        seed_salt: Restart seed offset for a reseeded retry (see run_gp).

    Returns:
        (fit, std, grad, grad_std, hyps) in the data's own units, or None when
        cleaning left nothing usable or the GP fit failed. Gradients are not
        clamped: negative slopes are physical.
    """
    cleaned = clean_channels(x, y, err, bounds, anchors, pedestal_rho, scale_per_slice)
    if cleaned is None:
        return None
    cx, cy, cerr, scale, scaled_anchors = cleaned
    pf = fit_profile(
        cx, cy, cerr, x_star, bounds, scaled_anchors, pedestal_rho, seed_salt=seed_salt
    )
    if pf is None:
        return None
    return (
        pf.fit * scale,
        pf.std * scale,
        pf.grad * scale,
        pf.grad_std * scale,
        pf.hyps,
    )


def _fit_problems(y_out, x_star, x, y, err):
    """Run the nonphysical-fit checks on one fitted slice.

    Args:
        y_out: Fitted profile in the data's own units.
        x_star: rho grid of the fit.
        x: Channel rho positions (as fit, i.e. after any channel drop).
        y: Channel values.
        err: Channel errors.

    Returns:
        (peak_rho, biased): the nonphysical peak location or None, and whether
        the fit ignores its innermost channels (only checked when there is no
        peak - a peak's repair comes first).
    """
    peak = nonphysical_peak(y_out, x_star, x, y, err)
    biased = peak is None and fit_ignores_data(x, y, err, x_star, y_out)
    return peak, biased


def _fit_variable(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    x_star: np.ndarray,
    min_points: int,
    scale_per_slice: bool,
    bounds: FitBounds,
    anchors: FitAnchors,
    pedestal_rho: float,
) -> VariableFit:
    """Fit one variable of one time slice, with up to three repairs.

    1: Fit the variable.
    2: If nonphysical or failed, retry from a fresh restart seed
    (repair a: a borderline slice sits between two near-equal LML basins and the input-hashed seed picks one,
    so a different, still deterministic draw can land the healthy basin).
    3: If still nonphysical, retry with the core length scale floored at SMOOTH_RESCUE_L1_MIN
    (repair b: the LML can prefer a short-l1 basin that explains mid-profile wiggles
    by sacrificing the innermost channel cluster,
    and the floor forces the smooth basin, which the checks then judge like any other fit).
    4: If still nonphysical with a peak over droppable channels, refit without them
    (repair c: a stray point or a miscalibrated block).
    5: A peak where there is no data to drop is extrapolation ringing,
    and a biased fit with no peak has no channel subset that repairs it. Both get culled.

    Args:
        x: Channel rho positions.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        x_star: Target rho grid.
        min_points: Minimum valid channels to attempt a fit.
        scale_per_slice: Normalize by the cleaned slice max before fitting.
        bounds: The variable's staged bound knobs.
        anchors: The variable's anchors, in the data's own units.
        pedestal_rho: The kernel's length-scale transition center.

    Returns:
        The variable's fit with its STATUS_* code; arrays are None for
        skipped, failed, and culled slices.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if int(valid.sum()) < min_points:
        return _no_fit(STATUS_SKIPPED)

    # The attempt ladder: as staged, then a fresh restart draw (repair a),
    # then the smooth basin (repair b).
    # The first clean fit returns, and the last flagged fit feeds the channel-drop repair below.
    attempts = [(0, bounds), (1, bounds)]
    if bounds.l1_min < SMOOTH_RESCUE_L1_MIN:
        attempts.append((1, replace(bounds, l1_min=SMOOTH_RESCUE_L1_MIN)))

    last_flagged = None
    for n_attempt, (seed_salt, attempt_bounds) in enumerate(attempts):
        result = _attempt_fit(
            x,
            y,
            err,
            x_star,
            scale_per_slice,
            attempt_bounds,
            anchors,
            pedestal_rho,
            seed_salt=seed_salt,
        )
        if result is None:
            continue
        peak, biased = _fit_problems(result[0], x_star, x, y, err)
        if peak is None and not biased:
            status = STATUS_OK if n_attempt == 0 else STATUS_REPAIRED
            return VariableFit(*result, status=status)
        last_flagged = (peak, biased)

    if last_flagged is None:
        return _no_fit(STATUS_FAILED)
    peak, biased = last_flagged

    # Repair c: drop the channels under the nonphysical peak and refit once.
    if peak is None:
        return _no_fit(STATUS_CULLED)
    drop = valid & (np.abs(x - peak) <= REPAIR_HALFWIDTH)
    if not drop.any():
        return _no_fit(STATUS_CULLED)
    y = np.where(drop, np.nan, y)
    still_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if int(still_valid.sum()) < min_points:
        return _no_fit(STATUS_CULLED)
    result = _attempt_fit(
        x, y, err, x_star, scale_per_slice, bounds, anchors, pedestal_rho
    )
    if result is None:
        return _no_fit(STATUS_CULLED)
    peak, biased = _fit_problems(result[0], x_star, x, y, err)
    if peak is not None or biased:
        return _no_fit(STATUS_CULLED)
    return VariableFit(*result, status=STATUS_REPAIRED)


def _fit_slice(task: SliceTask) -> SliceResult:
    """Fit Te and ne for one (shot, time slice), independently.

    Args:
        task: The slice's channel data and fit settings.

    Returns:
        Both variables' fits for the slice.
    """
    te = _fit_variable(
        task.x,
        task.te_y,
        task.te_err,
        task.x_star,
        task.min_points,
        task.scale_per_slice,
        task.te_bounds,
        task.te_anchors,
        task.pedestal_rho_tor_norm,
    )
    ne = _fit_variable(
        task.x,
        task.ne_y,
        task.ne_err,
        task.x_star,
        task.min_points,
        task.scale_per_slice,
        task.ne_bounds,
        task.ne_anchors,
        task.pedestal_rho_tor_norm,
    )
    return SliceResult(shot=task.shot, i_time=task.i_time, te=te, ne=ne)


def fit_batch(
    batch: FitBatch,
    num_workers: int = 1,
    *,
    max_slices_per_shot: int | None = None,
) -> dict[int, ShotFitOutput]:
    """Fit every (shot, time slice) in the batch with the mkgp method.

    Hyperparameters are optimized for each individual slice, since plasma
    conditions (and thus profile shapes) change over the course of a shot.
    Slices are fit serially (num_workers <= 1) or across worker processes;
    mkgp fits are single-threaded, so parallelism comes only from the
    slice-level pool (BLAS threads are pinned at module top).

    Args:
        batch: Staged batch inputs.
        num_workers: Slice-level worker processes.
        max_slices_per_shot: If set, only fit the first N time slices of each
            shot (debug aid).

    Returns:
        Fitted profiles keyed by shot number.
    """
    return map_slices(
        _fit_slice,
        batch,
        num_workers=num_workers,
        max_slices_per_shot=max_slices_per_shot,
    )


def main(argv: list[str] | None = None):
    """Run the mkgp worker CLI: input.npz output.npz --num-workers N.

    Args:
        argv: Command-line arguments; None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="worker_zk", argv=argv)


if __name__ == "__main__":
    main()
