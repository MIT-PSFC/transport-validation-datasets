"""GP profile fitting: an analytic mtanh or cubic pre-fit, then a GP on the residual.

Each (shot, time slice) is fit independently, and Te and ne are independelty fit as well.
A variable's fit runs the analytic pre-fit (akho/fit_functions.py),
fits the residual between the data and that curve with a rational-quadratic GP (akho/gp.py),
and adds the two back together, value and derivative (see _fit_variable).
This method never repairs or culls a slice.

FitBatch.fit_coordinate can substitute psi_norm, sqrt_psi_norm, phi_norm or sqrt_phi_norm for rho
(see _resolve_fit_coordinate).
x_star is staged in the fit coordinate's units.

Runs standalone on the cluster like so:
`python -m transport_validation_datasets.gp_fitting.worker_akho input.npz output.npz --num-workers N`

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
"""

import os

# Limit BLAS threads before numpy loads so slice-level multiprocessing
# (fit_batch num_workers) does not oversubscribe cores.
# mkgp is single-threaded, so one thread per worker is right.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from transport_validation_datasets.gp_fitting.akho.fit_functions import (  # noqa: E402
    evaluate_with_gradient,
    fit_analytic_profile,
)

# HYP_NAMES is re-exported, workflow.py reads it off the worker module
from transport_validation_datasets.gp_fitting.akho.gp import (  # noqa: E402, F401
    HYP_NAMES,
    fit_residual,
)
from transport_validation_datasets.gp_fitting.batch_io import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    FitAnchors,
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.coordinates import (  # noqa: E402
    SUBSTITUTABLE_COORDINATES,
    substitute_channel_positions,
)
from transport_validation_datasets.gp_fitting.worker_base import (  # noqa: E402
    SliceResult,
    SliceTask,
    VariableFit,
    map_slices,
    run_worker_cli,
)

# The value anchors join the analytic pre-fit with their errors inflated by this factor,
# so they nudge its SOL level rather than deform its shape.
# The GP stage sees them at full weight.
_PREFIT_ANCHOR_ERR_FACTOR = 5.0

# fit_coordinate values other than "rho" that coordinates_from_psi_norm resolves
_ALT_COORDINATES = SUBSTITUTABLE_COORDINATES


def _resolve_fit_coordinate(task: SliceTask) -> np.ndarray:
    """Get the slice's channel positions in the batch's fit coordinate.

    "rho" returns task.x unchanged.
    Any other coordinate pivots through task.psi_norm
    with coordinates.substitute_channel_positions, which the fit plots also use.
    A phi coordinate on a slice without finite qpsi gives NaN positions,
    and the min_points gate skips the slice.

    Args:
        task: The slice's fit task.

    Returns:
        (n_ch,) channel positions in task.fit_coordinate's units.

    Raises:
        ValueError: If task.psi_norm was never staged for a coordinate other than "rho",
            or the coordinate is unknown.
    """  # noqa: DOC502 -- the ValueError propagates from substitute_channel_positions
    return substitute_channel_positions(
        task.x, task.psi_norm, task.qpsi, task.fit_coordinate
    )


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


def _fit_variable(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    x_star: np.ndarray,
    min_points: int,
    variable: str,
    anchors: FitAnchors,
) -> VariableFit:
    """Fit one variable of one time slice: analytic pre-fit + GP-residual correction.

    Args:
        x: Channel positions in the fit coordinate.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        x_star: Target grid, same coordinate as `x`.
        min_points: Minimum valid channels to attempt a fit.
        variable: 'te' or 'ne', which selects the pre-fit bounds and initial guesses.
        anchors: The variable's anchors, in the data's own units.

    Returns:
        The variable's fit and its STATUS_* code.
        The arrays are None for a skipped or failed slice.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if int(valid.sum()) < min_points:
        return _no_fit(STATUS_SKIPPED)
    cx, cy, cerr = x[valid], y[valid], err[valid]

    scale = float(np.nanmax(cy))
    if not np.isfinite(scale) or scale <= 0.0:
        return _no_fit(STATUS_FAILED)

    scaled_anchors = anchors.scaled(scale)
    value_rows = scaled_anchors.value
    grad_rows = scaled_anchors.grad

    # curve_fit takes no gradient observations, so only the value anchors join the pre-fit
    pre_x = np.concatenate([cx, value_rows[:, 0]])
    pre_y = np.concatenate([cy / scale, value_rows[:, 1]])
    prefit_anchor_err = _PREFIT_ANCHOR_ERR_FACTOR * value_rows[:, 2]
    pre_err = np.concatenate([cerr / scale, prefit_anchor_err])
    is_channel = np.arange(pre_x.size) < cx.size
    analytic = fit_analytic_profile(
        pre_x, pre_y, pre_err, is_channel, variable, edge_thresh=0.93
    )
    if analytic is None:
        return _no_fit(STATUS_FAILED)
    mean_func, popt = analytic

    mean_at_data, _ = evaluate_with_gradient(mean_func, popt, cx)
    mean_x_star, mean_grad_x_star = evaluate_with_gradient(mean_func, popt, x_star)

    # The GP fits data / scale - analytic mean,
    # so an anchor on the total profile becomes the anchor minus the mean there.
    mean_at_value_anchors, _ = evaluate_with_gradient(mean_func, popt, value_rows[:, 0])
    _, mean_grad_at_grad_anchors = evaluate_with_gradient(
        mean_func, popt, grad_rows[:, 0]
    )
    residual_value_anchors = np.column_stack(
        [value_rows[:, 0], value_rows[:, 1] - mean_at_value_anchors, value_rows[:, 2]]
    )
    residual_grad_anchors = np.column_stack(
        [
            grad_rows[:, 0],
            grad_rows[:, 1] - mean_grad_at_grad_anchors,
            grad_rows[:, 2],
        ]
    )

    residual = cy / scale - mean_at_data
    residual_err = cerr / scale
    resid = fit_residual(
        cx,
        residual,
        residual_err,
        x_star,
        residual_value_anchors,
        residual_grad_anchors,
    )
    if resid is None:
        return _no_fit(STATUS_FAILED)

    # The GP saw only the residual, so the analytic mean goes back onto its value and derivative.
    # The errors are the GP's alone, the analytic mean's parameter uncertainty is treated as zero.
    fit = scale * (mean_x_star + resid.fit)
    std = scale * resid.std
    grad = scale * (mean_grad_x_star + resid.grad)
    grad_std = scale * resid.grad_std
    return VariableFit(
        fit=fit,
        std=std,
        grad=grad,
        grad_std=grad_std,
        hyps=resid.hyps,
        status=STATUS_OK,
    )


def _fit_slice(task: SliceTask) -> SliceResult:
    """Fit Te and ne for one (shot, time slice), independently.

    Te and ne share the channel positions, so the fit coordinate is resolved once.

    Args:
        task: The slice's channel data and fit settings.

    Returns:
        Both variables' fits for the slice.
    """
    x = _resolve_fit_coordinate(task)
    te = _fit_variable(
        x, task.te_y, task.te_err, task.x_star, task.min_points, "te", task.te_anchors
    )
    ne = _fit_variable(
        x, task.ne_y, task.ne_err, task.x_star, task.min_points, "ne", task.ne_anchors
    )
    return SliceResult(shot=task.shot, i_time=task.i_time, te=te, ne=ne)


def fit_batch(
    batch: FitBatch,
    num_workers: int = 1,
    *,
    max_slices_per_shot: int | None = None,
) -> dict[int, ShotFitOutput]:
    """Fit every (shot, time slice) in the batch with the akho method.

    Slices are fit serially (num_workers <= 1) or across worker processes.
    curve_fit and mkgp are single-threaded,
    so the slice pool is the only parallelism (BLAS threads are pinned at module top).

    Args:
        batch: Staged batch inputs (see batch_io.FitBatch).
        num_workers: Slice-level worker processes.
        max_slices_per_shot: If set, fit only the first N time slices of each shot, for debugging.

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
    """Run the akho worker CLI: input.npz output.npz --num-workers N.

    Args:
        argv: Command-line arguments, None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="worker_akho", argv=argv)


if __name__ == "__main__":
    main()
