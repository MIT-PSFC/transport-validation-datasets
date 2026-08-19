"""GP profile fitting: analytic mtanh/cubic pre-fit + mkgp GP-residual correction.

Each (shot, time slice) is fit independently, te and ne each on their own
(unlike worker_zk.py, this method does not tie the Te pedestal location to
ne's -- see below). Each variable's fit runs the akho.fit_functions analytic
pre-fit (an Osborne-tanh/cubic curve_fit, picking whichever gets the better
chi-squared), then GP-fits the residual between the data and that analytic
curve with a rational-quadratic kernel (akho.gp), and stacks the two back
together for both the fitted value and its derivative (see `_fit_variable`).

Ported from `cmod_to_imas/fit_cmod.py`'s `pre_fit_te`/`pre_fit_ne` (this
project's cmod_to_imas directory), adapted to consume/produce this package's
`SliceTask`/`VariableFit` instead of raw MDSplus-shaped inputs -- the raw
Thomson data fetch/staging those functions also did is not ported, since
`transport_validation_datasets.machine.cmod.cmod_dataset` already does that
job (via disruption_py) independently.

Two things `fit_cmod.py`'s original algorithm did are intentionally left out,
since they need external per-shot calibration data that `FitBatch`/
`ShotFitInput` do not carry today: the two-point-model Te-separatrix shift
(`apply_2pt_shift`, which also ties ne's x-shift to Te's -- the reason te/ne
fit fully independently here, unlike in fit_cmod.py) and TCI-based ne
recalibration (`tci_calibration.scale_core_ne_to_tci`). Both would need a
`FitBatch`/`FitBounds` schema extension to carry the calibration targets, not
implemented here.

STATUS_REPAIRED/STATUS_CULLED (worker_zk.py's nonphysical-peak repair
heuristics, see zk/quality.py) have no counterpart in fit_cmod.py's original
algorithm and are unused by this method -- batch_io.py's status vocabulary is
explicitly method-defined, so this is a deliberate choice, not a gap.

Runs standalone on the cluster like so:
`python -m transport_validation_datasets.gp_fitting.worker_akho input.npz output.npz --num-workers N`
The import chain must stay within stdlib + numpy + mkgp (see batch_io.py's module docstring)
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
    CubicZeroAxisSlope,
    _fit_one_profile,
    evaluate_with_gradient,
    get_fit_function,
)
from transport_validation_datasets.gp_fitting.akho.gp import (  # noqa: E402
    HYP_NAMES,  # noqa: F401 -- re-exported: workflow.py reads it off this module
    fit_residual,
)
from transport_validation_datasets.gp_fitting.batch_io import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.worker_base import (  # noqa: E402
    SliceResult,
    SliceTask,
    VariableFit,
    map_slices,
    run_worker_cli,
)

# The analytic pre-fit always uses a cubic-inboard, flat-SOL Osborne tanh (or
# its zero-axis-slope cubic-polynomial fallback) -- fit_cmod.py never varied
# this, so it is not exposed as a batch-level knob.
_FIT_FUNC, _N_PARAMS = get_fit_function(core_order=3, sol_order=0)


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
) -> VariableFit:
    """Fit one variable of one time slice: analytic pre-fit + GP-residual correction.

    Args:
        x: Channel rho positions.
        y: Channel values, NaN where invalid.
        err: Channel errors.
        x_star: Target rho grid.
        min_points: Minimum valid channels to attempt a fit.
        variable: 'te' or 'ne' (selects the internal unit-scale factor the
            analytic pre-fit uses, see below).

    Returns:
        The variable's fit with its STATUS_* code; arrays are None for
        skipped and failed slices (this method never repairs/culls, see
        module docstring).
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

    prof, _chi, fname, popt = _fit_one_profile(
        cx, cy / scale, cerr / scale, cx, _FIT_FUNC, _N_PARAMS,
        enforce_mtanh=False, use_edge_chi_squared=False,
        profile_type=variable, edge_thresh=0.93,
    )
    if prof is None:
        return _no_fit(STATUS_FAILED)

    # _fit_one_profile normalizes ne by an extra 1e20 internally (its own
    # `scale = 1e20 if is_ne else 1.0`) that is NOT reflected in `popt` --
    # re-evaluating the fit function at `popt` needs the same factor, or the
    # result is wrong by exactly 1e20 for ne (a no-op for te, whose internal
    # factor is 1.0). See fit_cmod.py's _fit_and_gp_correct_slice docstring
    # for the original discovery of this.
    internal_scale = 1.0e20 if variable == "ne" else 1.0
    mean_func = _FIT_FUNC if fname == "mtanh" else CubicZeroAxisSlope
    mean_at_data, _ = evaluate_with_gradient(mean_func, popt, cx)
    mean_at_data = mean_at_data * internal_scale
    mean_x_star, mean_grad_x_star = evaluate_with_gradient(mean_func, popt, x_star)
    mean_x_star = mean_x_star * internal_scale
    mean_grad_x_star = mean_grad_x_star * internal_scale

    residual = cy / scale - mean_at_data
    residual_err = cerr / scale
    resid = fit_residual(cx, residual, residual_err, x_star)
    if resid is None:
        return _no_fit(STATUS_FAILED)

    # Stack the GP-residual fit back onto the analytic mean for both the
    # value and its derivative. The GP only ever saw the residual, so its own
    # posterior mean/derivative do not include the analytic mean's own
    # value/slope. The error terms are the GP-residual's alone: the analytic
    # mean's own parameter uncertainty (from curve_fit's covariance) is
    # treated as zero for now, per explicit direction.
    fit = scale * (mean_x_star + resid.fit)
    std = scale * resid.std
    grad = scale * (mean_grad_x_star + resid.grad)
    grad_std = scale * resid.grad_std
    return VariableFit(
        fit=fit, std=std, grad=grad, grad_std=grad_std, hyps=resid.hyps,
        status=STATUS_OK,
    )


def _fit_slice(task: SliceTask) -> SliceResult:
    """Fit Te and ne for one (shot, time slice), independently.

    Args:
        task: The slice's channel data and fit settings.

    Returns:
        Both variables' fits for the slice.
    """
    te = _fit_variable(task.x, task.te_y, task.te_err, task.x_star, task.min_points, "te")
    ne = _fit_variable(task.x, task.ne_y, task.ne_err, task.x_star, task.min_points, "ne")
    return SliceResult(shot=task.shot, i_time=task.i_time, te=te, ne=ne)


def fit_batch(
    batch: FitBatch,
    num_workers: int = 1,
    *,
    max_slices_per_shot: int | None = None,
) -> dict[int, ShotFitOutput]:
    """Fit every (shot, time slice) in the batch with the akho method.

    Slices are fit serially (num_workers <= 1) or across worker processes;
    both fit stages (curve_fit and mkgp) are single-threaded, so parallelism
    comes only from the slice-level pool (BLAS threads are pinned at module
    top).

    Args:
        batch: Staged batch inputs (see batch_io.FitBatch).
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
    """Run the akho worker CLI: input.npz output.npz --num-workers N.

    Args:
        argv: Command-line arguments; None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="worker_akho", argv=argv)


if __name__ == "__main__":
    main()
