"""GP profile fitting: analytic mtanh/cubic pre-fit + mkgp GP-residual correction.

Each (shot, time slice) is fit independently, te and ne each on their own
(unlike worker_zk.py, this method does not tie the Te pedestal location to
ne's -- see below). Each variable's fit runs the akho.fit_functions analytic
pre-fit (an Osborne-tanh/cubic curve_fit, picking whichever gets the
better chi-squared), then GP-fits the residual between the data and that analytic
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

Optional coordinate substitution: `FitBatch.fit_coordinate` (default "rho")
lets a batch be fit in `psi_norm`, `sqrt_psi_norm`, `phi_norm`, or
`sqrt_phi_norm` instead -- see `_resolve_fit_coordinate`, called once per
slice at the top of `_fit_slice`, before either variable's analytic pre-fit/
GP-residual stages run. Unlike worker_zk.py's rho-tuned pipeline, this
method's analytic bounds/initial guesses/edge_thresh in fit_functions.py were
themselves ported from `fit_cmod.py` already tuned against psi_norm, not
rho (see fit_functions.py's own history) -- so no bound recalibration is
needed for that coordinate specifically; the other three are not recalibrated
either, by explicit direction, since psi_norm was already the intended one.
`x_star` is not re-derived here: it is staged directly in `fit_coordinate`'s
units (a batch fit in psi_norm stages an x_star grid of psi_norm values, not
a per-slice transform of a shared rho grid, since the rho<->other-coordinate
mapping is equilibrium/time-slice dependent and there is no single shared
grid across slices to transform).

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
from transport_validation_datasets.gp_fitting.akho.gp import (  # noqa: F401 -- re-exported: workflow.py reads it off this module; noqa: E402
    HYP_NAMES,
    fit_residual,
)
from transport_validation_datasets.gp_fitting.batch_io import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.coordinates import (  # noqa: E402
    coordinates_from_psi_norm,
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
# this, so it is not exposed as a batch-level knob. The zero_axis_slope
# variant (a departure from fit_cmod.py, per explicit direction) eliminates
# the inboard linear term so the mtanh shares the cubic fallback's exact
# df/drho = 0 axis boundary condition: the innermost Thomson channel often
# sits at rho ~ 0.25, and an unconstrained inboard polynomial extrapolated a
# strongly peaked, finite-slope core through the axis.
_FIT_FUNC, _N_PARAMS = get_fit_function(core_order=3, sol_order=0, zero_axis_slope=True)

# Outer-SOL anchors (an addition over fit_cmod.py's original algorithm):
# virtual observations at rho > 1 pinning the *total* profile (analytic mean
# + GP residual) to zero value and zero slope well outside the plasma, so the
# GP stage cannot drift in the data-free far SOL. Applied only to the
# variables in _OUTER_ANCHOR_VARIABLES: ne wants the far-SOL pin, while for
# Te these points pulled the analytic edge fit away from the data (per
# explicit direction), so Te relies on the staged rho 1.05/1.08 SOL anchor
# channels alone. Errors are in fit units --
# 100 eV = 1.0e-1 keV for te, 1.0e19 m^-3 = 1.0e-1 x 1e20 m^-3 for ne -- and
# the same magnitudes per unit rho for the slope constraint. The value
# anchors are appended to the analytic pre-fit's input data (echoing the
# original algorithm's add_SOL_zeros_in_psi_coords, pulling the mtanh SOL
# offset toward zero) with their errors inflated by
# _OUTER_ANCHOR_PREFIT_ERR_FACTOR, so they only nudge the analytic shape
# rather than deform it -- the GP stage, which sees them at full weight,
# does the actual pinning. curve_fit takes no derivative observations, so
# the slope constraint acts only at the GP stage. `_fit_variable` also
# converts both to the GP's residual space (subtracting the analytic mean's
# own value/slope and normalizing by the slice scale) before handing them to
# fit_residual, which adds them alongside its rho=0 axis constraint.
# Variables whose analytic pre-fit may pick the mtanh; the rest compete only
# against the zero-axis-slope cubic fallback. Te's mtanh candidate is
# additionally gated on edge-region evidence of a real pedestal (see
# fit_functions._fit_one_profile's Te-only gate) -- without it mtanh's extra
# core freedom let it win almost every window regardless of confinement
# regime. A quintic candidate (x^4/x^5 terms) was tried and dropped: it
# routinely overfit the data-free extrapolation region inside the innermost
# Thomson channel into a nonphysical core dip (see fit_functions.py).
_MTANH_VARIABLES = ("te", "ne")

_OUTER_ANCHOR_VARIABLES = ("ne",)
_OUTER_ANCHOR_RHO = np.array([1.2, 1.3, 1.4])
_OUTER_ANCHOR_ERR = {"te": 1.0e-1, "ne": 1.0e-1}
_OUTER_ANCHOR_GRAD_ERR = {"te": 1.0e-1, "ne": 1.0e-1}
_OUTER_ANCHOR_PREFIT_ERR_FACTOR = 5.0

# fit_coordinate values coordinates_from_psi_norm can resolve, beyond the
# "rho" default (task.x, staged separately -- see batch_io.ShotFitInput.x).
_ALT_COORDINATES = ("psi_norm", "sqrt_psi_norm", "phi_norm", "sqrt_phi_norm")


def _resolve_fit_coordinate(task: SliceTask) -> np.ndarray:
    """Optional top-of-method step: substitute the channel radial coordinate.

    Default (`task.fit_coordinate == "rho"`) is a no-op returning `task.x`
    unchanged -- the calibrated default path. Any other coordinate pivots
    through `task.psi_norm` (see `ShotFitInput`/`SliceTask`) via
    `gp_fitting.coordinates.coordinates_from_psi_norm`.

    A slice with `task.qpsi` missing/NaN (phi_norm/sqrt_phi_norm only, e.g.
    MAST's best-effort qpsi) is not special-cased here: `coordinates_from_psi_norm`
    already returns an all-NaN column for it, which `_fit_variable`'s
    existing `min_points` gate naturally turns into STATUS_SKIPPED for that
    slice -- the same graceful-degradation path a slice with too few valid
    rho channels already takes.

    Args:
        task: The slice's fit task.

    Returns:
        (n_ch,) channel positions in `task.fit_coordinate`'s units.

    Raises:
        ValueError: `fit_coordinate` is not "rho" and `task.psi_norm` was
            never staged (a batch/method configuration mismatch, not a
            per-slice data gap -- see `batch_io.FitBatch.fit_coordinate`),
            or `fit_coordinate` is not a name this method recognizes.
    """
    if task.fit_coordinate == "rho":
        return task.x
    if task.fit_coordinate not in _ALT_COORDINATES:
        raise ValueError(
            f"Unknown fit_coordinate {task.fit_coordinate!r}; expected 'rho' or "
            f"one of {_ALT_COORDINATES}"
        )
    if task.psi_norm is None:
        raise ValueError(
            f"fit_coordinate={task.fit_coordinate!r} requires psi_norm to be "
            "staged on ShotFitInput -- this batch was not staged for it"
        )
    values, _jacobians = coordinates_from_psi_norm(task.psi_norm, None, task.qpsi)
    return getattr(values, task.fit_coordinate)


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
        x: Channel positions, in whatever coordinate the slice is being fit
            in (rho by default; see `_resolve_fit_coordinate` for the
            optional substitution).
        y: Channel values, NaN where invalid.
        err: Channel errors.
        x_star: Target grid, same coordinate as `x`.
        min_points: Minimum valid channels to attempt a fit.
        variable: 'te' or 'ne' (selects the pre-fit bounds, whether the
            mtanh candidate is attempted, and the outer-anchor settings).

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

    # The outer-SOL value anchors join the pre-fit's data (the slope
    # constraint cannot: curve_fit fits values only). The residual/GP stage
    # below still uses the bare channel data -- the anchors re-enter there as
    # their own virtual observations. Anchored variables only (see
    # _OUTER_ANCHOR_VARIABLES).
    use_outer_anchors = variable in _OUTER_ANCHOR_VARIABLES
    pre_x, pre_y, pre_err = cx, cy / scale, cerr / scale
    if use_outer_anchors:
        pre_anchor_err = (
            _OUTER_ANCHOR_PREFIT_ERR_FACTOR * _OUTER_ANCHOR_ERR[variable] / scale
        )
        pre_x = np.concatenate([cx, _OUTER_ANCHOR_RHO])
        pre_y = np.concatenate([pre_y, np.zeros_like(_OUTER_ANCHOR_RHO)])
        pre_err = np.concatenate(
            [pre_err, np.full_like(_OUTER_ANCHOR_RHO, pre_anchor_err)]
        )
    prof, _chi, fname, popt = _fit_one_profile(
        pre_x,
        pre_y,
        pre_err,
        pre_x,
        _FIT_FUNC,
        _N_PARAMS,
        enforce_mtanh=False,
        use_edge_chi_squared=False,
        profile_type=variable,
        edge_thresh=0.93,
        attempt_mtanh=variable in _MTANH_VARIABLES,
    )
    if prof is None:
        return _no_fit(STATUS_FAILED)

    # popt is in the same normalized space as the data handed to
    # _fit_one_profile (its historical internal 1e20 ne rescale is gone --
    # see the note in its body), so the fit function re-evaluates directly.
    mean_func = {
        "mtanh": _FIT_FUNC,
        "cubic": CubicZeroAxisSlope,
    }[fname]
    mean_at_data, _ = evaluate_with_gradient(mean_func, popt, cx)
    mean_x_star, mean_grad_x_star = evaluate_with_gradient(mean_func, popt, x_star)

    # Outer-SOL anchors in residual space: the GP fits
    # (data / scale - analytic mean), so pinning the total profile to zero
    # value/slope at these rho means pinning the residual to minus the
    # analytic mean's own value/slope there.
    anchor_kwargs = {}
    if use_outer_anchors:
        anchor_mean, anchor_mean_grad = evaluate_with_gradient(
            mean_func, popt, _OUTER_ANCHOR_RHO
        )
        anchor_kwargs = dict(
            anchor_x=_OUTER_ANCHOR_RHO,
            anchor_y=-anchor_mean,
            anchor_err=np.full_like(
                _OUTER_ANCHOR_RHO, _OUTER_ANCHOR_ERR[variable] / scale
            ),
            anchor_dx=_OUTER_ANCHOR_RHO,
            anchor_dy=-anchor_mean_grad,
            anchor_dyerr=np.full_like(
                _OUTER_ANCHOR_RHO, _OUTER_ANCHOR_GRAD_ERR[variable] / scale
            ),
        )

    residual = cy / scale - mean_at_data
    residual_err = cerr / scale
    resid = fit_residual(cx, residual, residual_err, x_star, **anchor_kwargs)
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
        fit=fit,
        std=std,
        grad=grad,
        grad_std=grad_std,
        hyps=resid.hyps,
        status=STATUS_OK,
    )


def _fit_slice(task: SliceTask) -> SliceResult:
    """Fit Te and ne for one (shot, time slice), independently.

    Resolves the fit coordinate once (te and ne share the same channel
    positions) via `_resolve_fit_coordinate`, then fits both variables
    against it.

    Args:
        task: The slice's channel data and fit settings.

    Returns:
        Both variables' fits for the slice.
    """
    x = _resolve_fit_coordinate(task)
    te = _fit_variable(
        x, task.te_y, task.te_err, task.x_star, task.min_points, "te"
    )
    ne = _fit_variable(
        x, task.ne_y, task.ne_err, task.x_star, task.min_points, "ne"
    )
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
