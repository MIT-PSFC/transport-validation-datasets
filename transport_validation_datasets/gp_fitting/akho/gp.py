"""GP-residual-correction stage of the akho method.

The two-stage strategy fits `data - analytic_mean` with a GP (rational-
quadratic kernel), rather than GP-fitting the raw data directly: the analytic
mtanh/cubic fit (fit_functions.py) supplies the dominant pedestal/core shape
and a well-behaved derivative in sparse regions, and the GP picks up whatever
shape that fit misses. worker_akho.py's `_fit_variable` stacks this residual
fit back on top of the analytic mean (both value and derivative) to get the
final profile.

Ported from `cmod_to_imas/fit_cmod.py`'s `setup_gpr1d_fit_subtracted_mean`/
`perform_gpr1d_fit`/`save_gpr1d_data`, stripped of their plotting/netCDF side
effects (standalone-script diagnostics, not needed inside a cluster worker)
and reduced to a single call returning arrays.
"""

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from mkgp.core.kernels import RQ_Kernel
from mkgp.core.routines import GaussianProcess

# Kernel start point and hyperparameter bounds: [amplitude, length scale, RQ
# order] rows (lower, upper). Matches fit_cmod.py's
# setup_gpr1d_fit_subtracted_mean exactly.
_KERNEL_START = (1.0e0, 3.0e-1, 1.0e1)
_KERNEL_BOUNDS = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 5.0e1]])
_ERROR_KERNEL_START = (1.0e0, 3.0e-1, 1.0e1)
_ERROR_KERNEL_BOUNDS = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e0, 1.0e0, 5.0e1]])
_ERROR_NRESTARTS = 5
_NRESTARTS = 5
_IMAX = 1000

# Main-kernel regularization (mkgp regpar: multiplies the kernel-complexity
# penalty in the hyperparameter search). fit_cmod.py used 2.0; set to 1.5
# per explicit direction (after a stint at mkgp's default of 1.0).
_REGPAR = 1.5

# Error-kernel regularization (mkgp regpar: multiplies the kernel-complexity
# penalty in the error-kernel hyperparameter search). fit_cmod.py used 3.0;
# lowered per explicit direction. Note the direction of the knob, measured on
# shot 1030516024 / 0.7-0.9s: LOWER values free the heteroscedastic error
# kernel to absorb more of the residual as noise, further shrinking the GP
# posterior mean's contribution to the stacked fit (te GP-mean rms 0.008 at
# 3.0, 0.003 at 1.5, ~0.0005 at 1.0, vs residual rms 0.042) and slightly
# widening the predictive std; ne is largely insensitive. Raise it instead to
# push residual structure into the GP mean.
_ERROR_REGPAR = 1.5

# Columns of this method's hyps diagnostic arrays.
HYP_NAMES = ("rq_amplitude", "rq_length_scale", "rq_order")


@dataclass
class ResidualFit:
    """One GP-fitted residual, evaluated on the target grid.

    Attributes:
        fit: (n_x,) posterior mean of the residual.
        std: (n_x,) predictive std of the residual.
        grad: (n_x,) posterior derivative d/drho of the residual.
        grad_std: (n_x,) latent derivative std of the residual.
        hyps: Optimized RQ kernel hyperparameters (see HYP_NAMES).
    """

    fit: np.ndarray
    std: np.ndarray
    grad: np.ndarray
    grad_std: np.ndarray
    hyps: np.ndarray


def fit_residual(
    x,
    residual,
    err,
    x_star,
    *,
    anchor_x=None,
    anchor_y=None,
    anchor_err=None,
    anchor_dx=None,
    anchor_dy=None,
    anchor_dyerr=None,
) -> ResidualFit | None:
    """GP-fit a profile's residual (data - analytic mean) and predict on x_star.

    A zero-gradient virtual observation is anchored at rho=0 (the axis
    boundary condition `fit_cmod.py`'s original algorithm always included).
    Callers may add further virtual observations -- values via anchor_x/y/err
    and derivatives via anchor_dx/dy/dyerr -- all expressed in the same
    residual space as `residual` (worker_akho.py uses these for its outer-SOL
    anchors).

    Args:
        x: Channel rho positions (NaN-free).
        residual: Channel residual values (data - analytic mean), normalized.
        err: Channel errors, normalized the same way as residual.
        x_star: Target rho grid.
        anchor_x: Rho positions of extra value observations, or None.
        anchor_y: Their residual values (same length as anchor_x).
        anchor_err: Their errors (same length as anchor_x).
        anchor_dx: Rho positions of extra derivative observations, or None.
        anchor_dy: Their residual derivatives (same length as anchor_dx).
        anchor_dyerr: Their errors (same length as anchor_dx).

    Returns:
        The fitted residual, or None if the GP fit failed.
    """
    x = np.asarray(x, dtype=float)
    residual = np.asarray(residual, dtype=float)
    err = np.asarray(err, dtype=float)
    if anchor_x is not None:
        x = np.concatenate([x, np.asarray(anchor_x, dtype=float)])
        residual = np.concatenate([residual, np.asarray(anchor_y, dtype=float)])
        err = np.concatenate([err, np.asarray(anchor_err, dtype=float)])
    dxdata = np.array([0.0])
    dydata = np.array([0.0])
    dyerr = np.array([0.0])
    if anchor_dx is not None:
        dxdata = np.concatenate([dxdata, np.asarray(anchor_dx, dtype=float)])
        dydata = np.concatenate([dydata, np.asarray(anchor_dy, dtype=float)])
        dyerr = np.concatenate([dyerr, np.asarray(anchor_dyerr, dtype=float)])

    gp = GaussianProcess()
    gp._imax = _IMAX
    gp.set_kernel(
        kernel=RQ_Kernel(*_KERNEL_START), kbounds=_KERNEL_BOUNDS, regpar=_REGPAR
    )
    gp.set_error_kernel(
        kernel=RQ_Kernel(*_ERROR_KERNEL_START),
        kbounds=_ERROR_KERNEL_BOUNDS,
        regpar=_ERROR_REGPAR,
        nrestarts=_ERROR_NRESTARTS,
    )
    gp.set_raw_data(
        xdata=x,
        ydata=residual,
        yerr=err,
        xerr=np.zeros_like(x),
        dxdata=dxdata,
        dydata=dydata,
        dyerr=dyerr,
    )
    gp.set_search_parameters(epsilon=1.0e-1, method="adam", spars=[1.0e-2, 0.9, 0.99])
    gp.set_error_search_parameters(
        epsilon=1.0e-1, method="adam", spars=[1.0e-2, 0.9, 0.99]
    )

    try:
        # mkgp prints optimizer status to stdout; keep worker logs clean.
        with contextlib.redirect_stdout(io.StringIO()):
            gp.GPRFit(
                np.asarray(x_star, dtype=float), hsgp_flag=True, nrestarts=_NRESTARTS
            )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None

    fit_y, fit_y_err, fit_dydx, fit_dydx_err = gp.get_gp_results()
    _, hyps, _ = gp.get_gp_kernel_details()
    return ResidualFit(
        fit=np.asarray(fit_y, dtype=float).ravel(),
        std=np.asarray(fit_y_err, dtype=float).ravel(),
        grad=np.asarray(fit_dydx, dtype=float).ravel(),
        grad_std=np.asarray(fit_dydx_err, dtype=float).ravel(),
        hyps=np.asarray(hyps, dtype=float),
    )
