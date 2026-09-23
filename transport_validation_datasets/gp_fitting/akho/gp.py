"""GP residual stage of the akho method.

A rational-quadratic GP fits the data minus the analytic mean (fit_functions.py).
The analytic fit supplies the pedestal and core shape,
and a well-behaved derivative where channels are sparse.
The GP picks up the shape that the analytic fit misses.
worker_akho.py's _fit_variable adds the two back together, value and derivative.
"""

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from mkgp.core.kernels import RQ_Kernel
from mkgp.core.routines import GaussianProcess

# Kernel start points and hyperparameter bounds, each over [amplitude, length scale, RQ order].
# The bounds rows are (lower, upper).
_KERNEL_START = (1.0e0, 3.0e-1, 1.0e1)
_KERNEL_BOUNDS = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 5.0e1]])
_ERROR_KERNEL_START = (1.0e0, 3.0e-1, 1.0e1)
_ERROR_KERNEL_BOUNDS = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e0, 1.0e0, 5.0e1]])
_ERROR_NRESTARTS = 5
_NRESTARTS = 5
_IMAX = 1000

# Main-kernel regularization, mkgp's weight on the kernel-complexity penalty in the hyperparameter search.
_REGPAR = 1.5

# Error-kernel regularization, the same penalty weight in the error-kernel search.
# Lower values let the error kernel absorb more of the residual as noise,
# which shrinks the GP mean's share of the fit and slightly widens the predictive std.
# Higher values push residual structure into the GP mean.
# On shot 1030516024 at 0.7-0.9 s, the Te GP-mean rms was 0.008 at 3.0, 0.003 at 1.5 and ~0.0005 at 1.0,
# against a residual rms of 0.042. ne is largely insensitive.
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
    x, residual, err, x_star, value_anchors, grad_anchors
) -> ResidualFit | None:
    """GP-fit a profile's residual (data - analytic mean) and predict on x_star.

    Args:
        x: Channel rho positions (NaN-free).
        residual: Channel residual values (data - analytic mean), normalized.
        err: Channel errors, normalized the same way as residual.
        x_star: Target rho grid.
        value_anchors: (n, 3) rows of (rho, residual, error), in residual space.
        grad_anchors: (n, 3) rows of (rho, residual gradient, error), in residual space.

    Returns:
        The fitted residual, or None if the GP fit failed.
    """
    x_data = np.asarray(x, dtype=float)
    residual_data = np.asarray(residual, dtype=float)
    err_data = np.asarray(err, dtype=float)
    xdata = np.concatenate([x_data, value_anchors[:, 0]])
    ydata = np.concatenate([residual_data, value_anchors[:, 1]])
    yerr = np.concatenate([err_data, value_anchors[:, 2]])

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
        xdata=xdata,
        ydata=ydata,
        yerr=yerr,
        xerr=np.zeros_like(xdata),
        dxdata=grad_anchors[:, 0],
        dydata=grad_anchors[:, 1],
        dyerr=grad_anchors[:, 2],
    )
    gp.set_search_parameters(epsilon=1.0e-1, method="adam", spars=[1.0e-2, 0.9, 0.99])
    gp.set_error_search_parameters(
        epsilon=1.0e-1, method="adam", spars=[1.0e-2, 0.9, 0.99]
    )

    try:
        # mkgp prints optimizer status to stdout, which would clutter the worker logs
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
