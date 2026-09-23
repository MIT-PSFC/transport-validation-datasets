"""GP fitting core of the zk method: run_gp and fit_profile.

fit_profile fits one already-cleaned, already-normalized profile slice and
returns the profile, its gradient, and their uncertainties on the target rho
grid. run_gp is the raw mkgp interface underneath it (also used by the
cleaning module for its rough reference fit).
"""

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from mkgp.core.kernels import SE_Kernel
from mkgp.core.routines import GaussianProcess

from transport_validation_datasets.gp_fitting.batch_io import FitAnchors, FitBounds
from transport_validation_datasets.gp_fitting.zk.kernel import (
    ERR_HYP_BOUNDS,
    ERR_HYP_START,
    ERR_NRESTARTS,
    bounds_for,
    build_kernel,
    deterministic_seed,
    pinned_hyperparams,
)
from transport_validation_datasets.gp_fitting.zk.quality import (
    MONO_CHECK_RHO,
    MONO_GRAD_ERR,
    MONO_GRAD_TOL,
    MONO_MAX_PASSES,
    rise_is_data_supported,
)

# Half-width (in rho) of the x0 window used to pin the pedestal location when
# tying Te to the ne fit. Narrow enough to hold x0, wide enough to stay a
# valid (lower < upper) bound after clamping to the global x0 range.
X0_PIN_HALFWIDTH = 1.0e-3
# Extra optimizer attempts (beyond the first) when a fit pins a hyperparameter
# at its bound - a different random restart usually escapes the same basin.
MAX_HYP_RETRIES = 2
# Optimizer random restarts for a real fit.
# The cleaning module's rough reference fit runs fewer (only an outlier-judging reference).
NRESTARTS = 8


@dataclass
class ProfileFit:
    """One fitted profile slice, in the (possibly normalized) fit units.

    Attributes:
        fit: (n_x,) posterior mean, clipped at 0.
        std: (n_x,) predictive std (includes observation noise).
        grad: (n_x,) posterior derivative d/drho.
        grad_std: (n_x,) latent derivative std.
        hyps: Fitted [var, l1, l2, lw, x0].
    """

    fit: np.ndarray
    std: np.ndarray
    grad: np.ndarray
    grad_std: np.ndarray
    hyps: np.ndarray


def run_gp(
    data_X,
    data_y,
    err_y,
    x_eval,
    fit_bounds: FitBounds,
    anchors: FitAnchors,
    hyperparams=None,
    optimize=True,
    pin_x0=None,
    extra_grad_bc=None,
    nrestarts=NRESTARTS,
    hyp_retries=MAX_HYP_RETRIES,
    seed_salt=0,
) -> GaussianProcess | None:
    """Set up the GP with the anchors and fit it.

    With optimize=True and hyperparams=None the hyperparameters are tuned
    (nrestarts random restarts, mkgp's native LML maximization). Otherwise the
    GP predicts at the given (or default start) hyperparameters with no
    optimization.

    The restarts are seeded from the fit's own input data (deterministic_seed),
    so the result only depends on (data_X, data_y, err_y),
    never on multiprocessing scheduling or slice processing order.
    When optimizing, a fit that pins a hyperparameter at its bound
    (pinned_hyperparams) is retried from a fresh, differently-seeded restart
    set up to hyp_retries times. The attempt with the best log marginal
    likelihood is kept even if every attempt stays pinned
    (an unresolvable slice should still return its least-bad fit).

    Args:
        data_X: Channel rho positions (NaN-free).
        data_y: Channel values (normalized when scale_per_slice).
        err_y: Channel errors.
        x_eval: Points to predict at.
        fit_bounds: The variable's staged bound knobs (l1 floor, x0 lower
            bound).
        anchors: The variable's anchors, normalized like data_y (FitAnchors.scaled).
        hyperparams: Fixed [var, l1, l2, lw, x0] to predict at, or None.
        optimize: Tune the hyperparameters (only when hyperparams is None).
        pin_x0: Narrow the x0 (pedestal location) bounds to a tight window
            around this value, so bound enforcement holds the pedestal there
            (used to tie the Te pedestal location to the ne fit).
        extra_grad_bc: (n, 3) rows appended to the grad anchors (the
            monotonic-edge virtual observations, see MONO_CHECK_RHO).
        nrestarts: Optimizer random restarts per attempt.
        hyp_retries: Extra attempts when a fit pins a hyperparameter.
        seed_salt: Offsets the deterministic restart seeds, so a caller can
            rerun the same data with a fresh, still reproducible restart draw
            (the worker's reseeded repair of a nonphysical fit).

    Returns:
        The fitted GaussianProcess, or None if every attempt failed.
    """
    kbounds = bounds_for(fit_bounds)
    if pin_x0 is not None:
        lo, hi = kbounds[0, 4], kbounds[1, 4]
        kbounds[0, 4] = max(lo, pin_x0 - X0_PIN_HALFWIDTH)
        kbounds[1, 4] = min(hi, pin_x0 + X0_PIN_HALFWIDTH)
    xdata = np.concatenate([data_X, anchors.value[:, 0]])
    ydata = np.concatenate([data_y, anchors.value[:, 1]])
    yerr = np.concatenate([err_y, anchors.value[:, 2]])
    grad_bc = (
        anchors.grad
        if extra_grad_bc is None
        else np.vstack([anchors.grad, extra_grad_bc])
    )

    do_optimize = optimize and hyperparams is None
    n_attempts = (1 + hyp_retries) if do_optimize else 1

    best_gp, best_lml = None, -np.inf
    for attempt in range(n_attempts):
        gp = GaussianProcess()
        gp.set_kernel(kernel=build_kernel(hyperparams), kbounds=kbounds, regpar=1.0)
        # Heteroscedastic noise model: GP-fit the error bars with an SE kernel
        # (mkgp HSGP path). The main fit then uses the smoothed errors and the
        # predictive std picks up a rho-varying noise term (see ERR_HYP_START).
        err_kernel = SE_Kernel(*ERR_HYP_START)
        err_kernel.enforce_bounds(True)
        gp.set_error_kernel(
            kernel=err_kernel,
            kbounds=ERR_HYP_BOUNDS,
            regpar=1.0,
            nrestarts=ERR_NRESTARTS,
        )
        gp.set_error_search_parameters(epsilon=1.0e-2)
        gp.set_raw_data(
            xdata=xdata,
            ydata=ydata,
            yerr=yerr,
            dxdata=grad_bc[:, 0],
            dydata=grad_bc[:, 1],
            dyerr=grad_bc[:, 2],
        )
        gp.set_search_parameters(epsilon=1.0e-2)
        # Seed even on the predict-only path: the error-kernel fit inside
        # GPRFit runs its own random restarts, so an unseeded RNG would make
        # the result depend on process history (serial vs parallel workers).
        np.random.seed(
            deterministic_seed(data_X, data_y, err_y, salt=attempt + 17 * seed_salt)
        )
        if do_optimize:
            fit_restarts = nrestarts
        else:
            # predict-only at fixed hyperparameters
            # The public maxiter clamps to >=50, so poke _imax=0 to skip the gradient-ascent loop entirely.
            gp._imax = 0
            fit_restarts = 0
        try:
            # mkgp prints optimizer status to stdout; keep worker logs clean.
            with contextlib.redirect_stdout(io.StringIO()):
                gp.GPRFit(
                    np.asarray(x_eval, dtype=float),
                    hsgp_flag=True,
                    nrestarts=fit_restarts,
                )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        if not do_optimize:
            return gp

        hyps = np.asarray(gp.get_gp_kernel_details()[1], dtype=float)
        lml = gp.get_gp_lml()
        if lml is not None and lml > best_lml:
            best_gp, best_lml = gp, lml
        if not pinned_hyperparams(hyps, kbounds):
            return gp  # converged inside the physical range, no need to retry

    return best_gp


def fit_profile(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
    x_star: np.ndarray,
    fit_bounds: FitBounds,
    anchors: FitAnchors,
    pin_x0: float | None = None,
    seed_salt: int = 0,
) -> ProfileFit | None:
    """Fit one cleaned profile slice and predict on x_star.

    Expects data that already went through cleaning.clean_channels
    (NaN-free, outliers removed, normalized when scale_per_slice).
    the hyperparameters are always optimized. The fitted hyperparameters
    are returned so callers can read the pedestal location (x0).

    After the fit, positive posterior gradients on the edge check grid are
    suppressed by virtual zero-slope observations and a refit at fixed
    hyperparameters (see MONO_CHECK_RHO in quality.py).
    The returned hyperparameters are always the original fit's,
    since the refit runs at fixed hyperparameters.

    Args:
        data_X: Channel rho positions.
        data_y: Channel values.
        err_y: Channel errors.
        x_star: Target rho grid.
        fit_bounds: The variable's staged bound knobs.
        anchors: The variable's anchors, normalized like data_y.
        pin_x0: Hold the pedestal location at this value (see run_gp).
        seed_salt: Restart seed offset for a reseeded retry (see run_gp).

    Returns:
        The fitted slice, or None if the GP fit failed.
    """
    # Predict on x_star plus the edge check grid in one pass, so the
    # monotonicity check below reads the posterior gradient without a second
    # GPRFit. The check points are sliced off before returning.
    x_out = np.asarray(x_star, dtype=float).ravel()
    n_out = x_out.size
    x_eval = np.concatenate([x_out, MONO_CHECK_RHO])
    gp = run_gp(
        data_X,
        data_y,
        err_y,
        x_eval,
        fit_bounds,
        anchors,
        pin_x0=pin_x0,
        seed_salt=seed_salt,
    )
    if gp is None:
        return None
    hyps_out = np.asarray(gp.get_gp_kernel_details()[1], dtype=float)

    # Monotonic-edge repair (see the MONO_CHECK_RHO block comment in
    # quality.py): pin the slope to zero wherever the fit rises on the check
    # grid and refit at the fitted hyperparameters, so the pedestal-top bump
    # flattens while everything the constraint does not touch stays put. A
    # failed refit keeps the unconstrained fit rather than losing the slice.
    mono_bc = np.empty((0, 3))
    for _ in range(MONO_MAX_PASSES):
        drv_check = np.asarray(gp.get_gp_drv_mean(), dtype=float).ravel()[n_out:]
        viol = np.isfinite(drv_check) & (drv_check > MONO_GRAD_TOL)
        new_rho = np.setdiff1d(MONO_CHECK_RHO[viol], mono_bc[:, 0])
        # Leave data-supported rises alone: constraining a genuine
        # hollow-profile flank biases the whole fit.
        new_rho = np.array(
            [r for r in new_rho if not rise_is_data_supported(data_X, data_y, err_y, r)]
        )
        if new_rho.size == 0:
            break
        new_rows = np.column_stack(
            [new_rho, np.zeros_like(new_rho), np.full_like(new_rho, MONO_GRAD_ERR)]
        )
        mono_bc = np.vstack([mono_bc, new_rows])
        gp_mono = run_gp(
            data_X,
            data_y,
            err_y,
            x_eval,
            fit_bounds,
            anchors,
            hyperparams=hyps_out,
            optimize=False,
            pin_x0=pin_x0,
            extra_grad_bc=mono_bc,
            seed_salt=seed_salt,
        )
        if gp_mono is None:
            break
        gp = gp_mono

    # Te/ne are physical (positive) quantities but the GP posterior is
    # Gaussian with unbounded support, so the mean can dip slightly negative
    # past the separatrix where the value anchors pull it to zero.
    # Clip the mean at 0 (downstream should read the band as truncated at 0 likewise)
    fit = np.maximum(np.asarray(gp.get_gp_mean(), dtype=float).ravel()[:n_out], 0.0)
    # Predictive std (includes observation noise), not the latent-function std.
    # With few, high-error core channels the latent band collapses to a
    # misleadingly tight interval - it conditions on the fitted amplitude
    # being exactly right and ignores the measurement scatter.
    # noise_flag=True widens the band where the data is noisy.
    # the noise term is rho-varying because run_gp fits an error kernel (HSGP),
    # so the band tracks the local error bars instead of a constant RMS.
    # The derivative std stays latent (the gradient is never directly
    # observed, so folding in point noise there is not meaningful).
    std = np.asarray(gp.get_gp_std(noise_flag=True), dtype=float).ravel()[:n_out]
    grad = np.asarray(gp.get_gp_drv_mean(), dtype=float).ravel()[:n_out]
    grad_std = np.asarray(gp.get_gp_drv_std(noise_flag=False), dtype=float).ravel()[
        :n_out
    ]
    return ProfileFit(fit=fit, std=std, grad=grad, grad_std=grad_std, hyps=hyps_out)
