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
from mkgp.core.routines import GaussianProcess

from transport_validation_datasets.gp_fitting.batch_io import FitAnchors, FitBounds
from transport_validation_datasets.gp_fitting.zk.kernel import (
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
        std: (n_x,) predictive std, the latent std and the interpolated channel errors in quadrature.
        grad: (n_x,) posterior derivative d/drho, 0 where fit is clipped.
        grad_std: (n_x,) latent derivative std.
        hyps: Fitted [var, l1, l2, lw].
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
    pedestal_rho: float,
    hyperparams=None,
    optimize=True,
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
        fit_bounds: The variable's staged bound knobs (amplitude ceiling, l1 floor).
        anchors: The variable's anchors, normalized like data_y (FitAnchors.scaled).
        pedestal_rho: The kernel's length-scale transition center.
        hyperparams: Fixed [var, l1, l2, lw] to predict at, or None.
        optimize: Tune the hyperparameters (only when hyperparams is None).
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
        kernel = build_kernel(pedestal_rho, hyperparams)
        gp.set_kernel(kernel=kernel, kbounds=kbounds, regpar=1.0)
        gp.set_raw_data(
            xdata=xdata,
            ydata=ydata,
            yerr=yerr,
            dxdata=grad_bc[:, 0],
            dydata=grad_bc[:, 1],
            dyerr=grad_bc[:, 2],
        )
        gp.set_search_parameters(epsilon=1.0e-2)
        # Seed the random restarts from the fit's own inputs,
        # so the result never depends on process history (serial vs parallel workers).
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
            # No heteroscedastic error model (hsgp_flag), every channel is fit with its own error.
            # mkgp's error kernel replaces the error bars with a smooth curve in rho.
            # Where the errors vary channel to channel, as across a C-Mod pedestal,
            # the curve is off by 0.5x to 4x either way, differently from slice to slice,
            # so the fit chased some pedestals and ignored others (chi2 > 4 in 12 percent of probe slices, 1 percent without it).
            # Under a flat error floor the curve equals the errors and the model is inert.
            with contextlib.redirect_stdout(io.StringIO()):
                gp.GPRFit(
                    np.asarray(x_eval, dtype=float),
                    hsgp_flag=False,
                    nrestarts=fit_restarts,
                )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        if not do_optimize:
            return gp

        hyps = np.asarray(gp.get_gp_kernel().hyperparameters, dtype=float)
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
    pedestal_rho: float,
    seed_salt: int = 0,
) -> ProfileFit | None:
    """Fit one cleaned profile slice and predict on x_star.

    Expects data that already went through cleaning.clean_channels
    (NaN-free, outliers removed, normalized when scale_per_slice).
    The hyperparameters are always optimized.

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
        pedestal_rho: The kernel's length-scale transition center.
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
        pedestal_rho,
        seed_salt=seed_salt,
    )
    if gp is None:
        return None
    hyps_out = np.asarray(gp.get_gp_kernel().hyperparameters, dtype=float)

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
            pedestal_rho,
            hyperparams=hyps_out,
            optimize=False,
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
    mean = np.asarray(gp.get_gp_mean(), dtype=float).ravel()[:n_out]
    fit = np.maximum(mean, 0.0)
    # Predictive std (includes observation noise), not the latent-function std.
    # With few, high-error core channels the latent band collapses to a
    # misleadingly tight interval - it conditions on the fitted amplitude
    # being exactly right and ignores the measurement scatter.
    # The noise is the channel errors interpolated in rho, held flat past the first and last channel,
    # so the band tracks the local error bars.
    # mkgp's own noise term (noise_flag=True) is one RMS over every error when there is no error kernel,
    # 3-10x the edge errors on C-Mod Te.
    # The derivative std stays latent (the gradient is never directly
    # observed, so folding in point noise there is not meaningful).
    order = np.argsort(data_X)
    noise = np.interp(x_out, data_X[order], err_y[order])
    latent_std_eval = gp.get_gp_std(noise_flag=False)
    latent_std = np.asarray(latent_std_eval, dtype=float).ravel()[:n_out]
    std = np.sqrt(latent_std**2 + noise**2)
    # The gradient of the clipped profile, 0 wherever the clip holds the mean at 0.
    # The unclipped mean climbs back up from its dip to the anchors,
    # and that rise would otherwise show as a positive gradient under a flat zero profile.
    grad_mean = np.asarray(gp.get_gp_drv_mean(), dtype=float).ravel()[:n_out]
    grad = np.where(mean > 0.0, grad_mean, 0.0)
    grad_std = np.asarray(gp.get_gp_drv_std(noise_flag=False), dtype=float).ravel()[
        :n_out
    ]
    return ProfileFit(fit=fit, std=std, grad=grad, grad_std=grad_std, hyps=hyps_out)
