"""mkgp kernel construction for the zk method.

The kernel is a Gibbs kernel whose length scale follows a tanh warp,
long in the core and short across the pedestal.
The transition is centred on the configured pedestal location (DeviceSettings.pedestal_rho_tor_norm),
a constant of the warp, never optimized.
"""

import hashlib

import numpy as np
from mkgp.core.kernels import Gibbs_Kernel, Tanh_WarpingFunction

from transport_validation_datasets.gp_fitting.batch_io import FitBounds

# Hyperparameters, order [var, l1, l2, lw]:
# amplitude,
# core (small-rho) length scale,
# edge (large-rho) length scale,
# and tanh transition width.
# The likelihood often has two basins, a long core scale and a short one,
# so the optimizer starts once from each and keeps the better one.
# mkgp clips a start outside the bounds (bounds_for) onto them.
HYP_START = np.array([2.0, 0.8, 0.4, 0.1])
HYP_START_SHORT_CORE = np.array([2.0, 0.3, 0.5, 0.2])
# A hyperparameter within this fraction of its log10 range of a bound counts as pinned (pinned_hyperparams).
# 2 percent of the range tells a bound-hugging fit from an interior optimum
# that happens to converge near a bound.
PINNED_MARGIN_LOG10_FRACTION = 0.02


def build_kernel(
    pedestal_rho: float, hyperparams: np.ndarray | None = None
) -> Gibbs_Kernel:
    """Build the Gibbs kernel with mkgp's tanh warp.

    Bound enforcement is turned on for both the kernel and its warp.
    mkgp's gradient-ascent optimizer never clamps to kbounds, so without this the
    hyperparameters can become a degenerate "all noise" fit
    (amplitude -> 0, edge length scale -> inf, profile pulled to ~0).
    The kernel copies the warp with its enforce flag, and set_kernel copies the kernel the same way.

    Args:
        pedestal_rho: The warp's transition center.
        hyperparams: [var, l1, l2, lw] to build at, None uses HYP_START.

    Returns:
        The kernel, ready for GaussianProcess.set_kernel.
    """
    hyps = HYP_START if hyperparams is None else np.asarray(hyperparams, dtype=float)
    var, l1, l2, lw = hyps
    warp = Tanh_WarpingFunction(l1, l2, lw, x0=pedestal_rho)
    warp.enforce_bounds(True)
    kernel = Gibbs_Kernel(var, wfunc=warp)
    kernel.enforce_bounds(True)
    return kernel


def bounds_for(fit_bounds: FitBounds) -> np.ndarray:
    """Build the optimizer bounds of [var, l1, l2, lw].

    var and l1 take the variable's staged knobs.
    l2 runs 0.05-0.5 and lw 0.05-0.2 for every variable.
    The optimizer draws its random restarts uniform in log10 between the bounds.

    Args:
        fit_bounds: The variable's staged bound knobs.

    Returns:
        (2, 4) array of [lower, upper] hyperparameter bounds.
    """
    lower = [fit_bounds.var_min, fit_bounds.l1_min, 0.05, 0.05]
    upper = [fit_bounds.var_max, fit_bounds.l1_max, 0.5, 0.2]
    return np.array([lower, upper], dtype=float)


def pinned_hyperparams(hyps: np.ndarray, kbounds: np.ndarray) -> bool:
    """Check whether the optimizer pushed a hyperparameter to its bound.

    Flags only pins a differently-seeded restart could plausibly escape.
    Bound enforcement (build_kernel) keeps a bad restart out of the degenerate
    collapse mkgp is prone to (amplitude -> 0, edge scale -> infinity).
    A hyperparameter still sitting at that bound after optimization means the
    search ran out of room rather than converging, so run_gp retries from a
    different restart.

    The length-scale ceilings are excluded, because a retry typically re-lands on them.
    They are regularization (see bounds_for), and most fits rest on them.

    The margin (PINNED_MARGIN_LOG10_FRACTION) is measured in log10 space, matching how restarts are drawn.
    var and lw span 2-3 decades, so a fraction of the raw range is huge in log
    terms and would flag converged interior optima as pinned.

    Args:
        hyps: Fitted [var, l1, l2, lw].
        kbounds: (2, 4) bounds the fit ran under.

    Returns:
        True if any escapable hyperparameter sits at its bound.
    """
    lo, hi = kbounds[0], kbounds[1]
    log_lo, log_hi, log_hyps = np.log10(lo), np.log10(hi), np.log10(hyps)
    margin = PINNED_MARGIN_LOG10_FRACTION * (log_hi - log_lo)
    pinned_lo = log_hyps <= log_lo + margin
    pinned_hi = log_hyps >= log_hi - margin
    pinned_hi[1:] = False  # length-scale ceilings
    return bool((pinned_lo | pinned_hi).any())


def deterministic_seed(*arrays: np.ndarray, salt: int = 0) -> int:
    """Derive a stable RNG seed from the fit's own input data.

    Hashing the fit's own inputs makes every fit reproducible
    regardless of how the batch happens to be scheduled.

    Args:
        *arrays: The fit's input arrays, hashed as float64 bytes.
        salt: Distinguishes retry attempts on the same input.

    Returns:
        32-bit seed for np.random.seed

    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(int(salt).to_bytes(8, "little", signed=True))
    return int(h.hexdigest()[:8], 16)
