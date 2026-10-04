"""mkgp kernel construction for the zk method.

The kernel is a Gibbs kernel whose length scale follows a tanh warp,
long in the core and short across the pedestal.
The transition is centred on the configured pedestal location (DeviceSettings.pedestal_rho_tor_norm),
a constant of the warp, never optimized.
mkgp ships no tanh warp, so Tanh_WarpingFunction adds one,
with analytic derivatives so mkgp's analytic LML-gradient optimizer path stays valid.
"""

import hashlib

import numpy as np
from mkgp.core.baseclasses import _WarpingFunction
from mkgp.core.kernels import Gibbs_Kernel

from transport_validation_datasets.gp_fitting.batch_io import FitBounds

# Hyperparameters, order [var, l1, l2, lw]:
# amplitude,
# core (small-rho) length scale,
# edge (large-rho) length scale,
# and tanh transition width.
# The likelihood often has two basins: a long core scale and a short core scale
# Ensure each basin is tried at least once to find the best.
HYP_START = np.array([2.0, 0.8, 0.4, 0.1])
HYP_START_SHORT_CORE = np.array([2.0, 0.3, 0.5, 0.2])
# Bounds define the optimizer's random-restart ranges (drawn uniform in log10)
# bounds_for replaces the var bounds and the l1 bounds with the variable's FitBounds.
HYP_BOUNDS = np.array([[1.0e-2, 0.4, 0.05, 0.05], [2.0e1, 0.7, 0.5, 0.2]])
# A hyperparameter within this fraction of its log10 range of a bound counts as pinned (pinned_hyperparams).
# 2 percent of the range tells a bound-hugging fit from an interior optimum
# that happens to converge near a bound.
PINNED_MARGIN_LOG10_FRACTION = 0.02


class Tanh_WarpingFunction(_WarpingFunction):
    """tanh length-scale warp for the mkgp Gibbs kernel.

    l(z) = 0.5 * ((l1 + l2) - (l1 - l2) * tanh((z - pedestal_rho) / lw))

    mkgp ships only Constant/Linear/IG warps, so this adds the tanh length scale.
    The hyperparameters are [l1, l2, lw], and pedestal_rho is an mkgp constant the optimizer never moves.
    Analytic z- and hyperparameter-derivatives are provided (verified against finite differences)
    so mkgp's analytic LML-gradient optimizer path stays valid.
    """

    def __calc_warp(self, zz, der=0, hder=None):
        l1, l2, lw = self.hyperparameters
        (pedestal_rho,) = self.constants
        u = (zz - pedestal_rho) / lw
        tt = np.tanh(u)
        ss = 1.0 - tt * tt
        warp = np.zeros(np.shape(zz), dtype=self._dtype)
        if der == 0:
            if hder is None:
                warp = 0.5 * ((l1 + l2) - (l1 - l2) * tt)
            elif hder == 0:
                warp = 0.5 * (1.0 - tt)
            elif hder == 1:
                warp = 0.5 * (1.0 + tt)
            elif hder == 2:
                warp = 0.5 * (l1 - l2) * ss * u / lw
        elif der == 1:
            if hder is None:
                warp = -0.5 * (l1 - l2) * ss / lw
            elif hder == 0:
                warp = -0.5 * ss / lw
            elif hder == 1:
                warp = 0.5 * ss / lw
            elif hder == 2:
                warp = -0.5 * (l1 - l2) * ss / (lw * lw) * (2.0 * tt * u - 1.0)
        return warp

    def __init__(self, pedestal_rho, l1=1.0, l2=0.5, lw=0.1, dtype=None):
        """Build the warp at the given hyperparameters.

        Args:
            pedestal_rho: Transition center, the pedestal location. A constant, not a hyperparameter.
            l1: Core (small-rho) length scale.
            l2: Edge (large-rho) length scale.
            lw: tanh transition width.
            dtype: Optional numpy dtype for evaluations.
        """
        hyps = np.array([float(l1), float(l2), float(lw)])
        csts = np.array([float(pedestal_rho)])
        super().__init__("Wtanh", self.__calc_warp, True, hyps, csts, dtype=dtype)

    def __copy__(self):
        hyps = self.hyperparameters
        (pedestal_rho,) = self.constants
        bnds = self.bounds
        kcopy = Tanh_WarpingFunction(
            pedestal_rho, hyps[0], hyps[1], hyps[2], dtype=self._dtype
        )
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy


def build_kernel(
    pedestal_rho: float, hyperparams: np.ndarray | None = None
) -> Gibbs_Kernel:
    """Build the Gibbs kernel with the tanh warp.

    Bound enforcement is turned on for both the kernel and its warp.
    mkgp's gradient-ascent optimizer never clamps to kbounds, so without this the
    hyperparameters can become a degenerate "all noise" fit
    (amplitude -> 0, edge length scale -> inf, profile pulled to ~0).
    set_kernel/__copy__ both preserve the enforce flag.

    Args:
        pedestal_rho: The warp's transition center.
        hyperparams: [var, l1, l2, lw] to build at, None uses HYP_START.

    Returns:
        The kernel, ready for GaussianProcess.set_kernel.
    """
    hyps = HYP_START if hyperparams is None else np.asarray(hyperparams, dtype=float)
    warp = Tanh_WarpingFunction(pedestal_rho, *hyps[1:])
    kernel = Gibbs_Kernel(hyps[0], wfunc=warp)
    kernel.enforce_bounds(True)
    kernel._wfunc.enforce_bounds(True)
    return kernel


def bounds_for(fit_bounds: FitBounds) -> np.ndarray:
    """Build the optimizer bounds with the per-variable knobs applied.

    Args:
        fit_bounds: The variable's staged bound knobs.

    Returns:
        (2, 4) array of [lower, upper] hyperparameter bounds.
    """
    bounds = HYP_BOUNDS.astype(float).copy()
    bounds[0, 0] = float(fit_bounds.var_min)
    bounds[1, 0] = float(fit_bounds.var_max)
    bounds[0, 1] = float(fit_bounds.l1_min)
    bounds[1, 1] = float(fit_bounds.l1_max)
    return bounds


def pinned_hyperparams(hyps: np.ndarray, kbounds: np.ndarray) -> bool:
    """Check whether the optimizer pushed a hyperparameter to its bound.

    Flags only pins a differently-seeded restart could plausibly escape.
    Bound enforcement (build_kernel) keeps a bad restart out of the degenerate
    collapse mkgp is prone to (amplitude -> 0, edge scale -> infinity).
    A hyperparameter still sitting at that bound after optimization means the
    search ran out of room rather than converging, so run_gp retries from a
    different restart.

    The length-scale ceilings are excluded, because a retry typically re-lands on them.
    They are regularization (see HYP_BOUNDS), and most fits rest on them.

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
        *arrays: The fit's input arrays; hashed as float64 bytes.
        salt: Distinguishes retry attempts on the same input.

    Returns:
        32-bit seed for np.random.seed

    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(int(salt).to_bytes(8, "little", signed=True))
    return int(h.hexdigest()[:8], 16)
