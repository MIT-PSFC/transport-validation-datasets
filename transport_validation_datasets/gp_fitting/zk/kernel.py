"""mkgp kernel construction for the zk method.

The kernel is a Gibbs kernel whose length scale follows a tanh warp: long in
the core, short across the pedestal, with the transition center x0 tracking
the pedestal location. mkgp ships no tanh warp, so Tanh_WarpingFunction adds
one, with analytic derivatives so mkgp's analytic LML-gradient optimizer path
stays valid.
"""

import hashlib

import numpy as np
from mkgp.core.baseclasses import _WarpingFunction
from mkgp.core.kernels import Gibbs_Kernel

from transport_validation_datasets.gp_fitting.batch_io import FitBounds

# Hyperparameters, order [var, l1, l2, lw, x0]:
# amplitude
# core (small-rho) length scale,
# edge (large-rho) length scale,
# tanh transition width
# and transition center.
HYP_START = np.array([2.0, 0.8, 0.4, 0.1, 1.0])
# Bounds define the optimizer's random-restart ranges (drawn uniform in log10)
HYP_BOUNDS = np.array([[1.0e-2, 0.4, 0.2, 0.05, 0.95], [2.0e1, 0.9, 0.5, 0.2, 1.05]])
# The x0 lower bound is device-dependent (FitBounds.x0_min, staged per batch):
# edge-peaked MAST ne (early-time edge accumulation, e.g. 28978 t=0.26) has
# its sharp structure at rho 0.85-0.95, and with the transition center held
# at 0.95+ only the long core length scale covers that region, flattening a
# data-supported edge peak into a shelf - MAST runs 0.85. C-Mod keeps the
# default 0.95: its pedestals live at 0.95-1.05 and widening the restart
# range there just dilutes the draws into late-pedestal basins (round-5
# sample validation, 2026-07-30 stupid_fits audit).
# The l1 (core length scale) floor is likewise device-dependent
# (FitBounds.l1_min): 0.4 keeps a stiff core that cannot chase channel
# scatter, but is too stiff to bend down-up-down through a hollow MAST ne
# profile - the fit rounds off the off-axis crest. MAST runs 0.2 (crest bias
# gone, monotonic control fits unchanged, icddps2 audit 2026-07); C-Mod runs
# 0.35 for te and 0.55 for ne.

# Error kernel (heteroscedastic noise model): a squared-exponential GP is fit
# to the input error bars themselves (mkgp's HSGP path, make_HSGP_errors).
# This does two things: the main fit sees smoothed error bars instead of raw
# ones, and predictions get a rho-varying noise estimate, so the reported
# predictive std widens where the data is genuinely noisy (sparse fat-error
# core) and narrows across dense precise channels - instead of the constant
# RMS-of-errors band mkgp falls back to without an error kernel. Hyps:
# [amplitude, length scale] on scale_per_slice-normalized data (errors are
# O(0.01-0.3)). Length scale floor 0.2 keeps the noise model a smooth radial
# trend rather than chasing individual channels' error bars.
ERR_HYP_START = np.array([0.1, 0.5])
ERR_HYP_BOUNDS = np.array([[1.0e-3, 0.2], [1.0, 1.5]])
ERR_NRESTARTS = 2


class Tanh_WarpingFunction(_WarpingFunction):
    """tanh length-scale warp for the mkgp Gibbs kernel.

    l(z) = 0.5 * ((l1 + l2) - (l1 - l2) * tanh((z - x0) / lw))

    mkgp ships only Constant/Linear/IG warps, so this adds the tanh length
    scale. hyps = [l1, l2, lw, x0]. Analytic z- and hyperparameter-derivatives
    are provided (verified against finite differences) so mkgp's analytic
    LML-gradient optimizer path stays valid.
    """

    def __calc_warp(self, zz, der=0, hder=None):
        l1, l2, lw, x0 = self.hyperparameters
        u = (zz - x0) / lw
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
            elif hder == 3:
                warp = 0.5 * (l1 - l2) * ss / lw
        elif der == 1:
            if hder is None:
                warp = -0.5 * (l1 - l2) * ss / lw
            elif hder == 0:
                warp = -0.5 * ss / lw
            elif hder == 1:
                warp = 0.5 * ss / lw
            elif hder == 2:
                warp = -0.5 * (l1 - l2) * ss / (lw * lw) * (2.0 * tt * u - 1.0)
            elif hder == 3:
                warp = -0.5 * (l1 - l2) * 2.0 * tt * ss / (lw * lw)
        return warp

    def __init__(self, l1=1.0, l2=0.5, lw=0.1, x0=1.0, dtype=None):
        """Build the warp at the given hyperparameters.

        Args:
            l1: Core (small-rho) length scale.
            l2: Edge (large-rho) length scale.
            lw: tanh transition width.
            x0: Transition center (pedestal location).
            dtype: Optional numpy dtype for evaluations.
        """
        hyps = np.array([float(l1), float(l2), float(lw), float(x0)])
        super().__init__("Wtanh", self.__calc_warp, True, hyps, dtype=dtype)

    def __copy__(self):
        hyps = self.hyperparameters
        bnds = self.bounds
        kcopy = Tanh_WarpingFunction(
            hyps[0], hyps[1], hyps[2], hyps[3], dtype=self._dtype
        )
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy


def build_kernel(hyperparams: np.ndarray | None = None) -> Gibbs_Kernel:
    """Build the Gibbs kernel with the tanh warp.

    Bound enforcement is turned on for both the kernel and its warp. mkgp's
    gradient-ascent optimizer never clamps to kbounds, so without this the
    hyperparameters can wander out of the physical region into the degenerate
    "all noise" fit (amplitude -> 0, edge length scale -> inf, profile pulled
    to ~0). Enforcement also lets run_gp pin the pedestal location by
    narrowing the x0 bounds. set_kernel/__copy__ both preserve the enforce
    flag.

    Args:
        hyperparams: [var, l1, l2, lw, x0] to build at; None uses HYP_START.

    Returns:
        The kernel, ready for GaussianProcess.set_kernel.
    """
    hyps = HYP_START if hyperparams is None else np.asarray(hyperparams, dtype=float)
    kernel = Gibbs_Kernel(hyps[0], wfunc=Tanh_WarpingFunction(*hyps[1:]))
    kernel.enforce_bounds(True)
    kernel._wfunc.enforce_bounds(True)
    return kernel


def bounds_for(fit_bounds: FitBounds) -> np.ndarray:
    """Build the optimizer bounds with the per-variable knobs applied.

    Args:
        fit_bounds: The variable's staged bound knobs.

    Returns:
        (2, 5) array of [lower, upper] hyperparameter bounds.
    """
    bounds = HYP_BOUNDS.astype(float).copy()
    bounds[0, 1] = float(fit_bounds.l1_min)
    bounds[0, 4] = float(fit_bounds.x0_min)
    return bounds


def is_pedestal_resolved(x0: float, x0_min: float) -> bool:
    """Check that a fitted pedestal location sits inside the x0 bounds.

    An x0 pinned at a bound means the optimizer found no clear pedestal in
    range, so it should not be trusted to drive the other profile's location.

    Args:
        x0: Fitted transition center.
        x0_min: The variable's x0 lower bound (FitBounds.x0_min).

    Returns:
        True if x0 sits inside (not pushed to) the bounds.
    """
    lo = float(x0_min)
    hi = HYP_BOUNDS[1, 4]
    margin = 0.02 * (hi - lo)
    return lo + margin < x0 < hi - margin


def pinned_hyperparams(hyps: np.ndarray, kbounds: np.ndarray) -> bool:
    """Check whether the optimizer pushed a hyperparameter to its bound.

    Flags only pins a differently-seeded restart could plausibly escape.
    Bound enforcement (build_kernel) keeps a bad restart out of the degenerate
    collapse mkgp is prone to (amplitude -> 0, edge scale -> infinity). A
    hyperparameter still sitting at that bound after optimization means the
    search ran out of room rather than converging, so run_gp retries from a
    different restart.

    Two edges are excluded because a retry provably re-lands on them (measured
    on a real shot: x0 pinned in 10/10 sampled slices, l2's ceiling in half,
    every retry re-landing, tripling fit time for no change):
    - x0 (pedestal location): its bounds are tight by design, not slack.
    - l2's ceiling: with x0 held near the edge there is often no short-scale
      structure left beyond it, so a long, smooth l2 is the right answer.

    The margin is measured in log10 space, matching how restarts are drawn.
    var and lw span 2-3 decades, so a fraction of the raw range is huge in log
    terms and would flag converged interior optima as pinned.

    Args:
        hyps: Fitted [var, l1, l2, lw, x0].
        kbounds: (2, 5) bounds the fit ran under.

    Returns:
        True if any escapable hyperparameter sits at its bound.
    """
    lo, hi = kbounds[0], kbounds[1]
    log_lo, log_hi, log_hyps = np.log10(lo), np.log10(hi), np.log10(hyps)
    margin = 0.02 * (log_hi - log_lo)
    pinned_lo = log_hyps <= log_lo + margin
    pinned_hi = log_hyps >= log_hi - margin
    pinned_lo[4] = pinned_hi[4] = False  # x0
    pinned_hi[2] = False  # l2 ceiling
    return bool((pinned_lo | pinned_hi).any())


def deterministic_seed(*arrays: np.ndarray, salt: int = 0) -> int:
    """Derive a stable RNG seed from the fit's own input data, not call order.

    mkgp draws its optimizer restarts from the global numpy RNG (see
    GaussianProcess.GPRFit), so without reseeding a fit's result depends on
    whatever else already consumed random draws earlier in the process: slice
    processing order in serial mode, or multiprocessing.Pool scheduling and
    fork-inherited RNG state in parallel mode. Hashing the fit's own inputs
    makes every fit reproducible regardless of how the batch happens to be
    scheduled.

    Args:
        *arrays: The fit's input arrays; hashed as float64 bytes.
        salt: Distinguishes retry attempts on the same input.

    Returns:
        32-bit seed for np.random.seed.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(int(salt).to_bytes(8, "little", signed=True))
    return int(h.hexdigest()[:8], 16)
