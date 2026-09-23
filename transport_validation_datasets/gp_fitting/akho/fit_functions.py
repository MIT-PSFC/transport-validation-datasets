"""Vendored analytic mtanh/cubic pre-fit stage of the akho method.

Vendored (a second time) from `cmod_to_imas/profile_fit_vendored.py`, which
itself was vendored from the small `tokamak_profile_fitting` slice
`fit_cmod.py` actually needed (`fit.py`, `profiles/fit_functions.py`,
`utils.py`) -- that package is not pip-installable and only reachable via a
`sys.path` hack into a sibling repo, so vendoring avoids that fragile
dependency entirely (see `profile_fit_vendored.py`'s own docstring for the
original provenance). `apply_2pt_shift` was dropped: it implements the
two-point-model Te-separatrix shift, which needs an external per-shot
calibration target that `FitBatch`/`ShotFitInput` do not carry (see
worker_akho.py's module docstring) -- unused here, so not vendored.

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
"""

import numpy as np
from scipy.optimize import curve_fit

# Te mtanh fits wider than this are broad core-shaping sigmoids pinned near
# the width bound, not pedestals, so the cubic takes those slices.
_TE_MTANH_MAX_WIDTH = 0.10


def chi_squared(ydata, yfit, yerr):
    """Compute the error-weighted chi-squared of a fit.

    Returns:
        The scalar chi-squared value.
    """
    return np.sum((ydata - yfit) ** 2 / yerr**2)


def reduced_chi_squared_inside_separatrix(rho, ydata, yfit, yerr, num_params):
    """Chi-squared evaluated only inside the separatrix (rho < 1).

    Returns:
        The reduced chi-squared over the masked points,
        or np.inf if there are not enough points to constrain the fit.
    """
    mask = rho < 1.0
    n = np.sum(mask)
    if n <= num_params:
        return np.inf
    return chi_squared(ydata[mask], yfit[mask], yerr[mask]) / (n - num_params)


def Osborne_Tanh_cubic(x, c0, c1, c2, c3, c4, c5, c6):
    """Osborne tanh with linear, quadratic and cubic inboard terms and flat SOL.

    c0: pedestal centre, c1: full width, c2: top, c3: bottom,
    c4: linear, c5: quadratic, c6: cubic inboard terms.

    Returns:
        The function evaluated at `x`.
    """
    z = 2.0 * (c0 - x) / c1
    P1 = 1.0 + c4 * z + c5 * z**2 + c6 * z**3
    P2 = 1.0
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1 * E1 - P2 * E2) / (E1 + E2))


def Osborne_Tanh_cubic_zero_axis_slope(x, c0, c1, c2, c3, c5, c6):
    """Osborne tanh (cubic inboard, flat SOL) with zero slope at the axis.

    Same family as Osborne_Tanh_cubic, but the linear inboard term c4 is not
    free: it is eliminated in closed form so that df/dx == 0 holds exactly at
    x=0 -- the magnetic axis boundary condition CubicZeroAxisSlope enforces
    for the cubic fallback. Setting the axis derivative to zero at
    z0 = 2*c0/c1 gives a condition linear in c4,

        P1'(z0)*(1+u) + (2*P1(z0) + 2)*u = 0,    u = exp(-2*z0),

    solved for c4 below. For realistic pedestal parameters u is ~1e-6 or
    smaller, so this is effectively P1'(z0) = 0 (the inboard polynomial's
    slope in z vanishes at the axis).

    c0: pedestal centre, c1: full width, c2: top, c3: bottom,
    c5: quadratic, c6: cubic inboard terms.

    Returns:
        The function evaluated at `x`.
    """
    z0 = 2.0 * c0 / c1
    u = np.exp(-2.0 * z0)
    c4 = -(
        (2.0 * c5 * z0 + 3.0 * c6 * z0**2) * (1.0 + u)
        + (4.0 + 2.0 * c5 * z0**2 + 2.0 * c6 * z0**3) * u
    ) / (1.0 + u + 2.0 * z0 * u)
    return Osborne_Tanh_cubic(x, c0, c1, c2, c3, c4, c5, c6)


def CubicZeroAxisSlope(x, c0, c2, c3):
    """Cubic polynomial with its linear term dropped.

    f'(0) == 0 identically for any c2/c3, enforcing zero profile gradient
    at the magnetic axis (x=0) exactly, without a bounded/constrained
    optimizer.

    Returns:
        The polynomial evaluated at `x`.
    """
    return c0 + c2 * x**2 + c3 * x**3


def Osborne_linear_initial_guesses(rho_edge, values_edge, n_params=7):
    """Rough initial guesses for an Osborne tanh fit based on edge-only data.

    Returns:
        A list of length n_params with trailing zeros for polynomial terms.
    """
    if len(values_edge) < 4:
        return [1.0, 0.04, float(np.nanmax(values_edge)), 0.0] + [0.0] * (n_params - 4)

    avg = np.nanmean(values_edge)
    bottom = avg * 0.3
    top = avg * 1.1

    max_r = rho_edge[-1]
    min_r = rho_edge[-1]
    for i in range(len(rho_edge) - 1, -1, -1):
        if values_edge[i] > bottom:
            max_r = rho_edge[i]
            break
    for i in range(len(rho_edge) - 1, -1, -1):
        if values_edge[i] > top:
            min_r = rho_edge[i]
            break

    width = max_r - min_r
    if width <= 0:
        width = abs(rho_edge[-3] - rho_edge[-5]) if len(rho_edge) > 5 else 0.05
    centre = (max_r + min_r) / 2.0

    return [centre, width, top, bottom] + [0.0] * (n_params - 4)


def evaluate_with_gradient(fit_func, popt, x, h=1.0e-4):
    """Evaluate a vendored fit function and its central-difference derivative.

    Used to reconstruct the analytic mean fit's own contribution to the total
    profile gradient/value: the mkgp GP stage (gp.py) only fits the
    *residual* against this mean, so its own posterior mean/derivative do not
    include the mean's own value/slope -- worker_akho.py's `_fit_variable`
    stacks this on top of the GP-residual result. No closed-form derivative
    is used (rather than deriving one per fit function) since both functions
    `fit_analytic_profile` can pick between need one and a single
    finite-difference helper covers them.

    Args:
        fit_func: One of this module's Osborne_Tanh_*/CubicZeroAxisSlope functions.
        popt: Its fitted parameters (from fit_analytic_profile).
        x: Points to evaluate at.
        h: Central-difference step.

    Returns:
        (mean, dmean_dx), each shaped like `x`.
    """
    x = np.asarray(x, dtype=float)
    mean = fit_func(x, *popt)
    dmean_dx = (fit_func(x + h, *popt) - fit_func(x - h, *popt)) / (2.0 * h)
    return mean, dmean_dx


def fit_analytic_profile(rho, values, errors, is_channel, profile_type, edge_thresh):
    """Fit the zero-axis-slope mtanh and cubic to one normalized profile, keep the better.

    The candidates compete on reduced chi-squared inside the separatrix.
    An mtanh needs three measured channels inside its pedestal width,
    and a Te mtanh wider than _TE_MTANH_MAX_WIDTH is rejected.

    Args:
        rho: (n,) point positions, measured channels and value anchors.
        values: (n,) values normalized by the slice maximum.
        errors: (n,) errors, normalized the same way.
        is_channel: (n,) True for measured channels, False for anchors.
        profile_type: 'te' or 'ne', selects the bounds and initial guesses.
        edge_thresh: rho above which points seed the edge-based initial guess.

    Returns:
        (fit_func, popt) of the winning candidate, or None if both fail.
    """
    is_ne = profile_type == "ne"
    max_val = max(values)
    # Osborne_Tanh_cubic_zero_axis_slope's c0, c1, c2, c3, c5, c6
    n_params = 6

    # Bounds on [centre, width, top, bottom] + inboard polynomial terms
    if is_ne:
        lb = [0.85, 0.01, 0.0, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.25, max_val, np.inf] + [np.inf] * (n_params - 4)
    else:
        lb = [0.85, 0.01, 0.0, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.15, max_val, max_val] + [np.inf] * (n_params - 4)

    # Hard-coded initial guesses, [c0, c1, c2, c3, c5, c6]
    if is_ne:
        hardcoded = [
            # 650 kA C-Mod
            [
                1.00604712,
                3.7400836e-02,
                2.10662412,
                1.68897974e-02,
                2.29233952e-03,
                -2.0627212e-05,
            ],
            # 1 MA C-Mod
            [
                1.02123755e00,
                5.02744526e-02,
                2.54219267e00,
                -9.99999694e-04,
                -2.32961078e-03,
                4.20279037e-05,
            ],
            # D3D
            [0.99, 0.04, 1.0, 0.05, 0.0, 0.0],
            [0.99, 0.04, 0.3, 0.05, 0.0, 0.0],
        ]
    else:
        hardcoded = [
            [
                9.92614859e-01,
                4.01791101e-02,
                2.55550908e02,
                1.28542623e01,
                -3.45196862e-03,
                1.42947373e-04,
            ],
        ]
    guesses = [np.array(g) for g in hardcoded]
    # The edge-based guess sees the value anchors too, like the original's SOL zeros
    try:
        edge_sel = rho > edge_thresh
        auto = Osborne_linear_initial_guesses(rho[edge_sel], values[edge_sel], n_params)
        guesses.insert(0, np.array(auto))
    except Exception:
        pass

    # Clamp the guesses to the bounds, the hard-coded pedestal tops sit far above normalized data
    lb_arr = np.array(lb)
    ub_arr = np.array(ub)
    guesses = [np.clip(g, lb_arr, ub_arr) for g in guesses]

    # Data-adaptive guesses with the pedestal top near the data maximum go first
    for c0_guess, c1_guess in [(0.99, 0.04), (0.98, 0.04), (0.99, 0.05), (1.00, 0.03)]:
        g = np.zeros(n_params)
        g[0] = c0_guess
        g[1] = c1_guess
        g[2] = 0.85 * max_val
        g[3] = 0.02 * max_val
        guesses.insert(0, g)

    params_mtanh = chi_mtanh = None
    for guess in guesses:
        try:
            params_mtanh, _ = curve_fit(
                Osborne_Tanh_cubic_zero_axis_slope,
                rho,
                values,
                p0=guess,
                sigma=errors,
                absolute_sigma=True,
                maxfev=2000,
                bounds=(lb, ub),
            )
            mtanh_at_data = Osborne_Tanh_cubic_zero_axis_slope(rho, *params_mtanh)
            chi_mtanh = reduced_chi_squared_inside_separatrix(
                rho, values, mtanh_at_data, errors, n_params
            )
            break
        except Exception:
            continue

    params_cubic = chi_cubic = None
    try:
        params_cubic, _ = curve_fit(
            CubicZeroAxisSlope,
            rho,
            values,
            sigma=errors,
            absolute_sigma=True,
            maxfev=2000,
        )
        cubic_at_data = CubicZeroAxisSlope(rho, *params_cubic)
        chi_cubic = reduced_chi_squared_inside_separatrix(
            rho, values, cubic_at_data, errors, 3
        )
    except Exception:
        pass

    # Discard fits with a nonsensical chi-squared
    def _bad_chi(chi):
        return chi is None or chi <= 0 or chi > 20

    if _bad_chi(chi_mtanh):
        params_mtanh = chi_mtanh = None
    if _bad_chi(chi_cubic):
        params_cubic = chi_cubic = None

    if params_mtanh is not None:
        lo = params_mtanh[0] - params_mtanh[1]
        hi = params_mtanh[0] + params_mtanh[1]
        in_pedestal = is_channel & (rho > lo) & (rho < hi)
        if np.count_nonzero(in_pedestal) < 3:
            params_mtanh = chi_mtanh = None

    if (
        profile_type == "te"
        and params_mtanh is not None
        and params_mtanh[1] > _TE_MTANH_MAX_WIDTH
    ):
        params_mtanh = chi_mtanh = None

    # Lowest reduced chi-squared wins, the mtanh on a tie
    candidates = [
        (chi_mtanh, Osborne_Tanh_cubic_zero_axis_slope, params_mtanh),
        (chi_cubic, CubicZeroAxisSlope, params_cubic),
    ]
    viable = [c for c in candidates if c[0] is not None]
    if not viable:
        return None
    _, best_func, best_params = min(viable, key=lambda c: c[0])
    return best_func, best_params
