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

Pure numpy/scipy: this module sits on the worker's `python -m` import path
into a minimal cluster venv (stdlib + numpy + mkgp only, see batch_io.py's
module docstring) and must stay importable there.
"""

import numpy as np
from scipy.optimize import curve_fit


def chi_squared(ydata, yfit, yerr):
    """Compute the error-weighted chi-squared of a fit.

    Returns:
        The scalar chi-squared value.
    """
    return np.sum((ydata - yfit) ** 2 / yerr**2)


def reduced_chi_squared_inside_separatrix(
    psi, ydata, yfit, yerr, num_params, only_edge=False
):
    """Chi-squared evaluated only inside the separatrix (psi < 1).

    If only_edge=True, restrict further to 0.6 < psi < 1.0.

    Returns:
        The reduced chi-squared over the masked points, or np.inf if there
        are not enough points to constrain the fit.
    """
    if only_edge:
        mask = (psi > 0.6) & (psi < 1.0)
    else:
        mask = psi < 1.0
    n = np.sum(mask)
    if n <= num_params:
        return np.inf
    return chi_squared(ydata[mask], yfit[mask], yerr[mask]) / (n - num_params)


def Osborne_Tanh_linear(x, c0, c1, c2, c3, c4):
    """Osborne tanh with a linear inboard slope and flat outboard (SOL) term.

    c0: pedestal centre, c1: full width, c2: top, c3: bottom, c4: inboard linear.

    Returns:
        The function evaluated at `x`.
    """
    z = 2.0 * (c0 - x) / c1
    P1 = 1.0 + c4 * z
    P2 = 1.0
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1 * E1 - P2 * E2) / (E1 + E2))


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


def Osborne_Tanh_cubic_linear_SOL(x, c0, c1, c2, c3, c4, c5, c6, c7):
    """Osborne tanh with cubic inboard terms and a linear outboard (SOL) term.

    c7: outboard linear term.

    Returns:
        The function evaluated at `x`.
    """
    z = 2.0 * (c0 - x) / c1
    P1 = 1.0 + c4 * z + c5 * z**2 + c6 * z**3
    P2 = 1.0 + c7 * z
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1 * E1 - P2 * E2) / (E1 + E2))


def Osborne_Tanh_cubic_quadratic_SOL(x, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    """Osborne tanh with cubic inboard terms and a quadratic outboard (SOL) term.

    c7: outboard linear, c8: outboard quadratic.

    Returns:
        The function evaluated at `x`.
    """
    z = 2.0 * (c0 - x) / c1
    P1 = 1.0 + c4 * z + c5 * z**2 + c6 * z**3
    P2 = 1.0 + c7 * z + c8 * z**2
    E1 = np.exp(z)
    E2 = np.exp(-z)
    return 0.5 * (c2 + c3 + (c2 - c3) * (P1 * E1 - P2 * E2) / (E1 + E2))


def CubicZeroAxisSlope(x, c0, c2, c3):
    """Cubic polynomial with its linear term dropped.

    f'(0) == 0 identically for any c2/c3, enforcing zero profile gradient
    at the magnetic axis (x=0) exactly, without a bounded/constrained
    optimizer.

    Returns:
        The polynomial evaluated at `x`.
    """
    return c0 + c2 * x**2 + c3 * x**3


def Osborne_linear_initial_guesses(psi_edge, values_edge, n_params=7):
    """Rough initial guesses for an Osborne tanh fit based on edge-only data.

    Returns:
        A list of length n_params with trailing zeros for polynomial terms.
    """
    if len(values_edge) < 4:
        return [1.0, 0.04, float(np.nanmax(values_edge)), 0.0] + [0.0] * (n_params - 4)

    avg = np.nanmean(values_edge)
    bottom = avg * 0.3
    top = avg * 1.1

    max_r = psi_edge[-1]
    min_r = psi_edge[-1]
    for i in range(len(psi_edge) - 1, -1, -1):
        if values_edge[i] > bottom:
            max_r = psi_edge[i]
            break
    for i in range(len(psi_edge) - 1, -1, -1):
        if values_edge[i] > top:
            min_r = psi_edge[i]
            break

    width = max_r - min_r
    if width <= 0:
        width = abs(psi_edge[-3] - psi_edge[-5]) if len(psi_edge) > 5 else 0.05
    centre = (max_r + min_r) / 2.0

    return [centre, width, top, bottom] + [0.0] * (n_params - 4)


# Map (core_order, sol_order) to the fit function and number of parameters.
_FIT_FUNCTION_MAP = {
    (3, 0): (Osborne_Tanh_cubic, 7),
    (3, 1): (Osborne_Tanh_cubic_linear_SOL, 8),
    (3, 2): (Osborne_Tanh_cubic_quadratic_SOL, 9),
}


def get_fit_function(core_order, sol_order):
    """Return (fit_function, n_params) for the given polynomial orders.

    core_order: polynomial order for the inboard region (only 3 currently
    supported); sol_order: polynomial order for the outboard SOL (0, 1, or 2).

    Returns:
        A (fit_function, n_params) tuple.

    Raises:
        ValueError: If no fit function exists for the requested orders.
    """
    key = (core_order, sol_order)
    if key not in _FIT_FUNCTION_MAP:
        raise ValueError(
            f"No fit function for core_order={core_order}, sol_order={sol_order}. "
            f"Supported: {list(_FIT_FUNCTION_MAP.keys())}"
        )
    return _FIT_FUNCTION_MAP[key]


def evaluate_with_gradient(fit_func, popt, x, h=1.0e-4):
    """Evaluate a vendored fit function and its central-difference derivative.

    Used to reconstruct the analytic mean fit's own contribution to the total
    profile gradient/value: the mkgp GP stage (gp.py) only fits the
    *residual* against this mean, so its own posterior mean/derivative do not
    include the mean's own value/slope -- worker_akho.py's `_fit_variable`
    stacks this on top of the GP-residual result. No closed-form derivative
    is used (rather than deriving one per fit function) since the two
    functions `_fit_one_profile` can pick between (Osborne_Tanh_cubic,
    CubicZeroAxisSlope) both need one and a single finite-difference helper
    covers both.

    Args:
        fit_func: One of this module's Osborne_Tanh_*/CubicZeroAxisSlope functions.
        popt: Its fitted parameters (from _fit_one_profile).
        x: Points to evaluate at.
        h: Central-difference step.

    Returns:
        (mean, dmean_dx), each shaped like `x`.
    """
    x = np.asarray(x, dtype=float)
    mean = fit_func(x, *popt)
    dmean_dx = (fit_func(x + h, *popt) - fit_func(x - h, *popt)) / (2.0 * h)
    return mean, dmean_dx


def _fit_one_profile(
    psi,
    values,
    errors,
    psi_grid,
    fit_func,
    n_params,
    enforce_mtanh,
    use_edge_chi_squared,
    profile_type,
    last_params=None,
    debug_plot=False,
    edge_thresh=0.8,
):
    """Fit one profile (Te or ne) with mtanh and optionally cubic.

    Returns:
        A (profile_on_grid, chi_squared, fit_type_str, updated_last_params)
        tuple. Any element may be None if all fits fail.
    """
    is_ne = profile_type == "ne"
    scale = 1e20 if is_ne else 1.0
    vals_s = values / scale
    errs_s = errors / scale
    max_val = max(vals_s)

    # Bounds: [c0, c1, c2, c3] + polynomial terms
    # c0=centre, c1=width, c2=top, c3=bottom — matching original code exactly
    if is_ne:
        lb = [0.85, 0.01, 0.0, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.25, max_val, np.inf] + [np.inf] * (n_params - 4)
    else:
        lb = [0.85, 0.01, 10.0 / scale, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.2, max_val, max_val] + [np.inf] * (n_params - 4)

    # Hardcoded initial guesses — from original repo
    if is_ne:
        hardcoded = [
            # 650 kA C-Mod
            [
                1.00604712,
                3.7400836e-02,
                2.10662412,
                1.68897974e-02,
                -6.32778417e-02,
                2.29233952e-03,
                -2.0627212e-05,
            ],
            # 1 MA C-Mod
            [
                1.02123755e00,
                5.02744526e-02,
                2.54219267e00,
                -9.99999694e-04,
                2.58724602e-02,
                -2.32961078e-03,
                4.20279037e-05,
            ],
            # D3D
            [0.99, 0.04, 1.0, 0.05, 0.0, 0.0, 0.0],
            [0.99, 0.04, 0.3, 0.05, 0.0, 0.0, 0.0],
        ]
    else:
        hardcoded = [
            # from fit_te_mtanh in original functions_profile_fitting.py
            [
                9.92614859e-01,
                4.01791101e-02,
                2.55550908e02 / scale,
                1.28542623e01 / scale,
                2.17777084e-01,
                -3.45196862e-03,
                1.42947373e-04,
            ],
        ]

    guesses = []
    for g in hardcoded:
        guesses.append(np.array((list(g) + [0.0] * n_params)[:n_params]))
    # Osborne_linear auto-guess uses edge data including the SOL zero anchor,
    # matching the original which passes raw_te_psi_edge/raw_ne_psi_edge after
    # add_SOL_zeros_in_psi_coords (functions_fit_1D.py lines 333-334, 473)
    try:
        edge_sel = psi > edge_thresh
        auto = Osborne_linear_initial_guesses(psi[edge_sel], vals_s[edge_sel], n_params)
        guesses.insert(0, np.array(auto))
    except Exception:
        pass
    if last_params is not None:
        guesses.insert(0, np.array(last_params))

    # Clamp all guesses to the parameter bounds.  Hardcoded C-Mod/D3D values
    # have c2 (pedestal top) much higher than D3D data, putting them outside
    # the ub[2]=max_val bound and causing curve_fit to raise immediately.
    lb_arr = np.array(lb)
    ub_arr = np.array(ub)
    guesses = [np.clip(g, lb_arr, ub_arr) for g in guesses]

    # Prepend data-adaptive guesses with c2 tuned to the actual data maximum
    # so the optimizer starts from a physically reasonable point.
    c2_est = max_val * 0.85
    c3_est = max_val * 0.02
    for _c0, _c1 in [(0.99, 0.04), (0.98, 0.04), (0.99, 0.05), (1.00, 0.03)]:
        g = np.zeros(n_params)
        g[0] = _c0
        g[1] = _c1
        g[2] = c2_est
        if n_params > 3:
            g[3] = c3_est
        guesses.insert(0, g)

    # ---- mtanh fit ----
    params_mtanh = profile_mtanh = chi_mtanh = None
    for guess in guesses:
        try:
            params_mtanh, _ = curve_fit(
                fit_func,
                psi,
                vals_s,
                p0=guess,
                sigma=errs_s,
                absolute_sigma=True,
                maxfev=2000,
                bounds=(lb, ub),
            )
            fitted_at_data = fit_func(psi, *params_mtanh)
            chi_mtanh = reduced_chi_squared_inside_separatrix(
                psi,
                vals_s,
                fitted_at_data,
                errs_s,
                n_params,
                only_edge=use_edge_chi_squared,
            )
            profile_mtanh = fit_func(psi_grid, *params_mtanh) * scale
            break
        except Exception:
            continue

    # if enforce_mtanh is True and the fit has succeeed, we can return immediately here
    if enforce_mtanh:
        if profile_mtanh is not None:
            return profile_mtanh, chi_mtanh, "mtanh", params_mtanh
        return None, None, None, last_params

    # ---- cubic fallback ----
    params_cubic = profile_cubic = chi_cubic = None
    try:
        params_cubic, _ = curve_fit(
            CubicZeroAxisSlope,
            psi,
            vals_s,
            sigma=errs_s,
            absolute_sigma=True,
            maxfev=2000,
        )
        chi_cubic = reduced_chi_squared_inside_separatrix(
            psi,
            vals_s,
            CubicZeroAxisSlope(psi, *params_cubic),
            errs_s,
            3,
            only_edge=use_edge_chi_squared,
        )
        profile_cubic = CubicZeroAxisSlope(psi_grid, *params_cubic) * scale
    except Exception:
        pass

    # discard fits with nonsensical chi-squared
    def _bad_chi(chi):
        return chi is None or chi <= 0 or chi > 20

    if _bad_chi(chi_mtanh):
        params_mtanh = profile_mtanh = chi_mtanh = None
    if _bad_chi(chi_cubic):
        params_cubic = profile_cubic = chi_cubic = None

    # discard mtanh if fewer than 3 points in the pedestal region
    if params_mtanh is not None:
        lo, hi = params_mtanh[0] - params_mtanh[1], params_mtanh[0] + params_mtanh[1]
        if np.sum((psi > lo) & (psi < hi)) < 3:
            params_mtanh = profile_mtanh = chi_mtanh = None

    # choose best
    if profile_mtanh is not None and profile_cubic is not None:
        if chi_cubic < chi_mtanh:
            return profile_cubic, chi_cubic, "cubic", params_cubic
        return profile_mtanh, chi_mtanh, "mtanh", params_mtanh
    if profile_mtanh is not None:
        return profile_mtanh, chi_mtanh, "mtanh", params_mtanh
    if profile_cubic is not None:
        return profile_cubic, chi_cubic, "cubic", params_cubic
    return None, None, None, last_params
