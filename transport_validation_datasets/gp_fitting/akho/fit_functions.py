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

# Te-only pedestal gate (see _fit_one_profile): mtanh's extra flexibility
# routinely beats the zero-axis-slope cubic on the FULL-domain chi-squared
# even with no real pedestal, because it fits the CORE shape better with its
# independent inboard polynomial term, not because it fits the edge better
# (see NOTES.md's iteration-2 Te mtanh investigation). Two earlier versions
# of this gate were tried and replaced per explicit direction: an edge-only
# (0.6<psi<1.0) chi-squared ratio (over-rejected legitimate pedestals), then
# a plain geometric anchor-gap test with a hand-picked threshold (the gap
# value alone was not cleanly monotonic with visual pedestal presence across
# the dataset -- similar gaps looked like very different profile shapes).
# This version turns the gap into a significance: compare where the cubic
# fit's 0.85-0.9 trend would land at rho=1.0 (with its own uncertainty,
# propagated from curve_fit's covariance -- robust because cubic borrows
# strength from the whole profile, not just a narrow noisy edge band)
# against where the SOL boundary anchors (rho 1.05/1.08) say it should
# actually be (with their own known measurement errors), as gap / sigma_gap.
# Calibrated against a known L-mode window (CSV index 42, shot 1030516024
# 0.7-0.9s: significance -0.89) and a known EDA H-mode window (shot
# 1160712015, 1.1-1.3s: significance +5.28), with six further windows
# showing a visually unambiguous pedestal shoulder in the raw data landing
# at 1.3-3.9 sigma -- _TE_MTANH_MIN_ANCHOR_SIGMA sits at the low end of that
# cluster (so all six still pass) and well above the L-mode reference.
# _TE_MTANH_MAX_WIDTH is a backstop against the unrelated "broad
# core-shaping sigmoid pinned at the width bound" failure mode also seen on
# idx42 (width pinned at 0.15).
_TE_MTANH_MIN_ANCHOR_SIGMA = 1.3
_TE_MTANH_MAX_WIDTH = 0.10


def chi_squared(ydata, yfit, yerr):
    """Compute the error-weighted chi-squared of a fit.

    Returns:
        The scalar chi-squared value.
    """
    return np.sum((ydata - yfit) ** 2 / yerr**2)


def reduced_chi_squared_inside_separatrix(
    rho, ydata, yfit, yerr, num_params, only_edge=False
):
    """Chi-squared evaluated only inside the separatrix (rho < 1).

    If only_edge=True, restrict further to 0.6 < rho < 1.0.

    Returns:
        The reduced chi-squared over the masked points, or np.inf if there
        are not enough points to constrain the fit.
    """
    if only_edge:
        mask = (rho > 0.6) & (rho < 1.0)
    else:
        mask = rho < 1.0
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


# Map (core_order, sol_order) to the fit function and number of parameters.
_FIT_FUNCTION_MAP = {
    (3, 0): (Osborne_Tanh_cubic, 7),
    (3, 1): (Osborne_Tanh_cubic_linear_SOL, 8),
    (3, 2): (Osborne_Tanh_cubic_quadratic_SOL, 9),
}

# Zero-axis-slope variants (c4 eliminated, one fewer free parameter).
_ZERO_AXIS_SLOPE_FIT_FUNCTION_MAP = {
    (3, 0): (Osborne_Tanh_cubic_zero_axis_slope, 6),
}


def get_fit_function(core_order, sol_order, zero_axis_slope=False):
    """Return (fit_function, n_params) for the given polynomial orders.

    core_order: polynomial order for the inboard region (only 3 currently
    supported); sol_order: polynomial order for the outboard SOL (0, 1, or 2).
    zero_axis_slope: use the variant whose inboard polynomial is constrained
    to zero profile slope at the axis (see Osborne_Tanh_cubic_zero_axis_slope;
    only supported for sol_order 0).

    Returns:
        A (fit_function, n_params) tuple.

    Raises:
        ValueError: If no fit function exists for the requested orders.
    """
    key = (core_order, sol_order)
    fit_map = _ZERO_AXIS_SLOPE_FIT_FUNCTION_MAP if zero_axis_slope else _FIT_FUNCTION_MAP
    if key not in fit_map:
        raise ValueError(
            f"No fit function for core_order={core_order}, sol_order={sol_order}, "
            f"zero_axis_slope={zero_axis_slope}. Supported: {list(fit_map.keys())}"
        )
    return fit_map[key]


def evaluate_with_gradient(fit_func, popt, x, h=1.0e-4):
    """Evaluate a vendored fit function and its central-difference derivative.

    Used to reconstruct the analytic mean fit's own contribution to the total
    profile gradient/value: the mkgp GP stage (gp.py) only fits the
    *residual* against this mean, so its own posterior mean/derivative do not
    include the mean's own value/slope -- worker_akho.py's `_fit_variable`
    stacks this on top of the GP-residual result. No closed-form derivative
    is used (rather than deriving one per fit function) since every function
    `_fit_one_profile` can pick between (Osborne_Tanh_cubic,
    CubicZeroAxisSlope) needs one and a single finite-difference helper
    covers them all.

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
    rho,
    values,
    errors,
    rho_grid,
    fit_func,
    n_params,
    enforce_mtanh,
    use_edge_chi_squared,
    profile_type,
    last_params=None,
    debug_plot=False,
    edge_thresh=0.8,
    attempt_mtanh=True,
):
    """Fit one profile (Te or ne) with mtanh (optional) and cubic.

    attempt_mtanh=False skips the mtanh candidate entirely, so the
    zero-axis-slope cubic is the only model in play (see
    worker_akho._MTANH_VARIABLES for why a caller would want that -- with
    enforce_mtanh it makes every fit fail). The surviving candidates compete
    on reduced chi-squared inside the separatrix.

    Returns:
        A (profile_on_grid, chi_squared, fit_type_str, updated_last_params)
        tuple. Any element may be None if all fits fail.
    """
    is_ne = profile_type == "ne"
    # The inputs arrive from worker_akho already in fit units (te keV,
    # ne 1e20 m^-3) AND normalized by the slice maximum, so they are O(1).
    # The original fit_cmod.py fed this function raw eV / m^-3 data and
    # rescaled ne by 1e20 here (scale = 1e20 if is_ne); kept against
    # normalized inputs, that divide stalled ne's curve_fit at O(1e-20)
    # magnitudes (popt never left p0), and te's raw-eV-era 10 eV pedestal-top
    # floor (lb[2] = 10.0) made its bounds infeasible (ub[2] = max_val ~ 1) --
    # both silently guaranteed the cubic fallback won every window.
    scale = 1.0
    vals_s = values / scale
    errs_s = errors / scale
    max_val = max(vals_s)

    # Bounds: [c0, c1, c2, c3] + polynomial terms
    # c0=centre, c1=width, c2=top, c3=bottom — matching original code exactly
    if is_ne:
        lb = [0.85, 0.01, 0.0, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.25, max_val, np.inf] + [np.inf] * (n_params - 4)
    else:
        # te width capped at 0.15 (per explicit direction; the original used
        # 0.2). The te-only bound-pinning rejection that used to accompany
        # this cap was removed (per explicit direction) so the mtanh competes
        # freely against the polynomial candidates on chi-squared.
        lb = [0.85, 0.01, 0.0, -0.001] + [-np.inf] * (n_params - 4)
        ub = [1.1, 0.15, max_val, max_val] + [np.inf] * (n_params - 4)

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
    zero_axis = fit_func is Osborne_Tanh_cubic_zero_axis_slope
    for g in hardcoded:
        g = list(g)
        if zero_axis:
            # Hardcoded guesses carry [c0, c1, c2, c3, c4, c5, c6]; the
            # zero-axis-slope variant has no free c4, so drop that slot to
            # keep the polynomial terms aligned.
            del g[4]
        guesses.append(np.array((g + [0.0] * n_params)[:n_params]))
    # Osborne_linear auto-guess uses edge data including the SOL zero anchor,
    # matching the original which passes raw_te_psi_edge/raw_ne_psi_edge after
    # add_SOL_zeros_in_psi_coords (functions_fit_1D.py lines 333-334, 473)
    try:
        edge_sel = rho > edge_thresh
        auto = Osborne_linear_initial_guesses(rho[edge_sel], vals_s[edge_sel], n_params)
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
    for guess in guesses if attempt_mtanh else []:
        try:
            params_mtanh, _ = curve_fit(
                fit_func,
                rho,
                vals_s,
                p0=guess,
                sigma=errs_s,
                absolute_sigma=True,
                maxfev=2000,
                bounds=(lb, ub),
            )
            fitted_at_data = fit_func(rho, *params_mtanh)
            chi_mtanh = reduced_chi_squared_inside_separatrix(
                rho,
                vals_s,
                fitted_at_data,
                errs_s,
                n_params,
                only_edge=use_edge_chi_squared,
            )
            profile_mtanh = fit_func(rho_grid, *params_mtanh) * scale
            break
        except Exception:
            continue

    # if enforce_mtanh is True and the fit has succeeed, we can return immediately here
    if enforce_mtanh:
        if profile_mtanh is not None:
            return profile_mtanh, chi_mtanh, "mtanh", params_mtanh
        return None, None, None, last_params

    # ---- zero-axis-slope polynomial fallback (cubic) ----
    # A quintic candidate (QuinticZeroAxisSlope, x^4/x^5 terms) was tried in
    # iteration 2 and dropped in iteration 3: its extra core freedom is
    # unconstrained by data inside the innermost Thomson channel (typically
    # rho ~ 0.26-0.29), so it routinely overfit that extrapolation region
    # into a nonphysical dip below its own axis value before rising back out
    # to meet the data -- confirmed on 218 of 611 (36%) quintic-selected Te
    # windows in a full-CSV scan, not a corner case.
    params_cubic = profile_cubic = chi_cubic = pcov_cubic = None
    try:
        params_cubic, pcov_cubic = curve_fit(
            CubicZeroAxisSlope,
            rho,
            vals_s,
            sigma=errs_s,
            absolute_sigma=True,
            maxfev=2000,
        )
        chi_cubic = reduced_chi_squared_inside_separatrix(
            rho,
            vals_s,
            CubicZeroAxisSlope(rho, *params_cubic),
            errs_s,
            3,
            only_edge=use_edge_chi_squared,
        )
        profile_cubic = CubicZeroAxisSlope(rho_grid, *params_cubic) * scale
    except Exception:
        pass

    # discard fits with nonsensical chi-squared
    def _bad_chi(chi):
        return chi is None or chi <= 0 or chi > 20

    if _bad_chi(chi_mtanh):
        params_mtanh = profile_mtanh = chi_mtanh = None
    if _bad_chi(chi_cubic):
        params_cubic = profile_cubic = chi_cubic = pcov_cubic = None

    # discard mtanh if fewer than 3 points in the pedestal region
    if params_mtanh is not None:
        lo, hi = params_mtanh[0] - params_mtanh[1], params_mtanh[0] + params_mtanh[1]
        if np.sum((rho > lo) & (rho < hi)) < 3:
            params_mtanh = profile_mtanh = chi_mtanh = None

    # Te-only pedestal gate: judge whether a real pedestal-like drop is
    # needed between the confined profile and the synthetic SOL boundary
    # anchors at rho 1.05/1.08 (injected upstream by cmod_dataset.py's
    # channel prefilters -- present as two ordinary points in rho/vals_s
    # here, identified by their exact rho since real Thomson channels never
    # land there). Draw a line through the cubic fit's values at rho 0.85
    # and 0.90 and extrapolate it to rho=1.0 (where the confined-region
    # trend would land if nothing special happened), and a second line
    # through the two anchors extrapolated back to rho=1.0 (the actual SOL
    # boundary condition). Both extrapolations carry a propagated
    # uncertainty (the cubic one from curve_fit's parameter covariance, the
    # anchor one from the anchors' own known measurement errors), so the
    # test is a significance (gap / sigma_gap), not a bare number -- a gap
    # only counts as a real pedestal if it is large relative to how well
    # both sides are actually determined. If the core-side extrapolation
    # lands at or below the anchor-implied value (within that uncertainty),
    # the profile is smoothly reaching the SOL on its own and mtanh's
    # pedestal shape is not needed; if it lands well above, a real steep
    # drop must happen past rho=0.9 that only mtanh can represent. A width
    # sanity check is kept as a backstop against the unrelated "broad
    # core-shaping sigmoid pinned at the width bound" failure mode (see the
    # module-level constants' docstring). ne is unaffected: it wants its own
    # edge structure regardless of confinement regime.
    if profile_type == "te" and params_mtanh is not None:
        width_ok = params_mtanh[1] <= _TE_MTANH_MAX_WIDTH
        gap_ok = True
        anchor_1 = np.isclose(rho, 1.05, atol=1.0e-4)
        anchor_2 = np.isclose(rho, 1.08, atol=1.0e-4)
        if (
            params_cubic is not None
            and pcov_cubic is not None
            and anchor_1.sum() == 1
            and anchor_2.sum() == 1
        ):
            def _core_at_1(p):
                y85 = CubicZeroAxisSlope(0.85, *p)
                y90 = CubicZeroAxisSlope(0.90, *p)
                return y90 + (y90 - y85) / 0.05 * 0.10

            core_at_1 = _core_at_1(params_cubic)
            h = 1.0e-6
            jac = np.zeros(3)
            for i in range(3):
                p_plus = np.array(params_cubic, dtype=float)
                p_plus[i] += h
                p_minus = np.array(params_cubic, dtype=float)
                p_minus[i] -= h
                jac[i] = (_core_at_1(p_plus) - _core_at_1(p_minus)) / (2.0 * h)
            core_var = float(jac @ pcov_cubic @ jac)

            y_a1, y_a2 = vals_s[anchor_1][0], vals_s[anchor_2][0]
            s_a1, s_a2 = errs_s[anchor_1][0], errs_s[anchor_2][0]
            t = (1.0 - 1.05) / (1.08 - 1.05)
            anchor_at_1 = y_a1 * (1.0 - t) + y_a2 * t
            anchor_var = (1.0 - t) ** 2 * s_a1**2 + t**2 * s_a2**2

            gap = core_at_1 - anchor_at_1
            gap_sigma = np.sqrt(max(core_var, 0.0) + max(anchor_var, 0.0))
            significance = gap / gap_sigma if gap_sigma > 0 else np.inf
            gap_ok = significance >= _TE_MTANH_MIN_ANCHOR_SIGMA
        if not (width_ok and gap_ok):
            params_mtanh = profile_mtanh = chi_mtanh = None

    # choose best (lowest reduced chi-squared; mtanh listed first so it keeps
    # the historical tie-break preference over the polynomial)
    candidates = [
        (chi_mtanh, profile_mtanh, "mtanh", params_mtanh),
        (chi_cubic, profile_cubic, "cubic", params_cubic),
    ]
    viable = [c for c in candidates if c[1] is not None]
    if not viable:
        return None, None, None, last_params
    chi_best, profile_best, name_best, params_best = min(viable, key=lambda c: c[0])
    return profile_best, chi_best, name_best, params_best
