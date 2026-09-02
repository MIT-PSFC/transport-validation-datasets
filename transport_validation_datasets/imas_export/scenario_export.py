"""Builds one shot's `equilibrium`/`core_profiles`/`summary`/`wall` IMAS IDS
from its fit results and unprocessed data, and writes them out.

Ported from `cmod_to_imas/export_scenario_to_imas.py` (this project's
cmod_to_imas directory), reshaped from "build one time slice from a flat
consolidated scenario file" to "build N time slices from
(fit_shots_dir/<shot>.nc, 01_unprocessed/<shot>.nc)" -- see this project's
plan notes for the full reasoning. In particular:

  - `equilibrium` gets one `time_slice` per real EFIT reconstruction time in
    the shot's unprocessed data (its own, full native time base -- not
    thinned to or deduped against the Thomson slice times).
  - `summary` is written at its own full native (1 kHz) 0D-signal
    resolution, for the same reason, and carries every
    `workflow.FINAL_0D_SIGNALS` entry the shot has (see
    `_SUMMARY_SIGNAL_PATHS` for where each one lands).
  - `core_profiles` gets one `profiles_1d` per usable Thomson slice time,
    electrons (with the GP fit's 1-sigma uncertainties in the DD's
    `*_error_upper` fields; `*_error_lower` is left unset, which the IMAS
    convention reads as a symmetric error) + a single hydrogenic
    main ion (Zeff=1, n_D=n_e). Zeff/impurity
    composition are deliberately NOT computed here, and this package holds no
    code for them at all -- that physics lives in a standalone
    `postprocess_ion_composition.py` script kept entirely outside this repo
    (in `cmod_to_imas`, this project's origin directory, to avoid a TORAX
    dependency clash -- see this project's plan notes), which computes them
    afterward from already-written IMAS output and writes a new
    `core_profiles.nc`. This module and `DataWorkflow.export_to_imas()` never
    call it -- that decoupling was a deliberate later revision.
  - No `edge_physics.py` port, no `edge_geometry.nc` output (dropped, see
    this project's plan notes) -- `_find_x_point` below is the one small
    exception: it classifies `equilibrium.time_slice.boundary.type`
    (diverted vs. limited), a core equilibrium-IDS field, not part of the
    dropped extended-Lengyel edge model, so it is vendored on its own rather
    than pulled in through the rest of that (otherwise unported) module.
  - No `imas_export/consolidate.py` / scenario file -- reads the two
    xr.Datasets directly (see `build_imas_from_shot`).

Each IDS is built as a real `imas` library object
(`imas.IDSFactory(...).<ids>()`, field assignment, `.validate()`) and written
straight to `<output_dir>/<ids>.nc` via `imas.DBEntry(...).put()`
(`write_ids`). The original went through `fusio`'s dotted-key `xr.Dataset`
round trip (`imas.util.to_xarray` -> `imas_io().write()`) for everything but
`wall`. That ended in the very same `DBEntry.put()` call, and its
array-of-structures shape inference silently kept the nested per-slice
arrays (`profiles_1d[i].ion[...]`, `time_slice[i].profiles_2d[...]`) only
for the first time slice, emptying them on every later one (confirmed
against fusio 0.4.4 on synthetic 4-slice IDS). Writing the built IDS
directly fixes that and drops the `fusio` dependency.
"""

from dataclasses import dataclass
from pathlib import Path

import eqdsk
import imas
import megpy
import numpy as np
import xarray as xr
from megpy import tracer as megpy_tracer
from scipy.integrate import cumulative_simpson
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from transport_validation_datasets.imas_export.geqdsk_writer import write_geqdsk

# COCOS convention of this project's raw C-Mod GEQDSK data (same as
# cmod_to_imas's config_builder.py convention for every real production run).
_SOURCE_COCOS = 7
# COCOS convention the written equilibrium IDS targets -- IMAS's own
# convention (per the same explicit user direction cmod_to_imas used).
_TARGET_COCOS = 11

DD_VERSION = "4.0.0"

# core_profiles' one ion species (see build_core_profiles): deuterium, with
# the same nominal mass number (2.0, not 2.014) and nuclear charge
# fusio.utils.plasma_tools.define_ion_species("D") used to supply, so the
# written output is unchanged.
_MAIN_ION_NAME = "D"
_MAIN_ION_A = 2.0
_MAIN_ION_Z_N = 1

# A true X-point is a magnetic null (B_pol = 0) sitting on the LCFS. Below
# this fraction of the LCFS's typical (median) B_pol, the minimum found by
# `_find_x_point` is treated as a genuine null rather than just the locally
# weakest field on an otherwise smooth (limited) boundary.
_X_POINT_BPOL_RATIO_THRESHOLD = 0.1

# Near-axis flux-surface-tracing refinement (see `_refine_near_axis_
# fluxsurfaces`'s own docstring): the number of native-grid surfaces closest
# to the magnetic axis to retrace on a locally bicubic-refined 2D psi(R,Z)
# map, the half-width (in meters) of that local refinement box around the
# magnetic axis, and the refinement factor relative to the source file's own
# native R/Z grid spacing.
_NEAR_AXIS_REFINE_N_SURFACES = 10
_NEAR_AXIS_REFINE_BOX_M = 0.15
_NEAR_AXIS_REFINE_FACTOR = 12


@dataclass
class ShotExportSlice:
    """One Thomson slice's worth of data assembled for the IMAS export.

    Lives only for the duration of building one shot's IMAS output.
    `core_profiles` gets electrons + a single hydrogenic main ion from this
    directly (Zeff=1, n_D=n_e) -- Zeff/impurity composition are deliberately
    not computed here or anywhere in this package; see the module docstring
    above for where that physics actually lives.

    Attributes:
        time: Thomson slice time [s].
        t_e: Electron temperature on `rho` [eV].
        n_e: Electron density on `rho` [m^-3].
        rho: Normalized rho grid t_e/n_e are given on.
        psi_axis: Matched equilibrium's psi at the magnetic axis (COCOS-11).
        psi_boundary: Matched equilibrium's psi at the boundary (COCOS-11).
        psin_neo: `rho` mapped onto the matched equilibrium's psi_norm grid
            (see `build_equilibrium`'s `derived_by_time`), for placing
            `profiles_1d.grid.psi`.
        ip: Nearest-time plasma current [A].
        t_e_error: 1-sigma uncertainty of `t_e` [eV], or None if the fit
            file carries none.
        n_e_error: 1-sigma uncertainty of `n_e` [m^-3], or None likewise.
    """

    time: float
    t_e: np.ndarray
    n_e: np.ndarray
    rho: np.ndarray
    psi_axis: float
    psi_boundary: float
    psin_neo: np.ndarray
    ip: float
    t_e_error: np.ndarray | None = None
    n_e_error: np.ndarray | None = None


# ---------------------------------------------------------------------------
# megpy equilibrium loading (ported from export_scenario_to_imas.py)
# ---------------------------------------------------------------------------


def _load_megpy_equilibrium(geqdsk_path):
    """Loads a `.geqdsk` file (source COCOS) into a `megpy.Equilibrium`,
    with flux surfaces, Miller shape parameters, and B-field quantities all
    populated. See `_refine_near_axis_fluxsurfaces`'s docstring for why the
    near-axis surfaces are then retraced.
    """
    eq = megpy.Equilibrium()
    eq.read_geqdsk(f_path=str(geqdsk_path))
    eq.add_derived(incl_fluxsurfaces=True, analytic_shape=True, incl_B=True)
    _refine_near_axis_fluxsurfaces(eq)
    return eq


def _refine_near_axis_fluxsurfaces(
    eq,
    n_surfaces=_NEAR_AXIS_REFINE_N_SURFACES,
    box_m=_NEAR_AXIS_REFINE_BOX_M,
    refine_factor=_NEAR_AXIS_REFINE_FACTOR,
):
    """Retraces the `n_surfaces` native-grid flux surfaces closest to the
    magnetic axis on a locally bicubic-refined 2D psi(R,Z) map, in place.

    megpy's default tracer traces each flux surface on the file's native 2D
    grid; for the surface closest to the axis this systematically
    underestimates its area (confirmed empirically in cmod_to_imas: ~8.66%
    low at the innermost surface, decaying to 1.4% by the sixth). This
    builds one `RectBivariateSpline` of the existing psi(R,Z) data (no new
    physics, just finer sampling where the native grid is too coarse) and
    re-traces the near-axis surfaces on it.
    """
    d = eq.derived
    R, Z, psirz = d["R"], d["Z"], d["psirz"]
    rmaxis, zmaxis = float(d["rmaxis"]), float(d["zmaxis"])
    mag_axis = np.array([rmaxis, zmaxis])

    spline = RectBivariateSpline(Z, R, psirz, kx=3, ky=3, s=0)
    n_r = max(int(2.0 * box_m / (R[1] - R[0]) * refine_factor), 4)
    n_z = max(int(2.0 * box_m / (Z[1] - Z[0]) * refine_factor), 4)
    R_fine = np.linspace(rmaxis - box_m, rmaxis + box_m, n_r)
    Z_fine = np.linspace(zmaxis - box_m, zmaxis + box_m, n_z)
    psirz_fine = spline(Z_fine, R_fine)

    fs = eq.fluxsurfaces
    n_last = min(n_surfaces, len(d["psi"]) - 1)
    for i in range(1, n_last + 1):
        psi_fs = float(d["psi"][i])
        c = megpy_tracer.contour(R_fine, Z_fine, psirz_fine, psi_fs, kind="l", ref_point=mag_axis)

        for key in list(c.keys()):
            if "X" in key or "Y" in key:
                c[key.replace("X", "R").replace("Y", "Z")] = c.pop(key)
        del c["level"]
        c.pop("contours", None)

        c["rho_tor"] = float(d["rho_tor"][i])
        c["psi"] = psi_fs
        c["q"] = float(d["qpsi"][i])
        c["fpol"] = float(d["fpol"][i])

        Rc, Zc = np.asarray(c["R"]), np.asarray(c["Z"])
        dpsidz = spline(Zc, Rc, dx=1, dy=0, grid=False)
        dpsidr = spline(Zc, Rc, dx=0, dy=1, grid=False)
        Bpol = np.sqrt((dpsidz / Rc) ** 2 + (dpsidr / Rc) ** 2)
        Btor = c["fpol"] / c["R0"]
        c["Bpol"] = Bpol
        c["Btor"] = np.full_like(Bpol, Btor)
        c["B"] = np.sqrt(Bpol ** 2 + Btor ** 2)
        flux_integrand = np.sqrt(np.diff(Rc) ** 2 + np.diff(Zc) ** 2) / np.abs(Bpol[:-1])
        c["Vprime"] = float(np.sum(flux_integrand))
        c["1/R"] = float(np.sum(flux_integrand / Rc[:-1]) / np.sum(flux_integrand))

        c["miller_geo"] = megpy.LocalEquilibrium.extract_analytic_shape(c)

        for key, value in c.items():
            if key not in fs:
                continue
            if isinstance(fs[key], dict):
                for subkey, subvalue in value.items():
                    fs[key][subkey][i] = subvalue
            else:
                fs[key][i] = value

    if n_last >= 1:
        for key in ("delta_u", "delta_l", "delta", "kappa", "zeta_uo", "zeta_ui", "zeta_li", "zeta_lo", "zeta"):
            if key in fs["miller_geo"]:
                fs["miller_geo"][key][0] = fs["miller_geo"][key][1]
    return eq


def _flux_surface_averages(eq):
    """Computes the IMAS `gm1`/`gm2`/`gm3`/`gm4`/`gm5`/`gm7` flux-surface
    averages `megpy` doesn't already provide directly, using the same
    `dl/|Bpol|`-weighted FSA definition `megpy` itself uses for
    `Vprime`/`1/R`. `gm2`/`gm3`/`gm7` (w.r.t. |grad rho_tor|) come from
    `|grad psi| = Bpol*R` and the chain rule `grad_rho = |drho/dpsi| *
    grad_psi`; the caller (`_gm2_gm3_gm7`) does that rescaling.

    Returns a dict of length-`len(eq.fluxsurfaces['R'])` arrays: `gm1`,
    `gm4`, `gm5`, plus the psi-gradient FSAs `_gm2_gm3_gm7` needs.
    """
    fs = eq.fluxsurfaces
    n = len(fs["R"])
    gm1 = np.empty(n)
    gm4 = np.empty(n)
    gm5 = np.empty(n)
    fsa_gradpsi = np.zeros(n)
    fsa_gradpsi2 = np.zeros(n)
    fsa_gradpsi2_over_r2 = np.zeros(n)

    R_axis = float(eq.derived["rmaxis"])
    Btor_axis = float(eq.derived["fpol"][0]) / R_axis
    gm1[0] = 1.0 / R_axis ** 2
    gm5[0] = Btor_axis ** 2
    gm4[0] = 1.0 / Btor_axis ** 2

    for i in range(1, n):
        R = np.asarray(fs["R"][i])
        Z = np.asarray(fs["Z"][i])
        Bpol = np.asarray(fs["Bpol"][i])
        B = np.asarray(fs["B"][i])
        dl = np.sqrt(np.diff(R) ** 2 + np.diff(Z) ** 2)
        weight = dl / np.abs(Bpol[:-1])
        denom = np.sum(weight)
        gm1[i] = np.sum(weight / R[:-1] ** 2) / denom
        gm5[i] = np.sum(weight * B[:-1] ** 2) / denom
        gm4[i] = np.sum(weight / B[:-1] ** 2) / denom
        grad_psi = Bpol[:-1] * R[:-1]
        fsa_gradpsi[i] = np.sum(weight * grad_psi) / denom
        fsa_gradpsi2[i] = np.sum(weight * grad_psi ** 2) / denom
        fsa_gradpsi2_over_r2[i] = np.sum(weight * grad_psi ** 2 / R[:-1] ** 2) / denom

    return {
        "gm1": gm1, "gm4": gm4, "gm5": gm5,
        "fsa_gradpsi": fsa_gradpsi, "fsa_gradpsi2": fsa_gradpsi2,
        "fsa_gradpsi2_over_r2": fsa_gradpsi2_over_r2,
    }


def _gm2_gm3_gm7(fsa, rho_tor_m, psi, qpsi, bcentr):
    """Converts the psi-based FSAs from `_flux_surface_averages` into the
    rho_tor(meters)-based IMAS `gm2`/`gm3`/`gm7`.

    `fsa_gradpsi*` come from the source-COCOS-native equilibrium, so they
    are first rescaled to target-COCOS (a factor of `2*pi` in magnitude
    between these two COCOS) before combining with the target-COCOS `psi`.
    `drho_dpsi` is a plain finite difference (`np.gradient`); on-axis values
    use the standard near-axis limits (`gm7->1`, `gm3->1`,
    `gm2->1/R_axis**2`, the last one set by the caller).
    """
    cocos_scale = 2.0 * np.pi
    fsa_gradpsi = fsa["fsa_gradpsi"] * cocos_scale
    fsa_gradpsi2 = fsa["fsa_gradpsi2"] * cocos_scale ** 2
    fsa_gradpsi2_over_r2 = fsa["fsa_gradpsi2_over_r2"] * cocos_scale ** 2

    drho_dpsi = np.gradient(rho_tor_m, psi)
    gm7 = fsa_gradpsi * np.abs(drho_dpsi)
    gm3 = fsa_gradpsi2 * drho_dpsi ** 2
    gm2 = fsa_gradpsi2_over_r2 * drho_dpsi ** 2
    gm7[0] = 1.0
    gm3[0] = 1.0
    gm2[0] = gm2[1] if len(gm2) > 1 else 0.0
    return gm2, gm3, gm7


def _spline_scalar(spline, r, z):
    """`RectBivariateSpline(..., grid=False)` at one scalar `(r, z)` point,
    as a plain float regardless of whether the installed scipy returns a
    0-d array, a shape-(1,) array, or a bare float for scalar inputs.
    """
    return float(np.ravel(spline(r, z, grid=False))[0])


def _find_x_point(eq, bp_spline):
    """Locates the magnetic null (X-point) nearest the LCFS, if any, and
    classifies the equilibrium as diverted or limited, for
    `equilibrium.time_slice.boundary.type`.

    Vendored from `cmod_to_imas/edge_physics.py`'s `_find_x_point` (a small,
    self-contained function -- not part of the dropped extended-Lengyel edge
    model, see this module's own docstring). Uses the physical definition of
    an X-point directly: B_pol vanishes there, so the minimum of B_pol along
    the LCFS boundary contour sits at (or very near) any true X-point,
    refined by locally minimizing the B_pol spline from that starting point.

    Returns:
        (r_x, z_x, diverted).
    """
    bp_boundary = bp_spline(eq.derived["rbbbs"], eq.derived["zbbbs"], grid=False)
    imin = int(np.argmin(bp_boundary))
    r0, z0 = eq.derived["rbbbs"][imin], eq.derived["zbbbs"][imin]
    result = minimize(
        lambda p: _spline_scalar(bp_spline, p[0], p[1]) ** 2,
        x0=[r0, z0],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-12},
    )
    r_x, z_x = float(result.x[0]), float(result.x[1])
    bp_min = _spline_scalar(bp_spline, r_x, z_x)
    bp_typical = float(np.median(bp_boundary))
    diverted = bool(bp_min / bp_typical < _X_POINT_BPOL_RATIO_THRESHOLD)
    return r_x, z_x, diverted


def _diverted_from_megpy(eq_mp):
    """Whether `eq_mp` is diverted (True) or limited (False); see `_find_x_point`."""
    bp_spline = RectBivariateSpline(eq_mp.derived["R"], eq_mp.derived["Z"], eq_mp.derived["B_pol_rz"])
    _, _, diverted = _find_x_point(eq_mp, bp_spline)
    return diverted


# ---------------------------------------------------------------------------
# equilibrium IDS (ported from export_scenario_to_imas.py's
# _build_equilibrium, split into a per-time-slice populate step so a
# multi-time-slice equilibrium IDS can be built by looping it)
# ---------------------------------------------------------------------------


@dataclass
class _EquilibriumTimeDerived:
    """Per-equilibrium-time quantities needed alongside the IDS fields
    `_populate_equilibrium_time_slice` writes directly into `ts`.

    `rho_tor_norm`/`psi_norm` are this equilibrium time's own arrays (the
    ones written to `ts.profiles_1d.rho_tor_norm`/`psi_norm`), reused
    directly in `build_imas_from_shot` to map a Thomson rho onto this psi
    grid (`ShotExportSlice.psin_neo`, for `core_profiles.profiles_1d.grid.
    psi`'s placement).
    """

    rho_tor_norm: np.ndarray
    psi_norm: np.ndarray
    bcentr: float
    psi_axis: float
    psi_boundary: float


def _populate_equilibrium_time_slice(ts, eqi, eq_mp):
    """Fills one `equilibrium.time_slice[i]` from one equilibrium
    reconstruction: `eqi` (`eqdsk.EQDSKInterface`, target-COCOS) supplies
    the quantities that genuinely change under the source->target COCOS
    conversion (psi/pressure/pprime/ffprime); `eq_mp` (`megpy.Equilibrium`,
    run on the *original, unconverted* file) supplies everything else
    (numerically COCOS-invariant for this project's conversion, and found
    empirically to come out wrong if `megpy` is instead fed the
    already-converted file -- see cmod_to_imas/export_scenario_to_imas.py's
    `_build_equilibrium` docstring for the full empirical finding this
    reproduces).

    Returns:
        An `_EquilibriumTimeDerived` with this equilibrium time's
        `rho_tor_norm`/`psi_norm`/`bcentr`/`psi_axis`/`psi_boundary`.
    """
    psi_axis = float(eqi.psimag)
    psi_boundary = float(eqi.psibdry)
    psi = np.linspace(psi_axis, psi_boundary, int(eqi.nx))

    ts.profiles_1d.psi = psi
    ts.profiles_1d.psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis)
    ts.profiles_1d.f = np.asarray(eq_mp.derived["fpol"], dtype=float)
    ts.profiles_1d.pressure = np.asarray(eqi.pressure, dtype=float)
    ts.profiles_1d.f_df_dpsi = np.asarray(eqi.ffprime, dtype=float)
    ts.profiles_1d.dpressure_dpsi = np.asarray(eqi.pprime, dtype=float)
    ts.profiles_1d.q = np.asarray(eq_mp.derived["qpsi"], dtype=float)

    r_hfs = np.asarray(eq_mp.fluxsurfaces["R_in"], dtype=float)
    r_lfs = np.asarray(eq_mp.fluxsurfaces["R_out"], dtype=float)
    # Outboard/inboard-midplane-based minor radius, used only for the
    # phi/rho_tor q-integration just below.
    minr_integration = (r_lfs - r_hfs) / 2.0

    fs = eq_mp.fluxsurfaces
    n_surf = len(fs["R"])
    area = np.zeros(n_surf)
    R_axis = float(eq_mp.derived["rmaxis"])
    for i in range(1, n_surf):
        R_i = np.asarray(fs["R"][i])
        Z_i = np.asarray(fs["Z"][i])
        area[i] = 0.5 * np.abs(np.sum(R_i * np.roll(Z_i, -1) - Z_i * np.roll(R_i, -1)))
    volume = area * 2.0 * np.pi * R_axis
    ts.profiles_1d.area = area
    ts.profiles_1d.volume = volume
    ts.global_quantities.area = area[-1]
    ts.global_quantities.volume = volume[-1]
    ts.profiles_1d.dvolume_dpsi = np.gradient(volume, psi)

    qpsi_native = np.asarray(eq_mp.derived["qpsi"], dtype=float)
    bcentr = float(eq_mp.derived["bcentr"])
    dpsi_drmin = np.gradient(psi, minr_integration)
    phi = np.zeros_like(psi)
    if len(psi) > 2:
        phi[1:] = cumulative_simpson(y=qpsi_native * dpsi_drmin, x=minr_integration)[: len(psi) - 1]
    phi = np.abs(phi)
    rho_tor = np.sqrt(phi / (np.pi * abs(bcentr))) if abs(bcentr) > 0 else minr_integration
    rho_tor_a = rho_tor[-1] if rho_tor[-1] > 0.0 else 1.0
    ts.profiles_1d.phi = phi
    ts.profiles_1d.rho_tor = rho_tor
    ts.profiles_1d.rho_tor_norm = rho_tor / rho_tor_a
    ts.profiles_1d.dpsi_drho_tor = np.gradient(psi, rho_tor)

    ts.profiles_1d.r_inboard = r_hfs
    ts.profiles_1d.r_outboard = r_lfs
    ts.profiles_1d.j_phi = np.asarray(eq_mp.derived["j_tor"], dtype=float)
    miller = eq_mp.fluxsurfaces["miller_geo"]
    ts.profiles_1d.triangularity_upper = np.asarray(miller["delta_u"], dtype=float)
    ts.profiles_1d.triangularity_lower = np.asarray(miller["delta_l"], dtype=float)
    ts.profiles_1d.elongation = np.asarray(miller["kappa"], dtype=float)

    fsa = _flux_surface_averages(eq_mp)
    gm2, gm3, gm7 = _gm2_gm3_gm7(fsa, rho_tor, psi, qpsi_native, bcentr)
    gm2[0] = 1.0 / R_axis ** 2
    ts.profiles_1d.gm1 = fsa["gm1"]
    ts.profiles_1d.gm2 = gm2
    ts.profiles_1d.gm3 = gm3
    ts.profiles_1d.gm4 = fsa["gm4"]
    ts.profiles_1d.gm5 = fsa["gm5"]
    ts.profiles_1d.gm7 = gm7
    gm9 = np.asarray(eq_mp.fluxsurfaces["1/R"], dtype=float)
    gm9[0] = 1.0 / R_axis
    ts.profiles_1d.gm9 = gm9

    ts.global_quantities.magnetic_axis.r = R_axis
    ts.global_quantities.magnetic_axis.z = float(eq_mp.derived["zmaxis"])
    ts.global_quantities.psi_axis = psi_axis
    ts.global_quantities.psi_boundary = psi_boundary
    ts.global_quantities.ip = float(eq_mp.derived["current"])

    ts.boundary.outline.r = np.asarray(eq_mp.derived["rbbbs"], dtype=float)
    ts.boundary.outline.z = np.asarray(eq_mp.derived["zbbbs"], dtype=float)
    ts.boundary.minor_radius = float(eq_mp.derived["a"])
    ts.boundary.type = 1 if _diverted_from_megpy(eq_mp) else 0

    ts.profiles_2d.resize(1)
    p2d = ts.profiles_2d[0]
    p2d.grid_type.name = "rectangular"
    p2d.grid.dim1 = np.asarray(eq_mp.derived["R"], dtype=float)
    p2d.grid.dim2 = np.asarray(eq_mp.derived["Z"], dtype=float)
    p2d.psi = np.asarray(eqi.psi, dtype=float)

    return _EquilibriumTimeDerived(
        rho_tor_norm=np.asarray(ts.profiles_1d.rho_tor_norm, dtype=float),
        psi_norm=np.asarray(ts.profiles_1d.psi_norm, dtype=float),
        bcentr=bcentr,
        psi_axis=psi_axis,
        psi_boundary=psi_boundary,
    )


def build_equilibrium(factory, times, geqdsk_paths):
    """`equilibrium` IDS: one `time_slice` per real EFIT reconstruction time.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        times: (n_eq,) real EFIT reconstruction times [s].
        geqdsk_paths: (n_eq,) `.geqdsk` file paths, one per time in `times`
            (see `geqdsk_writer.write_geqdsk`).

    Returns:
        (eq, eq_mp_by_time, derived_by_time): the validated `equilibrium`
        IDS; `eq_mp_by_time` maps each time to its `megpy.Equilibrium`
        (reused by `build_wall`); `derived_by_time` maps each time to its
        `_EquilibriumTimeDerived`, for `core_profiles.profiles_1d.grid.psi`
        placement (see `build_imas_from_shot`).
    """
    eq = factory.equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = np.asarray(times, dtype=float)
    eq.time_slice.resize(len(times))

    eq_mp_by_time = {}
    derived_by_time = {}
    bcentr_per_time = np.empty(len(times))
    r0 = None
    for i, (t, geqdsk_path) in enumerate(zip(times, geqdsk_paths)):
        eqi = eqdsk.EQDSKInterface.from_file(str(geqdsk_path), from_cocos=_SOURCE_COCOS, to_cocos=_TARGET_COCOS)
        eq_mp = _load_megpy_equilibrium(geqdsk_path)
        ts = eq.time_slice[i]
        derived = _populate_equilibrium_time_slice(ts, eqi, eq_mp)
        eq_mp_by_time[t] = eq_mp
        derived_by_time[t] = derived
        bcentr_per_time[i] = derived.bcentr
        if r0 is None:
            r0 = float(eq_mp.derived["rcentr"])

    eq.vacuum_toroidal_field.r0 = r0 if r0 is not None else 0.0
    eq.vacuum_toroidal_field.b0 = bcentr_per_time

    eq.validate()
    return eq, eq_mp_by_time, derived_by_time


# ---------------------------------------------------------------------------
# ion composition (decoupled Zeff/impurity post-processing, consumed here --
# see this project's plan notes, "Post-processing: Zeff and impurity
# composition")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# core_profiles IDS
# ---------------------------------------------------------------------------


def build_core_profiles(factory, slices: list[ShotExportSlice]):
    """`core_profiles` IDS: one `profiles_1d` per usable Thomson slice.

    Electrons + a single hydrogenic main ion (D, Zeff=1, n_D=n_e) -- Zeff and
    impurity composition are deliberately not computed here or anywhere in
    this package; see the module docstring above for where that physics
    actually lives.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        slices: One `ShotExportSlice` per usable Thomson slice, ordered by
            time.

    Returns:
        The validated `core_profiles` IDS.
    """
    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.time = np.asarray([s.time for s in slices], dtype=float)
    cp.profiles_1d.resize(len(slices))

    for i, s in enumerate(slices):
        p1d = cp.profiles_1d[i]
        p1d.time = float(s.time)
        p1d.grid.rho_tor_norm = s.rho
        p1d.grid.psi = s.psin_neo * (s.psi_boundary - s.psi_axis) + s.psi_axis
        p1d.electrons.density = s.n_e
        p1d.electrons.density_thermal = s.n_e
        p1d.electrons.temperature = s.t_e
        # The GP fit's 1-sigma predictive uncertainty is symmetric: per the
        # IMAS convention, filling only `*_error_upper` and leaving
        # `*_error_lower` unset declares exactly that.
        if s.t_e_error is not None:
            p1d.electrons.temperature_error_upper = s.t_e_error
        if s.n_e_error is not None:
            p1d.electrons.density_error_upper = s.n_e_error
            p1d.electrons.density_thermal_error_upper = s.n_e_error
        p1d.zeff = np.ones_like(s.rho, dtype=float)
        # D is the only ion species, so T_i = T_e (as every species would be
        # assigned anyway) makes the density-weighted average exactly t_e.
        p1d.t_i_average = s.t_e
        if s.t_e_error is not None:
            p1d.t_i_average_error_upper = s.t_e_error

        p1d.ion.resize(1)
        ion = p1d.ion[0]
        ion.name = _MAIN_ION_NAME
        ion.density = s.n_e
        ion.density_thermal = s.n_e
        ion.temperature = s.t_e
        # n_D = n_e exactly (single species, Zeff=1), so its uncertainty is
        # n_e's; T_i = T_e likewise.
        if s.n_e_error is not None:
            ion.density_error_upper = s.n_e_error
            ion.density_thermal_error_upper = s.n_e_error
        if s.t_e_error is not None:
            ion.temperature_error_upper = s.t_e_error
        ion.z_ion_1d = np.ones_like(s.t_e, dtype=float)
        ion.z_ion = 1.0
        ion.element.resize(1)
        ion.element[0].z_n = _MAIN_ION_Z_N
        ion.element[0].a = _MAIN_ION_A

    cp.global_quantities.ip = np.asarray([s.ip for s in slices], dtype=float)
    cp.validate()
    return cp


# ---------------------------------------------------------------------------
# summary IDS: full native 0D-signal resolution (see this module's own
# docstring)
# ---------------------------------------------------------------------------


# `workflow.FINAL_0D_SIGNALS` name -> (summary sub-structure, field) it is
# written to; each target is a `summary_dynamic` node whose `.value` holds the
# time series. Every FINAL_0D_SIGNALS entry has a home here, and the paths
# match the `ref` attrs the machine modules record on the unprocessed signals
# (confirmed against the installed DD 4.0.0 by introspection).
_SUMMARY_SIGNAL_PATHS = {
    "ip": ("global_quantities", "ip"),
    "b0": ("global_quantities", "b0"),
    "energy_mhd": ("global_quantities", "energy_mhd"),
    "beta_tor_norm": ("global_quantities", "beta_tor_norm"),
    "power_ohm": ("global_quantities", "power_ohm"),
    "power_radiated": ("global_quantities", "power_radiated"),
    "n_e_line_average": ("line_average", "n_e"),
    "minor_radius": ("boundary", "minor_radius"),
    "geometric_axis_r": ("boundary", "geometric_axis_r"),
    "elongation": ("boundary", "elongation"),
    "triangularity_upper": ("boundary", "triangularity_upper"),
    "triangularity_lower": ("boundary", "triangularity_lower"),
    "power_nbi": ("heating_current_drive", "power_nbi"),
    "power_ic": ("heating_current_drive", "power_ic"),
    "power_lh": ("heating_current_drive", "power_lh"),
}


def build_summary(factory, time, signals):
    """`summary` IDS at its own full native (unprocessed-file) 0D-signal resolution.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        time: (n,) time base [s] -- the unprocessed file's own 0D-signal
            sampling, independent of `equilibrium.time`/`core_profiles.time`.
        signals: `workflow.FINAL_0D_SIGNALS` name -> (n,) signal on `time`.
            Any subset; a signal that is absent, or NaN everywhere (how a
            device without it stages it), is left unset in the IDS. A name
            with no entry in `_SUMMARY_SIGNAL_PATHS` is an error, so a new
            FINAL_0D_SIGNALS entry cannot be dropped silently.

    Returns:
        The validated `summary` IDS.
    """
    unknown = sorted(set(signals) - set(_SUMMARY_SIGNAL_PATHS))
    if unknown:
        raise ValueError(f"No summary IDS field mapped for 0D signal(s) {unknown}")

    sm = factory.summary()
    sm.ids_properties.homogeneous_time = 1
    sm.time = np.asarray(time, dtype=float)
    for name, (group, field) in _SUMMARY_SIGNAL_PATHS.items():
        if name not in signals:
            continue
        values = np.asarray(signals[name], dtype=float)
        if not np.any(np.isfinite(values)):
            continue
        getattr(getattr(sm, group), field).value = values
    sm.validate()
    return sm


# ---------------------------------------------------------------------------
# wall IDS: shot-level (limiter contour is time-invariant)
# ---------------------------------------------------------------------------


def build_wall(factory, time, eq_mp):
    """`wall` IDS, single time slice: limiter contour only.

    Sourced from `eq_mp.raw['rlim']`/`['zlim']` -- the real limiter contour
    the `.geqdsk` file carries natively. `eq_mp` is any one of the shot's
    already-built `megpy.Equilibrium` objects (see `build_equilibrium`'s
    `eq_mp_by_time`): the limiter is time-invariant, so this needs only one.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        time: The equilibrium time `eq_mp` was built from [s].
        eq_mp: A `megpy.Equilibrium` from `build_equilibrium`.

    Returns:
        The validated `wall` IDS.
    """
    wall = factory.wall()
    wall.ids_properties.homogeneous_time = 1
    wall.time = np.atleast_1d(np.asarray(time, dtype=float))
    wall.description_2d.resize(1)
    d2d = wall.description_2d[0]
    d2d.limiter.unit.resize(1)
    unit = d2d.limiter.unit[0]
    unit.outline.r = np.asarray(eq_mp.raw["rlim"], dtype=float)
    unit.outline.z = np.asarray(eq_mp.raw["zlim"], dtype=float)
    wall.validate()
    return wall


def write_ids(ids, output_dir, dd_version=DD_VERSION, overwrite=False):
    """Writes one IDS to `<output_dir>/<ids name>.nc` via
    `imas.DBEntry`/`.put()` -- the same call every other IMAS netCDF writer
    (fusio's included) bottoms out in.

    Args:
        ids: A populated, validated IDS object (e.g. from
            `build_equilibrium`); its own `metadata.name` picks the file name.
        output_dir: Directory the `.nc` file goes in (created if missing).
        dd_version: IMAS data dictionary version to write with.
        overwrite: Rewrite the file if it already exists.
    """
    output_dir = Path(output_dir)
    ids_path = output_dir / f"{ids.metadata.name}.nc"
    if ids_path.exists() and not overwrite:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with imas.DBEntry(str(ids_path), "w", dd_version=dd_version) as entry:
        entry.put(ids)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def build_imas_from_shot(
    shot: int,
    fit_ds: xr.Dataset,
    unprocessed_ds: xr.Dataset,
    geqdsk_dir: Path | str,
    dd_version: str = DD_VERSION,
):
    """Builds one shot's `equilibrium`/`core_profiles`/`summary`/`wall` IDS
    from its fit results and unprocessed data.

    `core_profiles` gets electrons + a single hydrogenic main ion (see
    `build_core_profiles`) -- no Zeff/impurity composition; see the module
    docstring above for the standalone script that adds those afterward.

    Args:
        shot: Shot number.
        fit_ds: `_shot_fit_dataset()`'s output for this shot
            (`fit_shots_dir/<shot>.nc`) -- `t_e`/`n_e`/`t_e_fit_status`/
            `n_e_fit_status` on `(shot, TIME_DIM, rho)`, real slice times in
            `TIME_COORD`.
        unprocessed_ds: This shot's unprocessed data
            (`01_unprocessed/<shot>.nc`) -- needs `ip` and
            `workflow.FINAL_EQUILIBRIUM_SIGNALS`, all on the shot's common
            time grid (the equilibrium signals NaN outside a real EFIT
            reconstruction time). Every other `workflow.FINAL_0D_SIGNALS`
            entry present is written to `summary` (see `build_summary`).
        geqdsk_dir: Directory to write this shot's per-equilibrium-time
            `.geqdsk` files into (see `geqdsk_writer.write_geqdsk`).
        dd_version: IMAS data dictionary version.

    Returns:
        The four populated, validated IDS objects, in the order
        `equilibrium`, `core_profiles`, `summary`, `wall`; write each with
        `write_ids`.
    """
    from transport_validation_datasets.workflow import (
        FINAL_0D_SIGNALS,
        FINAL_EQUILIBRIUM_SIGNALS,
        TIME_COORD,
        TIME_DIM,
        USABLE_FIT_STATUSES,
    )

    factory = imas.IDSFactory(version=dd_version)
    geqdsk_dir = Path(geqdsk_dir)

    if "shot" in unprocessed_ds.dims:
        unprocessed_ds = unprocessed_ds.squeeze("shot", drop=True)

    # FINAL_EQUILIBRIUM_SIGNALS lives on the shot's common time grid (same as
    # the 0D signals), NaN outside a real EFIT reconstruction time -- not a
    # compact per-EFIT-time array. Filter down to the real reconstruction
    # times first (any one scalar field, e.g. simagx, is finite exactly
    # where every field in the block is, since they're all written together
    # for the same reconstruction).
    eq_valid = np.flatnonzero(np.isfinite(unprocessed_ds["simagx"].to_numpy()))
    eq_ds = unprocessed_ds[list(FINAL_EQUILIBRIUM_SIGNALS)].isel(time=eq_valid)
    eq_times = eq_ds["time"].to_numpy().astype(float)
    geqdsk_paths = [
        write_geqdsk(
            geqdsk_dir / f"{shot}_eq{i:04d}.geqdsk",
            eq_ds.isel(time=i),
            shot=shot,
            time_ms=int(round(eq_times[i] * 1000)),
        )
        for i in range(eq_times.size)
    ]
    eq, eq_mp_by_time, derived_by_time = build_equilibrium(factory, eq_times, geqdsk_paths)

    usable = (
        fit_ds["t_e_fit_status"].isin(list(USABLE_FIT_STATUSES))
        & fit_ds["n_e_fit_status"].isin(list(USABLE_FIT_STATUSES))
    ).squeeze("shot", drop=True)
    fit_ds = fit_ds.squeeze("shot", drop=True).isel({TIME_DIM: usable.values})
    ts_times = fit_ds[TIME_COORD].to_numpy().astype(float)
    rho = fit_ds["rho"].to_numpy().astype(float)
    te_arr = fit_ds["t_e"].to_numpy().astype(float)
    ne_arr = fit_ds["n_e"].to_numpy().astype(float)
    te_err_arr = fit_ds["t_e_error"].to_numpy().astype(float) if "t_e_error" in fit_ds else None
    ne_err_arr = fit_ds["n_e_error"].to_numpy().astype(float) if "n_e_error" in fit_ds else None

    unprocessed_time = unprocessed_ds["time"].to_numpy().astype(float)
    # Whichever FINAL_0D_SIGNALS the shot has; only `ip` is required (for
    # core_profiles.global_quantities.ip).
    signal_0d = {
        name: unprocessed_ds[name].to_numpy().astype(float)
        for name in FINAL_0D_SIGNALS
        if name in unprocessed_ds
    }
    if "ip" not in signal_0d:
        raise KeyError(f"Shot {shot}'s unprocessed data has no `ip` signal")

    slices = []
    for i, t in enumerate(ts_times):
        eq_idx = int(np.argmin(np.abs(eq_times - t)))
        eq_t = float(eq_times[eq_idx])
        derived = derived_by_time[eq_t]
        idx0d = int(np.argmin(np.abs(unprocessed_time - t)))

        # Maps Thomson's rho (RHO_DEFINITION: normalized outboard-midplane
        # minor radius) onto this equilibrium's own psi_norm grid via its
        # own rho_tor_norm -- both are standard "rho" definitions that agree
        # closely for weakly-shaped equilibria and differ mainly by shaping
        # corrections, the same class of approximation the original
        # cmod_to_imas pipeline already made (its own rho was sqrt(toroidal
        # flux), not this geometric definition, either).
        psin_neo = np.interp(rho, derived.rho_tor_norm, derived.psi_norm)

        slices.append(
            ShotExportSlice(
                time=float(t),
                t_e=te_arr[i],
                n_e=ne_arr[i],
                rho=rho,
                psi_axis=derived.psi_axis,
                psi_boundary=derived.psi_boundary,
                psin_neo=psin_neo,
                ip=signal_0d["ip"][idx0d],
                t_e_error=None if te_err_arr is None else te_err_arr[i],
                n_e_error=None if ne_err_arr is None else ne_err_arr[i],
            )
        )

    cp = build_core_profiles(factory, slices)
    sm = build_summary(factory, unprocessed_time, signal_0d)
    wall = build_wall(factory, eq_times[0], eq_mp_by_time[float(eq_times[0])])
    return eq, cp, sm, wall
