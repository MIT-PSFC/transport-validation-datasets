"""Builds and writes a shot's `equilibrium`/`core_profiles`/`summary`/`wall` IDS.

Each shot's IMAS IDS set is built from its fit results and unprocessed data,
N time slices per shot, straight from the two per-shot files
(fit_shots_dir/<shot>.nc, 01_unprocessed/<shot>.nc) -- no intermediate
consolidated scenario file. Design notes:

  - `equilibrium` gets one `time_slice` per real EFIT reconstruction time in
    the shot's unprocessed data (its own, full native time base -- not
    thinned to or deduped against the Thomson slice times).
  - `summary` is written at its own full native (1 kHz) 0D-signal
    resolution, for the same reason, and carries every
    `store_schema.DATASET_0D_SIGNALS` entry the shot has (see
    `_SUMMARY_SIGNAL_PATHS` for where each one lands).
  - `core_profiles` gets one `profiles_1d` per usable Thomson slice time,
    electrons (with the GP fit's 1-sigma uncertainties in the DD's
    `*_error_upper` fields; `*_error_lower` is left unset, which the IMAS
    convention reads as a symmetric error) + a single hydrogenic
    main ion (Zeff=1, n_D=n_e). Zeff/impurity
    composition are deliberately NOT computed here, and this package holds no
    code for them at all -- that physics lives in a standalone
    post-processing script kept entirely outside this repo (to avoid a TORAX
    dependency clash -- see this project's plan notes), which computes them
    afterward from already-written IMAS output and writes a new
    `core_profiles.nc`. This module and `DataWorkflow.export_to_imas()` never
    call it -- that decoupling is deliberate.
  - No edge-physics modeling (see this project's plan notes) --
    `_find_x_point` below is the one small piece in that direction: it
    classifies `equilibrium.time_slice.boundary.type` (diverted vs.
    limited), a core equilibrium-IDS field, and nothing beyond that.
  - No derived flux-surface geometry: the `equilibrium` IDS
    carries only what the EQDSK fields themselves supply (1D psi-grid
    profiles, the 2D psi map, boundary/limiter contours, global scalars)
    plus `phi`/`rho_tor` (a pure 1D integral of the EQDSK's own q profile,
    also needed internally to place `core_profiles` on psi) and the
    diverted/limited boundary classification (`_find_x_point`, scipy-only).
    Everything derivable only by tracing flux surfaces (area/volume,
    `r_inboard`/`r_outboard`, `j_phi`, elongation/triangularity, the
    `gm1`...`gm9` flux-surface averages) is deliberately NOT written here:
    that degree of processing belongs in an isolated post-processing step
    run on the saved IMAS output, not in the conversion itself.

Each IDS is built as a real `imas` library object
(`imas.IDSFactory(...).<ids>()`, field assignment, `.validate()`) and written
straight to `<output_dir>/<ids>.nc` via `imas.DBEntry(...).put()`
(`write_ids`) -- deliberately NOT through `fusio`'s dotted-key `xr.Dataset`
round trip (`imas.util.to_xarray` -> `imas_io().write()`). That route ends
in the very same `DBEntry.put()` call, but its array-of-structures shape
inference silently keeps the nested per-slice arrays
(`profiles_1d[i].ion[...]`, `time_slice[i].profiles_2d[...]`) only for the
first time slice, emptying them on every later one (confirmed against fusio
0.4.4 on synthetic 4-slice IDS). Writing the built IDS directly avoids that
and needs no `fusio` dependency.
"""

from dataclasses import dataclass
from pathlib import Path

import eqdsk
import imas
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

from transport_validation_datasets.imas_export.geqdsk_writer import write_geqdsk
from transport_validation_datasets.machine.generic import (
    cumulative_q_integral,
    psi_n_from_rho_tor_norm,
    usable_reconstructions,
)


def _latest_dd_version() -> str:
    """Get the newest data dictionary version the installed imas package ships.

    The export's default DD version tracks the installed imas-python rather
    than a hardcoded number, so upgrading the package upgrades the written
    DD (the target COCOS follows along through _target_cocos). Falls back
    to the DD 4.0 baseline if the lookup API is ever unavailable.

    Returns:
        The version string, e.g. "4.1.1".
    """
    try:
        from imas.dd_zip import latest_dd_version

        return str(latest_dd_version())
    except Exception:
        return "4.0.0"


DD_VERSION = _latest_dd_version()


def _target_cocos(dd_version: str) -> int:
    """Get the IMAS convention of a data dictionary version.

    The data dictionary switched its own convention from COCOS 11 to
    COCOS 17 at DD 4.0, so the equilibrium IDS targets whichever matches
    the version being written.

    Args:
        dd_version: IMAS data dictionary version string, e.g. "4.0.0".

    Returns:
        17 for DD 4.0 and later, 11 before.
    """
    return 11 if int(str(dd_version).split(".", 1)[0]) < 4 else 17


def _sigma_bp(cocos: int) -> int:
    """Get the sigma_Bp of a COCOS index, for the phi integral.

    Args:
        cocos: COCOS index.

    Returns:
        -1 for the conventions whose psi decreases along Ip (3, 4, 7, 8,
        and their e_Bp=1 partners 13, 14, 17, 18), +1 for the rest.
    """
    return -1 if cocos % 10 in (3, 4, 7, 8) else 1


# core_profiles' one ion species (see build_core_profiles):
# deuterium, with the nominal mass number 2.0 rather than 2.014.
_MAIN_ION_NAME = "D"
_MAIN_ION_A = 2.0
_MAIN_ION_Z_N = 1

# A true X-point is a magnetic null (B_pol = 0) sitting on the LCFS. Below
# this fraction of the LCFS's typical (median) B_pol, the minimum found by
# `_find_x_point` is treated as a genuine null rather than just the locally
# weakest field on an otherwise smooth (limited) boundary.
_X_POINT_BPOL_RATIO_THRESHOLD = 0.1


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
        t_e: Electron temperature on `rho_tor_norm` [eV].
        n_e: Electron density on `rho_tor_norm` [m^-3].
        rho_tor_norm: The fit grid t_e/n_e are given on.
        psi_axis: Matched equilibrium's psi at the magnetic axis (target COCOS).
        psi_boundary: Matched equilibrium's psi at the boundary (target COCOS).
        psi_n: `rho_tor_norm` mapped back onto the matched equilibrium's normalized psi,
            for placing `profiles_1d.grid.psi`.
        ip: Nearest-time plasma current [A].
        t_e_error: 1-sigma uncertainty of `t_e` [eV], or None if the fit
            file carries none.
        n_e_error: 1-sigma uncertainty of `n_e` [m^-3], or None likewise.
    """

    time: float
    t_e: np.ndarray
    n_e: np.ndarray
    rho_tor_norm: np.ndarray
    psi_axis: float
    psi_boundary: float
    psi_n: np.ndarray
    ip: float
    t_e_error: np.ndarray | None = None
    n_e_error: np.ndarray | None = None


# ---------------------------------------------------------------------------
# diverted/limited boundary classification (scipy-only, from the EQDSK's own
# 2D psi map)
# ---------------------------------------------------------------------------


def _spline_scalar(spline, r, z):
    """`RectBivariateSpline(..., grid=False)` at one scalar `(r, z)` point.

    Returns:
        The spline value as a plain float, regardless of whether the
        installed scipy returns a 0-d array, a shape-(1,) array, or a bare
        float for scalar inputs.
    """
    return float(np.ravel(spline(r, z, grid=False))[0])


def _find_x_point(eqi, bp_spline):
    """Locates the magnetic null (X-point) nearest the LCFS, if any.

    Classifies the equilibrium as diverted or limited, for
    `equilibrium.time_slice.boundary.type`. Uses the physical definition of an X-point directly:
    B_pol vanishes there, so the minimum of B_pol along
    the LCFS boundary contour sits at (or very near) any true X-point,
    refined by locally minimizing the B_pol spline from that starting point.

    Args:
        eqi: `eqdsk.EQDSKInterface` supplying the LCFS contour
            (`xbdry`/`zbdry`).
        bp_spline: `RectBivariateSpline` of B_pol(R, Z) (any overall scale;
            only ratios of its values are used).

    Returns:
        (r_x, z_x, diverted).
    """
    bp_boundary = bp_spline(eqi.xbdry, eqi.zbdry, grid=False)
    imin = int(np.argmin(bp_boundary))
    r0, z0 = eqi.xbdry[imin], eqi.zbdry[imin]
    result = minimize(
        lambda p: _spline_scalar(bp_spline, p[0], p[1]) ** 2,
        x0=[r0, z0],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-12},
    )
    r_x, z_x = float(result.x[0]), float(result.x[1])
    # A true X-point sits at (or just off) the LCFS's own B_pol minimum.
    # With no null nearby (a limited plasma) the unconstrained refinement
    # walks down the smooth B_pol landscape all the way to the magnetic
    # axis -- a genuine null, but not an X-point -- so a refinement that
    # left the starting point's neighborhood is discarded in favor of the
    # boundary minimum itself.
    minor_radius = 0.5 * float(np.max(eqi.xbdry) - np.min(eqi.xbdry))
    if np.hypot(r_x - r0, z_x - z0) > 0.2 * minor_radius:
        r_x, z_x = float(r0), float(z0)
    bp_min = _spline_scalar(bp_spline, r_x, z_x)
    bp_typical = float(np.median(bp_boundary))
    diverted = bool(bp_min / bp_typical < _X_POINT_BPOL_RATIO_THRESHOLD)
    return r_x, z_x, diverted


def _diverted(eqi):
    """Whether `eqi` is diverted (True) or limited (False); see `_find_x_point`.

    B_pol is built directly from the EQDSK's own 2D psi map as
    `|grad psi| / R` via a bicubic spline's analytic derivatives. The
    COCOS-dependent `2*pi` factor between that and the physical B_pol is
    deliberately left out: `_find_x_point` only ever compares B_pol values
    to each other, so any overall scale cancels.

    Args:
        eqi: `eqdsk.EQDSKInterface` supplying the psi(R, Z) map and LCFS.

    Returns:
        True if a genuine X-point sits on the LCFS.
    """
    psi_spline = RectBivariateSpline(eqi.x, eqi.z, eqi.psi, kx=3, ky=3, s=0)
    dpsi_dr = psi_spline(eqi.x, eqi.z, dx=1, dy=0)
    dpsi_dz = psi_spline(eqi.x, eqi.z, dx=0, dy=1)
    bpol_rz = np.hypot(dpsi_dr, dpsi_dz) / np.asarray(eqi.x, dtype=float)[:, np.newaxis]
    bp_spline = RectBivariateSpline(eqi.x, eqi.z, bpol_rz)
    _, _, diverted = _find_x_point(eqi, bp_spline)
    return diverted


# ---------------------------------------------------------------------------
# equilibrium IDS, populated one time slice at a time
# ---------------------------------------------------------------------------


@dataclass
class _EquilibriumTimeDerived:
    """Per-equilibrium-time quantities returned by `_populate_equilibrium_time_slice`.

    These are needed alongside the IDS fields it writes directly into `ts`.
    `qpsi` is this equilibrium time's q profile.
    build_imas_from_shot maps the fit grid back onto psi through it (`ShotExportSlice.psi_n`).
    """

    qpsi: np.ndarray
    bcentr: float
    psi_axis: float
    psi_boundary: float


def _populate_equilibrium_time_slice(ts, eqi, sigma_bp: int):
    """Fills one `equilibrium.time_slice[i]` from one equilibrium reconstruction.

    `eqi` is an `eqdsk.EQDSKInterface` already converted to target-COCOS.
    Writes only what the EQDSK fields themselves supply:
    the 1D psi-grid profiles, the 2D psi map, the boundary contour, and the
    global scalars, plus `phi`/`rho_tor` (a pure 1D integral of the EQDSK's
    own q profile -- in the e_Bp=1 conventions
    `dphi/dpsi = sigma_Bp * sigma_rho_theta_phi * q`, and both IMAS targets
    (11 and 17) have `sigma_rho_theta_phi = +1`, so
    `phi = sigma_bp * integral(q dpsi)` on the file's own uniform psi grid)
    and the
    diverted/limited classification (`_diverted`). Derived flux-surface
    geometry (area/volume, `r_inboard`/`r_outboard`, `j_phi`, shaping, the
    `gm*` averages) is deliberately not computed here -- see the module
    docstring.

    Args:
        ts: The `equilibrium.time_slice[i]` node to fill.
        eqi: `eqdsk.EQDSKInterface`, target-COCOS.
        sigma_bp: sigma_Bp of the target COCOS (see _sigma_bp), setting the
            sign of the phi integral.

    Returns:
        An `_EquilibriumTimeDerived` with this equilibrium time's
        `qpsi`/`bcentr`/`psi_axis`/`psi_boundary`.
    """
    psi_axis = float(eqi.psimag)
    psi_boundary = float(eqi.psibdry)
    psi = np.linspace(psi_axis, psi_boundary, int(eqi.nx))
    psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis)
    qpsi = np.asarray(eqi.qpsi, dtype=float)

    ts.profiles_1d.psi = psi
    ts.profiles_1d.psi_norm = psi_norm
    ts.profiles_1d.f = np.asarray(eqi.fpol, dtype=float)
    ts.profiles_1d.pressure = np.asarray(eqi.pressure, dtype=float)
    ts.profiles_1d.f_df_dpsi = np.asarray(eqi.ffprime, dtype=float)
    ts.profiles_1d.dpressure_dpsi = np.asarray(eqi.pprime, dtype=float)
    ts.profiles_1d.q = qpsi

    bcentr = float(eqi.bcentre)
    # dphi/dpsi = sigma_Bp * sigma_rho_theta_phi * q in the e_Bp=1 (11-18) conventions,
    # and both IMAS targets (11, 17) have sigma_rho_theta_phi = +1,
    # so phi = sigma_bp * integral(q dpsi).
    # That lands phi on the sign of B0, its physical direction, for either field polarity under either target.
    q_integral = cumulative_q_integral(psi_norm, qpsi)
    phi = sigma_bp * (psi_boundary - psi_axis) * q_integral
    # rho_tor needs a reference field to carry units of meters; phi and
    # bcentr share a sign, so the ratio is positive. A zero vacuum field is
    # nonphysical, but fall back to the dimensionless sqrt(psi_norm) rather
    # than divide by zero.
    rho_tor = np.sqrt(phi / (np.pi * bcentr)) if abs(bcentr) > 0 else np.sqrt(psi_norm)
    rho_tor_a = rho_tor[-1] if rho_tor[-1] > 0.0 else 1.0
    ts.profiles_1d.phi = phi
    ts.profiles_1d.rho_tor = rho_tor
    ts.profiles_1d.rho_tor_norm = rho_tor / rho_tor_a
    ts.profiles_1d.dpsi_drho_tor = np.gradient(psi, rho_tor)

    ts.global_quantities.magnetic_axis.r = float(eqi.xmag)
    ts.global_quantities.magnetic_axis.z = float(eqi.zmag)
    ts.global_quantities.psi_axis = psi_axis
    ts.global_quantities.psi_boundary = psi_boundary
    ts.global_quantities.ip = float(eqi.cplasma)

    rbdry = np.asarray(eqi.xbdry, dtype=float)
    zbdry = np.asarray(eqi.zbdry, dtype=float)
    ts.boundary.outline.r = rbdry
    ts.boundary.outline.z = zbdry
    # C-Mod EFIT boundary contours are zero-padded to a fixed size; (0, 0)
    # pairs are unphysical (R >= 0.4 m) and must not enter the R extent.
    real = ~((rbdry == 0.0) & (zbdry == 0.0))
    ts.boundary.minor_radius = float((rbdry[real].max() - rbdry[real].min()) / 2.0)
    ts.boundary.type = 1 if _diverted(eqi) else 0

    ts.profiles_2d.resize(1)
    p2d = ts.profiles_2d[0]
    p2d.grid_type.name = "rectangular"
    p2d.grid.dim1 = np.asarray(eqi.x, dtype=float)
    p2d.grid.dim2 = np.asarray(eqi.z, dtype=float)
    p2d.psi = np.asarray(eqi.psi, dtype=float)

    return _EquilibriumTimeDerived(
        qpsi=qpsi,
        bcentr=bcentr,
        psi_axis=psi_axis,
        psi_boundary=psi_boundary,
    )


def build_equilibrium(
    factory, times, geqdsk_paths, source_cocos: int, target_cocos: int | None = None
):
    """`equilibrium` IDS: one `time_slice` per real EFIT reconstruction time.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        times: (n_eq,) real EFIT reconstruction times [s].
        geqdsk_paths: (n_eq,) `.geqdsk` file paths, one per time in `times`
            (see `geqdsk_writer.write_geqdsk`).
        source_cocos: COCOS convention of the files,
            the unprocessed file's cocos attribute (see machine.generic.cocos_from_signs).
        target_cocos: COCOS convention the IDS is written in. None takes
            the convention matching DD_VERSION (see _target_cocos); pass
            the one matching the factory's DD version when it differs.

    Returns:
        (eq, eqi_first, derived_by_time): the validated `equilibrium` IDS;
        `eqi_first` is the first time's `eqdsk.EQDSKInterface` (reused by
        `build_wall` for the time-invariant limiter contour);
        `derived_by_time` maps each time to its `_EquilibriumTimeDerived`,
        for `core_profiles.profiles_1d.grid.psi` placement (see
        `build_imas_from_shot`).
    """
    if target_cocos is None:
        target_cocos = _target_cocos(DD_VERSION)
    sigma_bp = _sigma_bp(target_cocos)
    eq = factory.equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = np.asarray(times, dtype=float)
    eq.time_slice.resize(len(times))

    eqi_first = None
    derived_by_time = {}
    bcentr_per_time = np.empty(len(times))
    r0 = None
    for i, (t, geqdsk_path) in enumerate(zip(times, geqdsk_paths)):
        eqi = eqdsk.EQDSKInterface.from_file(
            str(geqdsk_path),
            from_cocos=source_cocos,
            to_cocos=target_cocos,
        )
        ts = eq.time_slice[i]
        derived = _populate_equilibrium_time_slice(ts, eqi, sigma_bp)
        derived_by_time[t] = derived
        bcentr_per_time[i] = derived.bcentr
        if eqi_first is None:
            eqi_first = eqi
            r0 = float(eqi.xcentre)

    eq.vacuum_toroidal_field.r0 = r0 if r0 is not None else 0.0
    eq.vacuum_toroidal_field.b0 = bcentr_per_time

    eq.validate()
    return eq, eqi_first, derived_by_time


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
        p1d.grid.rho_tor_norm = s.rho_tor_norm
        p1d.grid.psi = s.psi_n * (s.psi_boundary - s.psi_axis) + s.psi_axis
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
        p1d.zeff = np.ones_like(s.rho_tor_norm, dtype=float)
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


# `store_schema.DATASET_0D_SIGNALS` name -> (summary sub-structure, field) it is written to.
# Each target is a `summary_dynamic` node whose `.value` holds the time series.
# Every DATASET_0D_SIGNALS entry has a home here (checked against DD 4.1.1).
# A signal whose `ref` attr is under /summary lands on that same node.
# The others (/equilibrium, /bolometer) name where the device's value comes from,
# and land on summary's node for the same quantity.
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
    "power_nbi": ("heating_current_drive", "power_launched_nbi"),
    "power_ic": ("heating_current_drive", "power_launched_ic"),
    "power_lh": ("heating_current_drive", "power_launched_lh"),
    "power_ec": ("heating_current_drive", "power_launched_ec"),
}


def build_summary(factory, time, signals, r0):
    """`summary` IDS at its own full native (unprocessed-file) 0D-signal resolution.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        time: (n,) time base [s] -- the unprocessed file's own 0D-signal
            sampling, independent of `equilibrium.time`/`core_profiles.time`.
        signals: `store_schema.DATASET_0D_SIGNALS` name -> (n,) signal on `time`.
            Any subset; a signal that is absent, or NaN everywhere (how a
            device without it stages it), is left unset in the IDS. A name
            with no entry in `_SUMMARY_SIGNAL_PATHS` is an error, so a new
            DATASET_0D_SIGNALS entry cannot be dropped silently.
        r0: The reference major radius b0 is given at [m], a constant of the shot.

    beta_tor_norm is stored with B_geo = b0 r0 / geometric_axis_r (store_schema),
    and IMAS normalizes it with b0 at r0.
    beta_N goes as 1 / B at fixed pressure, so it is written times B_geo / b0 = r0 / geometric_axis_r.

    Returns:
        The validated `summary` IDS.

    Raises:
        ValueError: If a signal name has no `_SUMMARY_SIGNAL_PATHS` entry,
            or beta_tor_norm comes without geometric_axis_r.
    """
    unknown = sorted(set(signals) - set(_SUMMARY_SIGNAL_PATHS))
    if unknown:
        raise ValueError(f"No summary IDS field mapped for 0D signal(s) {unknown}")
    signals_imas = dict(signals)
    if "beta_tor_norm" in signals:
        if "geometric_axis_r" not in signals:
            raise ValueError(
                "beta_tor_norm needs geometric_axis_r to go from B_geo to the IMAS b0 at r0"
            )
        geometric_axis_r = np.asarray(signals["geometric_axis_r"], dtype=float)
        beta_tor_norm_b_geo = np.asarray(signals["beta_tor_norm"], dtype=float)
        signals_imas["beta_tor_norm"] = beta_tor_norm_b_geo * r0 / geometric_axis_r

    sm = factory.summary()
    sm.ids_properties.homogeneous_time = 1
    sm.time = np.asarray(time, dtype=float)
    sm.global_quantities.r0.value = float(r0)
    for name, (group, field) in _SUMMARY_SIGNAL_PATHS.items():
        if name not in signals_imas:
            continue
        values = np.asarray(signals_imas[name], dtype=float)
        if not np.any(np.isfinite(values)):
            continue
        getattr(getattr(sm, group), field).value = values
    sm.validate()
    return sm


# ---------------------------------------------------------------------------
# wall IDS: shot-level (limiter contour is time-invariant)
# ---------------------------------------------------------------------------


def build_wall(factory, time, eqi):
    """`wall` IDS, single time slice: limiter contour only.

    Sourced from `eqi.xlim`/`.zlim` -- the real limiter contour the
    `.geqdsk` file carries natively (COCOS-invariant geometry). `eqi` is
    any one of the shot's already-loaded `eqdsk.EQDSKInterface` objects
    (see `build_equilibrium`'s `eqi_first`): the limiter is time-invariant,
    so this needs only one.

    Args:
        factory: `imas.IDSFactory` to build the IDS from.
        time: The equilibrium time `eqi` was loaded from [s].
        eqi: An `eqdsk.EQDSKInterface` from `build_equilibrium`.

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
    unit.outline.r = np.asarray(eqi.xlim, dtype=float)
    unit.outline.z = np.asarray(eqi.zlim, dtype=float)
    wall.validate()
    return wall


def write_ids(ids, output_dir, dd_version=DD_VERSION, overwrite=False):
    """Writes one IDS to `<output_dir>/<ids name>.nc`.

    Goes via `imas.DBEntry`/`.put()` -- the same call every other IMAS
    netCDF writer (fusio's included) bottoms out in.

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
    """Builds one shot's `equilibrium`/`core_profiles`/`summary`/`wall` IDS.

    The IDS set is built from the shot's fit results and unprocessed data.
    `core_profiles` gets electrons + a single hydrogenic main ion (see
    `build_core_profiles`) -- no Zeff/impurity composition; see the module
    docstring above for the standalone script that adds those afterward.

    Args:
        shot: Shot number.
        fit_ds: `_shot_fit_dataset()`'s output for this shot
            (`fit_shots_dir/<shot>.nc`) -- `t_e`/`n_e`/`t_e_fit_status`/
            `n_e_fit_status` on `(shot, TIME_DIM, rho_tor_norm)`, real slice times in
            `TIME_COORD`, and the `sol_extension` attribute the channels were staged with.
        unprocessed_ds: This shot's unprocessed data
            (`01_unprocessed/<shot>.nc`) -- needs `ip`, the `cocos` attribute, and
            `workflow.DATASET_EQUILIBRIUM_SIGNALS`, all on the shot's common
            time grid (the equilibrium signals NaN outside a real EFIT
            reconstruction time). Every other `store_schema.DATASET_0D_SIGNALS`
            entry present is written to `summary` (see `build_summary`).
        geqdsk_dir: Directory to write this shot's per-equilibrium-time
            `.geqdsk` files into (see `geqdsk_writer.write_geqdsk`).
        dd_version: IMAS data dictionary version.

    Returns:
        The four populated, validated IDS objects, in the order
        `equilibrium`, `core_profiles`, `summary`, `wall`; write each with
        `write_ids`.

    Raises:
        KeyError: If the shot's unprocessed data has no `ip` signal.
    """
    from transport_validation_datasets.store_schema import DATASET_0D_SIGNALS
    from transport_validation_datasets.workflow import (
        DATASET_EQUILIBRIUM_SIGNALS,
        TIME_COORD,
        TIME_DIM,
        usable_slice_mask,
    )

    factory = imas.IDSFactory(version=dd_version)
    geqdsk_dir = Path(geqdsk_dir)

    if "shot" in unprocessed_ds.dims:
        unprocessed_ds = unprocessed_ds.squeeze("shot", drop=True)

    # DATASET_EQUILIBRIUM_SIGNALS lives on the shot's common time grid (same as the 0D signals),
    # NaN outside a real EFIT reconstruction time. Filter down to the usable reconstructions first.
    eq_usable = usable_reconstructions(unprocessed_ds)
    eq_valid = np.flatnonzero(eq_usable)
    eq_ds = unprocessed_ds[list(DATASET_EQUILIBRIUM_SIGNALS)].isel(time=eq_valid)
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
    source_cocos = int(unprocessed_ds.attrs["cocos"])
    target_cocos = _target_cocos(dd_version)
    eq, eqi_first, derived_by_time = build_equilibrium(
        factory, eq_times, geqdsk_paths, source_cocos, target_cocos=target_cocos
    )

    usable = usable_slice_mask(fit_ds)
    fit_ds = fit_ds.squeeze("shot", drop=True).isel({TIME_DIM: usable})
    ts_times = fit_ds[TIME_COORD].to_numpy().astype(float)
    rho_tor_norm = fit_ds["rho_tor_norm"].to_numpy().astype(float)
    sol_extension = fit_ds.attrs["sol_extension"]
    te_arr = fit_ds["t_e"].to_numpy().astype(float)
    ne_arr = fit_ds["n_e"].to_numpy().astype(float)
    te_err_arr = (
        fit_ds["t_e_error"].to_numpy().astype(float) if "t_e_error" in fit_ds else None
    )
    ne_err_arr = (
        fit_ds["n_e_error"].to_numpy().astype(float) if "n_e_error" in fit_ds else None
    )

    unprocessed_time = unprocessed_ds["time"].to_numpy().astype(float)
    # Whichever DATASET_0D_SIGNALS the shot has; only `ip` is required (for
    # core_profiles.global_quantities.ip).
    signal_0d = {
        name: unprocessed_ds[name].to_numpy().astype(float)
        for name in DATASET_0D_SIGNALS
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

        # The inverse of the map the channels were staged with, SOL extension included,
        # so the grid past the LCFS lands at psi_N > 1 rather than piling up at the boundary
        psi_n = psi_n_from_rho_tor_norm(rho_tor_norm, derived.qpsi, sol_extension)

        slices.append(
            ShotExportSlice(
                time=float(t),
                t_e=te_arr[i],
                n_e=ne_arr[i],
                rho_tor_norm=rho_tor_norm,
                psi_axis=derived.psi_axis,
                psi_boundary=derived.psi_boundary,
                psi_n=psi_n,
                ip=signal_0d["ip"][idx0d],
                t_e_error=None if te_err_arr is None else te_err_arr[i],
                n_e_error=None if ne_err_arr is None else ne_err_arr[i],
            )
        )

    cp = build_core_profiles(factory, slices)
    sm = build_summary(factory, unprocessed_time, signal_0d, unprocessed_ds.attrs["r0"])
    wall = build_wall(factory, eq_times[0], eqi_first)
    return eq, cp, sm, wall
