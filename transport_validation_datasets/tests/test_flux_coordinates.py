"""Correctness checks for gp_fitting/coordinates.py.

Uses a synthetic concentric-circular equilibrium (psi_n(R, Z) = ((R-R0)^2 +
(Z-Z0)^2) / a^2) with a linear q(psi_n) profile, chosen because both have
closed-form toroidal-flux and midplane-mapping solutions to check against:

    rho (normalized minor radius) == sqrt(psi_n) exactly, since the outboard
    midplane radius at psi_n is r = a*sqrt(psi_n) for this geometry.

    phi_norm(psi_n) = (q0*psi_n + (qedge-q0)*psi_n^2/2) / (same at psi_n=1),
    with d(phi_norm)/d(psi_n) = q(psi_n) / (that same normalizing integral).
"""

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets.gp_fitting.coordinates import (
    _JACOBIAN_EPS,
    _MISSING_Q_JACOBIAN_FALLBACK,
    build_midplane_flux_map,
    coordinates_from_psi_norm,
    toroidal_flux_norm_profile,
    transform_gradient,
)
from transport_validation_datasets.machine.generic import (
    map_ts_channels_to_flux_coordinates,
    map_ts_channels_to_rho,
)

R0, A, Z0 = 1.0, 0.3, 0.0
Q0, QEDGE = 1.0, 4.0
N_PSI = 129


@pytest.fixture
def circular_equilibrium():
    r_grid = np.linspace(0.5, 1.5, 401)
    z_grid = np.linspace(-0.4, 0.4, 321)
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="ij")
    psirz_slice = ((rr - R0) ** 2 + (zz - Z0) ** 2) / A**2  # simagx=0, sibdry=1 => == psi_n
    psi_norm_grid = np.linspace(0.0, 1.0, N_PSI)
    qpsi = Q0 + (QEDGE - Q0) * psi_norm_grid
    return {
        "r_grid": r_grid,
        "z_grid": z_grid,
        "psirz_slice": psirz_slice,
        "simagx": 0.0,
        "sibdry": 1.0,
        "zmagx": 0.0,
        "qpsi": qpsi,
    }


def phi_norm_closed_form(psi_n):
    unnorm = Q0 * psi_n + (QEDGE - Q0) * psi_n**2 / 2
    total = Q0 * 1.0 + (QEDGE - Q0) * 0.5
    return unnorm / total


def test_toroidal_flux_norm_profile_matches_closed_form(circular_equilibrium):
    psi_norm_grid = np.linspace(0.0, 1.0, N_PSI)
    got = toroidal_flux_norm_profile(circular_equilibrium["qpsi"])
    want = phi_norm_closed_form(psi_norm_grid)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_toroidal_flux_norm_profile_missing_qpsi_is_nan():
    assert np.all(np.isnan(toroidal_flux_norm_profile(np.full(N_PSI, np.nan))))


def test_midplane_flux_map_axis_and_lcfs(circular_equilibrium):
    fmap = build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    assert fmap is not None
    assert fmap.r_axis == pytest.approx(R0, abs=1e-2)
    assert fmap.r_lcfs_outboard == pytest.approx(R0 + A, abs=1e-2)


def test_midplane_flux_map_degenerate_equilibrium_returns_none(circular_equilibrium):
    assert build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        simagx=0.0,
        sibdry=0.0,  # zero flux range: degenerate
        zmagx=0.0,
    ) is None


def test_coordinates_from_psi_norm_matches_closed_form(circular_equilibrium):
    fmap = build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    psi_query = np.array([0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.99, 1.0])
    values, jac = coordinates_from_psi_norm(psi_query, fmap, circular_equilibrium["qpsi"])

    np.testing.assert_allclose(values.rho, np.sqrt(psi_query), atol=5e-3)
    np.testing.assert_allclose(values.sqrt_psi_norm, np.sqrt(psi_query), atol=1e-9)
    # phi_norm is a trapezoid-rule integral over N_PSI nodes plus linear
    # interpolation between them, so it carries discretization error against
    # the exact closed form (~1e-3 relative here, shrinking as N_PSI grows) --
    # unlike sqrt_psi_norm/rho above, which have no such discretization step.
    np.testing.assert_allclose(values.phi_norm, phi_norm_closed_form(psi_query), atol=2e-5)
    np.testing.assert_allclose(
        values.sqrt_phi_norm, np.sqrt(phi_norm_closed_form(psi_query)), atol=1e-4
    )

    q_at = Q0 + (QEDGE - Q0) * psi_query
    total_q = Q0 * 1.0 + (QEDGE - Q0) * 0.5
    np.testing.assert_allclose(jac.phi_norm, q_at / total_q, atol=1e-9)
    np.testing.assert_allclose(
        jac.sqrt_psi_norm[1:],
        0.5 / np.sqrt(psi_query[1:]),
        rtol=1e-3,
    )
    # away from the axis grid cell, rho's finite-difference Jacobian should
    # match sqrt(psi_norm)'s closed-form derivative (rho == sqrt(psi_norm) here)
    np.testing.assert_allclose(
        jac.rho[1:],
        0.5 / np.sqrt(psi_query[1:]),
        rtol=0.05,
    )


def test_coordinates_from_psi_norm_axis_jacobians_are_finite(circular_equilibrium):
    fmap = build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    values, jac = coordinates_from_psi_norm(
        np.array([0.0]), fmap, circular_equilibrium["qpsi"]
    )
    assert np.isfinite(jac.sqrt_psi_norm[0])
    assert np.isfinite(jac.sqrt_phi_norm[0])
    assert np.isfinite(jac.rho[0])
    assert jac.sqrt_psi_norm[0] == pytest.approx(0.5 / np.sqrt(_JACOBIAN_EPS))


def test_coordinates_from_psi_norm_missing_equilibrium_stays_nan(circular_equilibrium):
    values, jac = coordinates_from_psi_norm(
        np.array([0.1, 0.5]), None, circular_equilibrium["qpsi"]
    )
    assert np.all(np.isnan(values.rho))
    assert np.all(np.isnan(jac.rho))


def test_coordinates_from_psi_norm_missing_qpsi_falls_back(circular_equilibrium):
    fmap = build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    psi_query = np.array([0.1, 0.5])
    values, jac = coordinates_from_psi_norm(psi_query, fmap, np.full(N_PSI, np.nan))
    assert np.all(np.isnan(values.phi_norm))
    assert np.all(np.isnan(values.sqrt_phi_norm))
    np.testing.assert_allclose(jac.phi_norm, _MISSING_Q_JACOBIAN_FALLBACK)
    np.testing.assert_allclose(jac.sqrt_phi_norm, _MISSING_Q_JACOBIAN_FALLBACK)


def test_transform_gradient_round_trip_is_self_consistent(circular_equilibrium):
    """quantity(psi_norm) = psi_norm**2 has an exact d/d(psi_norm) = 2*psi_norm
    and, since rho == sqrt(psi_norm) here, an exact d/d(rho) = 4*rho**3 --
    two independent closed forms the chain-rule transform must connect."""
    fmap = build_midplane_flux_map(
        circular_equilibrium["psirz_slice"],
        circular_equilibrium["r_grid"],
        circular_equilibrium["z_grid"],
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    psi_query = np.array([0.05, 0.2, 0.5, 0.8])
    values, jac = coordinates_from_psi_norm(psi_query, fmap, circular_equilibrium["qpsi"])

    grad_wrt_psi_norm = 2 * psi_query
    grad_wrt_rho = transform_gradient(grad_wrt_psi_norm, jac.psi_norm, jac.rho)
    np.testing.assert_allclose(grad_wrt_rho, 4 * values.rho**3, rtol=0.05)

    grad_back = transform_gradient(grad_wrt_rho, jac.rho, jac.psi_norm)
    np.testing.assert_allclose(grad_back, grad_wrt_psi_norm, rtol=1e-3)


def test_transform_gradient_floors_near_zero_jacobian():
    grad = np.array([1.0])
    jac_from = np.array([1.0])
    jac_to = np.array([0.0])
    result = transform_gradient(grad, jac_from, jac_to)
    assert np.isfinite(result[0])
    assert abs(result[0]) == pytest.approx(1.0 / _JACOBIAN_EPS)


def test_rho_matches_map_ts_channels_to_rho(circular_equilibrium):
    """flux_coordinates' rho must agree with the rho GP fitting is staged on."""
    r_grid = circular_equilibrium["r_grid"]
    z_grid = circular_equilibrium["z_grid"]
    psirz_slice = circular_equilibrium["psirz_slice"]

    # A few outboard-midplane-ish channel positions inside the LCFS.
    ts_channel_r = np.array([1.0, 1.1, 1.2, 1.29])
    ts_channel_z = np.array([0.0, 0.0, 0.0, 0.0])
    n_ch = ts_channel_r.size

    ds_shot = xr.Dataset(
        data_vars={
            "psirz": (("time", "r_grid", "z_grid"), psirz_slice[None, :, :]),
            "simagx": ("time", np.array([circular_equilibrium["simagx"]])),
            "sibdry": ("time", np.array([circular_equilibrium["sibdry"]])),
            "zmagx": ("time", np.array([circular_equilibrium["zmagx"]])),
            "ts_channel_r": (("time", "ts_channel"), np.tile(ts_channel_r, (1, 1))),
            "ts_channel_z": (("time", "ts_channel"), np.tile(ts_channel_z, (1, 1))),
            "ts_channel_t_e": (
                ("time", "ts_channel"),
                np.ones((1, n_ch)),
            ),
            "ts_channel_n_e": (
                ("time", "ts_channel"),
                np.ones((1, n_ch)),
            ),
        },
        coords={"time": [0.0], "r_grid": r_grid, "z_grid": z_grid},
    )

    _, rho_pipeline = map_ts_channels_to_rho(ds_shot)
    _, rho_pipeline2, psi_norm_pipeline = map_ts_channels_to_flux_coordinates(ds_shot)
    np.testing.assert_allclose(rho_pipeline, rho_pipeline2)

    fmap = build_midplane_flux_map(
        psirz_slice,
        r_grid,
        z_grid,
        circular_equilibrium["simagx"],
        circular_equilibrium["sibdry"],
        circular_equilibrium["zmagx"],
    )
    # channels sit on the midplane at Z=0, so psi_n at each channel is just
    # ((r-R0)/a)^2 for this geometry.
    psi_n_ch = ((ts_channel_r - R0) / A) ** 2
    values, _ = coordinates_from_psi_norm(psi_n_ch, fmap, circular_equilibrium["qpsi"])

    np.testing.assert_allclose(values.rho, rho_pipeline[0], atol=1e-3)
    np.testing.assert_allclose(psi_n_ch, psi_norm_pipeline[0], atol=1e-3)


def test_phi_norm_extends_linearly_beyond_lcfs(circular_equilibrium):
    """Toroidal flux is undefined in the SOL, but real SOL channels
    (psi_norm > 1) must keep distinct, ordered positions: phi_norm extends
    linearly with the edge slope q(1)/total_q, continuous in value and
    derivative through the LCFS."""
    qpsi = circular_equilibrium["qpsi"]
    psi_query = np.array([0.98, 1.0, 1.02, 1.05, 1.10])
    values, jac = coordinates_from_psi_norm(psi_query, None, qpsi)

    total_q = Q0 * 1.0 + (QEDGE - Q0) * 0.5
    # Exactly 1 at the LCFS, then the closed-form linear extension.
    assert values.phi_norm[1] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        values.phi_norm[2:],
        1.0 + (psi_query[2:] - 1.0) * QEDGE / total_q,
        atol=1e-9,
    )
    np.testing.assert_allclose(values.sqrt_phi_norm, np.sqrt(values.phi_norm), atol=1e-12)
    # Strictly increasing through and beyond the LCFS -- no clamp pile-up.
    assert np.all(np.diff(values.phi_norm) > 0)
    assert np.all(np.diff(values.sqrt_phi_norm) > 0)
    # The Jacobian beyond the LCFS is the same edge slope the extension uses.
    np.testing.assert_allclose(jac.phi_norm[1:], QEDGE / total_q, atol=1e-9)
