"""Tests for the rho_tor_norm mapping of the Thomson channels (machine/generic.py).

The checks use a concentric circular equilibrium, psi_N = ((R - R0)^2 + Z^2) / a^2,
with a linear q = q0 + (q1 - q0) psi_N.
Its normalized toroidal flux has a closed form:

    Phi_N(psi_N) = (q0 psi_N + (q1 - q0) psi_N^2 / 2) / (q0 + (q1 - q0) / 2)
"""

import numpy as np
import pytest
import xarray as xr
from scipy.special import xlogy

from transport_validation_datasets.machine.generic import (
    SECANT_PSI_N,
    cumulative_q_integral,
    map_ts_channels_to_rho_tor_norm,
    phi_n_map,
    psi_n_from_rho_tor_norm,
    rho_tor_norm_from_psi_n,
)

R0 = 1.0
Q0, Q1 = 1.0, 4.0
N_PSI = 129
PSI_N_GRID = np.linspace(0.0, 1.0, N_PSI)
QPSI = Q0 + (Q1 - Q0) * PSI_N_GRID
# integral_0^1 q dpsi_N
Q_TOTAL = Q0 + (Q1 - Q0) / 2.0
R_GRID = np.linspace(0.5, 1.5, 201)
Z_GRID = np.linspace(-0.4, 0.4, 161)


def phi_n_closed_form(psi_n):
    return (Q0 * psi_n + (Q1 - Q0) * psi_n**2 / 2.0) / Q_TOTAL


# Flux surfaces at uniform rho_pol, so psi_N = rho_pol^2 is not uniform
PSI_N_SURFACES = np.linspace(0.0, 1.0, 41) ** 2


def log_q(psi_n):
    """q = 1.2 - 0.8 ln(1 - psi_N), the form a diverted q takes near the LCFS."""
    return 1.2 - 0.8 * np.log(1.0 - psi_n)


def log_q_integral(psi_n):
    """Closed-form integral of log_q from 0 to psi_N, finite at psi_N = 1."""
    one_minus = 1.0 - psi_n
    return 1.2 * psi_n + 0.8 * (xlogy(one_minus, one_minus) - one_minus + 1.0)


def diverted_qpsi(psi_n_surfaces):
    """log_q on the surfaces, infinite at the LCFS."""
    qpsi = np.full(psi_n_surfaces.size, np.inf)
    qpsi[:-1] = log_q(psi_n_surfaces[:-1])
    return qpsi


def circular_psirz(minor_radius):
    rr, zz = np.meshgrid(R_GRID, Z_GRID, indexing="ij")
    return ((rr - R0) ** 2 + zz**2) / minor_radius**2


def circular_shot(
    eq_rows,
    minor_radii,
    ts_rows,
    channel_r,
    channel_z,
    nan_qpsi_rows=(),
    n_t=40,
    flux_sign=1.0,
    boundary_radius_fraction=1.0,
):
    # One shot on a 1 kHz grid, circular reconstructions at eq_rows and Thomson at ts_rows.
    # psi_axis is 0 and psi_boundary is flux_sign, so psirz is flux_sign psi_N.
    # flux_sign -1 has psi decreasing outward with a negative q, as on MAST.
    # The boundary contour is the psi_N = 1 circle scaled by boundary_radius_fraction.
    time = np.arange(n_t) * 1e-3
    contour_angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rbdry = np.full((n_t, contour_angle.size), np.nan)
    zbdry = np.full((n_t, contour_angle.size), np.nan)
    psirz = np.full((n_t, R_GRID.size, Z_GRID.size), np.nan)
    simagx = np.full(n_t, np.nan)
    sibdry = np.full(n_t, np.nan)
    qpsi = np.full((n_t, N_PSI), np.nan)
    for row, minor_radius in zip(eq_rows, minor_radii):
        psirz[row] = flux_sign * circular_psirz(minor_radius)
        simagx[row] = 0.0
        sibdry[row] = flux_sign
        if row not in nan_qpsi_rows:
            qpsi[row] = flux_sign * QPSI
        contour_radius = boundary_radius_fraction * minor_radius
        rbdry[row] = R0 + contour_radius * np.cos(contour_angle)
        zbdry[row] = contour_radius * np.sin(contour_angle)
    channel_shape = (n_t, channel_r.size)
    ts_r = np.full(channel_shape, np.nan)
    ts_z = np.full(channel_shape, np.nan)
    ts_te = np.full(channel_shape, np.nan)
    for row in ts_rows:
        ts_r[row] = channel_r
        ts_z[row] = channel_z
        ts_te[row] = 1.0
    ts_dims = ("time", "ts_channel")
    return xr.Dataset(
        {
            "psirz": (("time", "r_grid", "z_grid"), psirz),
            "simagx": ("time", simagx),
            "sibdry": ("time", sibdry),
            "qpsi": (("time", "psi_idx"), qpsi),
            "rbdry": (("time", "boundary_idx"), rbdry),
            "zbdry": (("time", "boundary_idx"), zbdry),
            "ts_channel_r": (ts_dims, ts_r),
            "ts_channel_z": (ts_dims, ts_z),
            "ts_channel_t_e": (ts_dims, ts_te),
            "ts_channel_n_e": (ts_dims, np.full(channel_shape, np.nan)),
        },
        coords={"time": time, "r_grid": R_GRID, "z_grid": Z_GRID},
    )


class TestToroidalFlux:
    def test_q_integral_matches_closed_form(self):
        q_integral = cumulative_q_integral(PSI_N_GRID, QPSI)

        expected = Q0 * PSI_N_GRID + (Q1 - Q0) * PSI_N_GRID**2 / 2.0
        np.testing.assert_allclose(q_integral, expected, atol=1e-12)

    def test_inside_lcfs_matches_closed_form(self):
        psi_n = PSI_N_GRID[[0, 5, 38, 90, 122, 128]]

        rho_tor_norm = rho_tor_norm_from_psi_n(psi_n, QPSI, "secant")

        phi_n = phi_n_closed_form(psi_n)
        np.testing.assert_allclose(rho_tor_norm, np.sqrt(phi_n), atol=1e-12)

    @pytest.mark.parametrize("sol_extension", ["secant", "tangent"])
    def test_sol_extension_is_linear_in_psi_n_from_the_lcfs(self, sol_extension):
        # tangent: dPhi_N/dpsi_N at the LCFS, q(1) / integral_0^1 q dpsi_N
        # secant: the average slope over the last stretch inside the LCFS
        if sol_extension == "tangent":
            slope = Q1 / Q_TOTAL
        else:
            phi_n_start = phi_n_closed_form(SECANT_PSI_N)
            slope = (1.0 - phi_n_start) / (1.0 - SECANT_PSI_N)
        psi_n = np.array([1.0, 1.1, 1.3, 1.6])

        rho_tor_norm = rho_tor_norm_from_psi_n(psi_n, QPSI, sol_extension)

        expected_phi_n = 1.0 + slope * (psi_n - 1.0)
        np.testing.assert_allclose(rho_tor_norm**2, expected_phi_n, rtol=1e-4)

    @pytest.mark.parametrize("sol_extension", ["secant", "tangent"])
    def test_monotone_across_the_lcfs(self, sol_extension):
        psi_n = np.linspace(0.9, 1.5, 601)

        rho_tor_norm = rho_tor_norm_from_psi_n(psi_n, QPSI, sol_extension)

        assert np.all(np.diff(rho_tor_norm) > 0.0)

    def test_nan_stays_nan_and_negative_psi_n_is_the_axis(self):
        psi_n = np.array([np.nan, -1e-3])

        rho_tor_norm = rho_tor_norm_from_psi_n(psi_n, QPSI, "secant")

        assert np.isnan(rho_tor_norm[0])
        assert rho_tor_norm[1] == 0.0

    @pytest.mark.parametrize("sol_extension", ["secant", "tangent"])
    def test_inverse_round_trips_across_the_lcfs(self, sol_extension):
        # The IMAS export places the fit grid on psi through the inverse
        psi_n = np.array([0.0, 0.03, 0.4, 0.97, 1.0, 1.05, 1.4, np.nan])

        rho_tor_norm = rho_tor_norm_from_psi_n(psi_n, QPSI, sol_extension)
        psi_n_back = psi_n_from_rho_tor_norm(rho_tor_norm, QPSI, sol_extension)

        # The inverse interpolates a dense Phi_N table inside the LCFS
        np.testing.assert_allclose(psi_n_back, psi_n, atol=1e-8)


class TestPhiNMap:
    def test_constant_q_is_psi_n_for_either_sign(self):
        # Constant q to the LCFS (a limited plasma) makes Phi_N = psi_N, and the sign of q cancels
        psi_n = np.linspace(0.0, 1.0, 101)
        qpsi = np.full(PSI_N_SURFACES.size, 3.0)

        phi_n = phi_n_map(PSI_N_SURFACES, qpsi, "secant").phi_n(psi_n)
        phi_n_flipped_q = phi_n_map(PSI_N_SURFACES, -qpsi, "secant").phi_n(psi_n)

        np.testing.assert_allclose(phi_n, psi_n, atol=1e-12)
        np.testing.assert_allclose(phi_n_flipped_q, psi_n, atol=1e-12)

    def test_diverted_tail_matches_log_q(self):
        # q diverging at the LCFS goes through the analytic tail past the last finite surface (psi_N ~ 0.95)
        psi_n = np.concatenate(
            [np.linspace(0.0, 0.9, 10), np.linspace(0.951, 0.9999, 50), [1.0]]
        )
        qpsi = diverted_qpsi(PSI_N_SURFACES)

        phi_n = phi_n_map(PSI_N_SURFACES, qpsi, "secant").phi_n(psi_n)

        phi_n_expected = log_q_integral(psi_n) / log_q_integral(1.0)
        # Simpson over 41 surfaces is good to a few 1e-4 against q steepening toward the LCFS
        np.testing.assert_allclose(phi_n, phi_n_expected, rtol=1e-3)
        assert phi_n[-1] == 1.0

    def test_finite_q_at_the_lcfs_stays_close_to_the_diverted_integral(self):
        # EFIT writes a finite q(1), here q a little inside the LCFS, on a uniform 129-point grid
        psi_n_grid = np.linspace(0.0, 1.0, 129)
        qpsi = log_q(np.minimum(psi_n_grid, 1.0 - 1e-3))
        psi_n = np.linspace(0.0, 1.0, 201)

        rho_tor_norm = np.sqrt(phi_n_map(psi_n_grid, qpsi, "secant").phi_n(psi_n))

        rho_tor_norm_expected = np.sqrt(log_q_integral(psi_n) / log_q_integral(1.0))
        np.testing.assert_allclose(rho_tor_norm, rho_tor_norm_expected, atol=2e-3)

    def test_rejects_unusable_q_profiles(self):
        qpsi_good = diverted_qpsi(PSI_N_SURFACES)
        qpsi_nan = qpsi_good.copy()
        qpsi_nan[5] = np.nan
        # Diverging next to the axis leaves too few finite surfaces to fit the tail to
        qpsi_axis = qpsi_good.copy()
        qpsi_axis[2:] = np.inf
        # A tail whose q falls toward the LCFS
        qpsi_falling = qpsi_good.copy()
        qpsi_falling[-5:-1] = [4.0, 3.5, 3.0, 2.5]
        qpsi_sign_change = np.full(PSI_N_SURFACES.size, 2.0)
        qpsi_sign_change[10] = -2.0

        for qpsi in [qpsi_nan, qpsi_axis, qpsi_falling, qpsi_sign_change]:
            assert phi_n_map(PSI_N_SURFACES, qpsi, "secant") is None

    def test_rising_q_matches_closed_form_and_continues_along_the_secant(self):
        # q = 1 + 3 psi^2 gives Phi_N = (psi + psi^3) / 2 inside the LCFS
        psi_n = np.linspace(-0.01, 1.2, 242)
        psi_n_grid = np.linspace(0.0, 1.0, 129)
        qpsi = 1.0 + 3.0 * psi_n_grid**2

        phi_n = phi_n_map(psi_n_grid, qpsi, "secant").phi_n(psi_n)

        psi_n_clipped = np.maximum(psi_n, 0.0)
        mask_inside = psi_n_clipped <= 1.0
        phi_n_inside = (psi_n_clipped + psi_n_clipped**3) / 2
        np.testing.assert_allclose(
            phi_n[mask_inside], phi_n_inside[mask_inside], atol=1e-6
        )
        phi_n_at_secant_start = (SECANT_PSI_N + SECANT_PSI_N**3) / 2
        secant_slope = (1.0 - phi_n_at_secant_start) / (1.0 - SECANT_PSI_N)
        phi_n_outside = 1.0 + secant_slope * (psi_n[~mask_inside] - 1.0)
        np.testing.assert_allclose(phi_n[~mask_inside], phi_n_outside, rtol=1e-6)
        assert np.all(np.diff(phi_n[psi_n >= 0]) > 0)

    @pytest.mark.parametrize("sol_extension", ["secant", "tangent"])
    def test_inverse_round_trips_across_the_lcfs(self, sol_extension):
        psi_n = np.array([0.0, 0.03, 0.4, 0.97, 1.0, 1.05, 1.4, np.nan])
        if sol_extension == "secant":
            qpsi = diverted_qpsi(PSI_N_SURFACES)
        else:
            qpsi = log_q(PSI_N_SURFACES * 0.99)
        phi_n_mapping = phi_n_map(PSI_N_SURFACES, qpsi, sol_extension)

        phi_n = phi_n_mapping.phi_n(psi_n)
        psi_n_back = phi_n_mapping.psi_n(phi_n)

        # The inverse interpolates a dense Phi_N table inside the LCFS
        np.testing.assert_allclose(psi_n_back, psi_n, atol=1e-7)


class TestMapChannels:
    @pytest.mark.parametrize("flux_sign", [1.0, -1.0])
    def test_channels_map_through_the_flux_map(self, flux_sign):
        # Midplane channels on both sides, off-midplane ones, one on the LCFS and one in the SOL
        minor_radius = 0.3
        channel_psi_n = np.array([0.1, 0.5, 0.9, 0.4, 0.8, 1.0, 1.2])
        channel_distance = minor_radius * np.sqrt(channel_psi_n)
        angle = np.array([0.0, 0.0, np.pi, 0.5, -1.0, np.pi, 0.0])
        channel_r = R0 + channel_distance * np.cos(angle)
        channel_z = channel_distance * np.sin(angle)
        ds = circular_shot(
            [10], [minor_radius], [10], channel_r, channel_z, flux_sign=flux_sign
        )

        ts_times, rho_tor_norm = map_ts_channels_to_rho_tor_norm(ds, "tangent")

        expected = rho_tor_norm_from_psi_n(channel_psi_n, QPSI, "tangent")
        np.testing.assert_allclose(ts_times, [0.010])
        np.testing.assert_allclose(rho_tor_norm[0], expected, atol=1e-4)

    def test_slice_maps_through_nearest_reconstruction_in_reach(self):
        # Reconstructions every 5 ms, then a 20 ms gap, so the reach is 7.5 ms.
        # t = 3 ms takes the 5 ms reconstruction.
        # t = 12 ms is nearest the 10 ms one, which has no q profile,
        # so it takes the 5 ms one, 7 ms away.
        # t = 20 ms is out of reach of every usable reconstruction.
        # The unusable one still counts for the clock, without it the reach would be 22.5 ms.
        minor_radii = [0.30, 0.25, 0.28, 0.32]
        channel_r = np.array([R0 + 0.2])
        channel_z = np.array([0.0])
        ds = circular_shot(
            [0, 5, 10, 30],
            minor_radii,
            [3, 12, 20],
            channel_r,
            channel_z,
            nan_qpsi_rows=(10,),
        )

        _, rho_tor_norm = map_ts_channels_to_rho_tor_norm(ds, "secant")

        psi_n_at_5ms = (0.2 / minor_radii[1]) ** 2
        expected = rho_tor_norm_from_psi_n(np.array([psi_n_at_5ms]), QPSI, "secant")
        np.testing.assert_allclose(rho_tor_norm[0], expected, atol=1e-4)
        np.testing.assert_allclose(rho_tor_norm[1], expected, atol=1e-4)
        assert np.isnan(rho_tor_norm[2]).all()

    def test_channel_outside_the_boundary_below_separatrix_flux_is_unmapped(self):
        # A contour at 0.8 of the psi_N = 1 circle stands in for the LCFS around an X-point:
        # the channel at psi_N 0.81 (0.9 of the radius) lies outside it, like a channel under an X-point,
        # while the ones inside the contour and the SOL channel still map
        minor_radius = 0.3
        channel_psi_n = np.array([0.1, 0.4, 0.81, 1.2])
        channel_r = R0 + minor_radius * np.sqrt(channel_psi_n)
        channel_z = np.zeros(channel_psi_n.size)
        ds = circular_shot(
            [10],
            [minor_radius],
            [10],
            channel_r,
            channel_z,
            boundary_radius_fraction=0.8,
        )

        _, rho_tor_norm = map_ts_channels_to_rho_tor_norm(ds, "secant")

        expected = rho_tor_norm_from_psi_n(channel_psi_n, QPSI, "secant")
        np.testing.assert_allclose(
            rho_tor_norm[0, [0, 1, 3]], expected[[0, 1, 3]], atol=1e-4
        )
        assert np.isnan(rho_tor_norm[0, 2])
