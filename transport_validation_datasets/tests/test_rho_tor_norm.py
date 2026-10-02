"""Tests for the rho_tor_norm mapping of the Thomson channels (machine/generic.py).

The checks use a concentric circular equilibrium, psi_N = ((R - R0)^2 + Z^2) / a^2,
with a linear q = q0 + (q1 - q0) psi_N.
Its normalized toroidal flux has a closed form:

    Phi_N(psi_N) = (q0 psi_N + (q1 - q0) psi_N^2 / 2) / (q0 + (q1 - q0) / 2)

TestCoordinateComparison is not a check.
It draws where the candidate radial coordinates put the channels and the anchors
on a real MAST and C-Mod equilibrium,
and how far the coordinates move through a MAST current ramp.
It runs only with -m slow.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.special import xlogy

from transport_validation_datasets import PACKAGE_ROOT
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
):
    # One shot on a 1 kHz grid, circular reconstructions at eq_rows and Thomson at ts_rows.
    # psi_axis is 0 and psi_boundary is flux_sign, so psirz is flux_sign psi_N.
    # flux_sign -1 has psi decreasing outward with a negative q, as on MAST.
    time = np.arange(n_t) * 1e-3
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


# Where the comparison caches its trimmed inputs and writes its figures
COMPARISON_DIR = PACKAGE_ROOT / "tests" / "test_outputs" / "rho_tor_norm"

# Signals the comparison needs from a shot's source dataset
COMPARISON_SIGNALS = (
    "psirz",
    "simagx",
    "sibdry",
    "rmagx",
    "zmagx",
    "qpsi",
    "rbdry",
    "zbdry",
    "ts_channel_r",
    "ts_channel_z",
    "ts_channel_t_e",
    "ts_channel_n_e",
)

# Seconds kept either side of the drawn time, a few reconstructions on both devices
COMPARISON_HALF_WINDOW = 0.015

COMPARISON_CASES = [
    pytest.param("mast", 28956, 0.179, id="mast"),
    pytest.param("cmod", 1160712015, 1.2, id="cmod"),
]

# A MAST current ramp and the current penetration after it.
# Ip rises from 240 to 580 kA by 0.125 s, then q(0) falls from 2.1 to 0.8 at constant Ip.
RAMP_SHOT = 28956
RAMP_WINDOW = (0.030, 0.400)
RAMP_SIGNALS = ("ip", "psirz", "simagx", "sibdry", "zmagx", "qpsi", "rbdry", "zbdry")
# The ramp ends where Ip first reaches this fraction of its peak
RAMP_END_IP_FRACTION = 0.95


def _lcfs_crossing_radius(
    r_from_axis: np.ndarray, psi_n_from_axis: np.ndarray
) -> float:
    """Find the midplane radius where psi_n first crosses 1, walking outward.

    Both arrays must be ordered starting at the axis and moving outward.

    Args:
        r_from_axis: Midplane radii, axis outward [m].
        psi_n_from_axis: Normalized poloidal flux at those radii.

    Returns:
        The linearly interpolated crossing radius [m], or NaN if psi_n never reaches 1.
    """
    above = psi_n_from_axis >= 1.0
    if not above.any():
        return np.nan
    idx = int(np.argmax(above))
    if idx == 0:
        return float(r_from_axis[0])
    r0, r1 = float(r_from_axis[idx - 1]), float(r_from_axis[idx])
    p0, p1 = float(psi_n_from_axis[idx - 1]), float(psi_n_from_axis[idx])
    if p1 == p0:
        return r1
    r_cross = r0 + (1.0 - p0) * (r1 - r0) / (p1 - p0)
    return r_cross


def _refine_axis_radius(
    r_grid: np.ndarray, psi_n_mid: np.ndarray, i_axis: int
) -> float:
    """Refine the magnetic axis radius from the midplane psi_n minimum.

    When defining rho = (r - r_axis) / (r_lcfs - r_axis), must know where the axis is.
    EFIT grid can be coarse (a few cm), so may get rho errors ~ 5% (worst in the core).
    Here, use a simple 3-point parabola fit to refine the axis radius.

    Args:
        r_grid: Midplane radii [m].
        psi_n_mid: Normalized poloidal flux along the midplane.
        i_axis: Index of the psi_n_mid minimum.

    Returns:
        The refined axis radius [m].
    """
    r_axis = float(r_grid[i_axis])
    if 0 < i_axis < len(r_grid) - 1:
        p_m = psi_n_mid[i_axis - 1]
        p_0 = psi_n_mid[i_axis]
        p_p = psi_n_mid[i_axis + 1]
        curv = p_m - 2 * p_0 + p_p
        if curv > 0:
            r_axis += (
                0.5
                * (p_m - p_p)
                / curv
                * float(r_grid[i_axis + 1] - r_grid[i_axis - 1])
                / 2.0
            )
    return r_axis


def psi_n_map(ds_eq: xr.Dataset) -> np.ndarray:
    # (n_r, n_z) normalized poloidal flux of one reconstruction
    simagx = float(ds_eq["simagx"])
    sibdry = float(ds_eq["sibdry"])
    psirz = ds_eq["psirz"].transpose("r_grid", "z_grid").values
    return (psirz - simagx) / (sibdry - simagx)


def outboard_midplane_table(psi_n_grid, r_grid, z_grid, zmagx):
    """The flux map along the outboard midplane, as the previous rho mapping built it.

    The previous fit coordinate was the normalized outboard midplane minor radius,
    rho = (r_mid - r_axis) / (r_lcfs - r_axis),
    with each channel's psi_N inverted through this table to its midplane radius r_mid.

    Args:
        psi_n_grid: (n_r, n_z) normalized poloidal flux on the (r_grid, z_grid) grid.
        r_grid: (n_r,) major radii [m].
        z_grid: (n_z,) heights [m].
        zmagx: Height of the magnetic axis [m].

    Returns:
        (r_axis, r_lcfs, r_table, psi_n_table): the refined axis radius, the outboard LCFS radius,
        and the outboard radii with their psi_N, starting at the axis and non-decreasing in psi_N.
    """
    # psi_n along the midplane (z = magnetic axis height)
    psi_n_mid = np.array(
        [np.interp(zmagx, z_grid, psi_n_grid[j, :]) for j in range(len(r_grid))]
    )
    i_axis = int(np.argmin(psi_n_mid))
    r_axis = _refine_axis_radius(r_grid, psi_n_mid, i_axis)
    r_lcfs = _lcfs_crossing_radius(r_grid[i_axis:], psi_n_mid[i_axis:])
    # Anchor the table on the refined axis, where psi_n is 0 by definition,
    # and keep only the grid nodes outboard of it
    outboard = r_grid[i_axis:] > r_axis
    psi_n_table = np.concatenate([[0.0], psi_n_mid[i_axis:][outboard]])
    r_table = np.concatenate([[r_axis], r_grid[i_axis:][outboard]])
    keep = psi_n_table == np.maximum.accumulate(psi_n_table)
    return r_axis, r_lcfs, r_table[keep], psi_n_table[keep]


def fetch_source_dataset(device: str, shot: int) -> xr.Dataset:
    # Read one shot through the device workflow, skipping when its source is unreachable
    staging_dir = COMPARISON_DIR / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    shotlist_file = staging_dir / f"{device}_shotlist.txt"
    shotlist_file.write_text(f"{shot}\n")
    if device == "mast":
        from transport_validation_datasets.machine.mast.mast_dataset import (
            LEVEL2_PATH,
            MASTDataWorkflow,
            _store_path_exists,
        )

        if not _store_path_exists(f"{LEVEL2_PATH}/{shot}.zarr"):
            pytest.skip("no cached input and the MAST level 2 store is not reachable")
        workflow_cls = MASTDataWorkflow
    else:
        has_data = Path("/usr/local/mfe/ml_data_dump").exists()
        has_mdsplus = importlib.util.find_spec("MDSplus") is not None
        if not (has_data and has_mdsplus):
            pytest.skip("no cached input and C-Mod data is not available")
        from transport_validation_datasets.machine.cmod.cmod_dataset import (
            CModDataWorkflow,
        )

        workflow_cls = CModDataWorkflow
    workflow = workflow_cls(
        ds_name=f"{device}_coordinates",
        data_assembly_dir=staging_dir,
        shotlist_file=shotlist_file,
    )
    ds_source = workflow.get_source_dataset(shot)
    if ds_source is None:
        pytest.skip(f"{device} shot {shot} could not be read")
    return ds_source.squeeze("shot", drop=True)


def cached_input(device, shot, signals, window, cache_name) -> xr.Dataset:
    # Some signals of one shot over a time window, fetched once and cached.
    # Most rows hold no reconstruction, and compressed they cost almost nothing.
    path = COMPARISON_DIR / f"{cache_name}.nc"
    if not path.exists():
        ds_source = fetch_source_dataset(device, shot)
        t_start, t_end = window
        ds_window = ds_source[list(signals)].sel(time=slice(t_start, t_end))
        encoding = {name: {"zlib": True} for name in signals}
        ds_window.to_netcdf(path, encoding=encoding)
    with xr.open_dataset(path) as ds:
        return ds.load()


def device_settings(device: str):
    if device == "mast":
        from transport_validation_datasets.machine.mast.mast_dataset import (
            MASTSettings,
        )

        return MASTSettings()
    from transport_validation_datasets.machine.cmod.cmod_dataset import CModSettings

    return CModSettings()


def anchor_positions(settings) -> np.ndarray:
    # Every position any of the device's anchors sits at
    anchor_rows = [
        settings.te_value_anchors,
        settings.te_grad_anchors,
        settings.ne_value_anchors,
        settings.ne_grad_anchors,
    ]
    positions = [np.asarray(rows, dtype=float)[:, 0] for rows in anchor_rows]
    return np.unique(np.concatenate(positions))


@pytest.mark.slow  # fetches from the MAST store or MDSplus when nothing is cached
class TestCoordinateComparison:
    """Not a check: figures comparing the candidate radial coordinates on real equilibria."""

    @pytest.mark.parametrize(("device", "shot", "time"), COMPARISON_CASES)
    def test_draw(self, device, shot, time):
        """Where each coordinate puts the channels and the anchors.

        Left: the magnetic axis, the LCFS and the Thomson channels in (R, Z).
        Right: each coordinate along the outboard midplane,
        with the channels and the device's anchor positions placed on it.
        All four coordinates are functions of psi_N,
        so a channel lands at the same midplane radius in each.
        The anchors are fixed coordinate values, so they land at different radii.
        """
        window = (time - COMPARISON_HALF_WINDOW, time + COMPARISON_HALF_WINDOW)
        ds = cached_input(device, shot, COMPARISON_SIGNALS, window, f"{device}_{shot}")
        settings = device_settings(device)

        # The Thomson slice nearest the time, and the reconstruction nearest it
        has_ts = (ds["ts_channel_t_e"].notnull() | ds["ts_channel_n_e"].notnull()).any(
            "ts_channel"
        )
        ts_rows = np.flatnonzero(has_ts.values)
        ts_row_times = ds["time"].values[ts_rows]
        i_ts = int(ts_rows[np.argmin(np.abs(ts_row_times - time))])
        t_ts = float(ds["time"].values[i_ts])
        eq_rows = np.flatnonzero(np.isfinite(ds["simagx"].values))
        eq_row_times = ds["time"].values[eq_rows]
        i_eq = int(eq_rows[np.argmin(np.abs(eq_row_times - t_ts))])
        ds_eq = ds.isel(time=i_eq)
        ds_ts = ds.isel(time=i_ts)

        r_grid = ds["r_grid"].values
        z_grid = ds["z_grid"].values
        psi_n_grid = psi_n_map(ds_eq)
        qpsi = ds_eq["qpsi"].values
        zmagx = float(ds_eq["zmagx"])

        interp = RegularGridInterpolator(
            (r_grid, z_grid), psi_n_grid, bounds_error=False, fill_value=np.nan
        )
        channel_r = ds_ts["ts_channel_r"].values
        channel_z = ds_ts["ts_channel_z"].values
        channel_points = np.column_stack([channel_r, channel_z])
        channel_psi_n = interp(channel_points)

        r_axis, r_lcfs, r_table, psi_n_table = outboard_midplane_table(
            psi_n_grid, r_grid, z_grid, zmagx
        )
        on_table = np.isfinite(channel_psi_n) & (channel_psi_n <= psi_n_table[-1])
        channel_psi_n = channel_psi_n[on_table]
        channel_r_mid = np.interp(channel_psi_n, psi_n_table, r_table)
        minor_radius = r_lcfs - r_axis
        # The table is linear in R between grid nodes, as the previous mapping read it.
        # Sampling it finely keeps the curves from cutting corners where the SOL extension steepens.
        r_mid = np.linspace(r_table[0], r_table[-1], 600)
        psi_n_mid = np.interp(r_mid, r_table, psi_n_table)

        # Each coordinate along the outboard midplane, and at the channels
        coordinates = {
            "rho_tor_norm, tangent": (
                rho_tor_norm_from_psi_n(psi_n_mid, qpsi, "tangent"),
                rho_tor_norm_from_psi_n(channel_psi_n, qpsi, "tangent"),
            ),
            "rho_tor_norm, secant": (
                rho_tor_norm_from_psi_n(psi_n_mid, qpsi, "secant"),
                rho_tor_norm_from_psi_n(channel_psi_n, qpsi, "secant"),
            ),
            "rho_pol_norm": (np.sqrt(psi_n_mid), np.sqrt(channel_psi_n)),
            "midplane minor radius (previous)": (
                (r_mid - r_axis) / minor_radius,
                (channel_r_mid - r_axis) / minor_radius,
            ),
        }
        anchors = anchor_positions(settings)

        fig, (ax_rz, ax_mid) = plt.subplots(1, 2, figsize=(14, 6.5))
        rbdry = ds_eq["rbdry"].values
        zbdry = ds_eq["zbdry"].values
        # C-Mod pads the contour with zeros, MAST with NaN
        on_boundary = np.isfinite(rbdry) & np.isfinite(zbdry) & (rbdry > 0.0)
        ax_rz.plot(rbdry[on_boundary], zbdry[on_boundary], "k-", lw=1.5, label="LCFS")
        ax_rz.contour(
            r_grid,
            z_grid,
            psi_n_grid.T,
            levels=[0.2, 0.4, 0.6, 0.8],
            colors="0.75",
            linewidths=0.7,
        )
        ax_rz.plot(
            float(ds_eq["rmagx"]), zmagx, "k+", ms=12, mew=2, label="magnetic axis"
        )
        ax_rz.plot(channel_r, channel_z, "o", color="tab:red", ms=4, label="Thomson")
        ax_rz.axhline(zmagx, color="0.5", ls="--", lw=0.8, label="midplane")
        ax_rz.set_aspect("equal")
        ax_rz.set_xlabel("R [m]")
        ax_rz.set_ylabel("Z [m]")
        ax_rz.set_title(f"{device} {shot}  equilibrium t={float(ds_eq['time']):.3f} s")
        ax_rz.legend(loc="upper right", fontsize=8)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
        for color, (label, (mid_values, channel_values)) in zip(
            colors, coordinates.items()
        ):
            ax_mid.plot(r_mid, mid_values, "-", color=color, lw=1.5, label=label)
            ax_mid.plot(channel_r_mid, channel_values, "o", color=color, ms=4)
            reachable = anchors[anchors <= mid_values[-1]]
            anchor_r = np.interp(reachable, mid_values, r_mid)
            ax_mid.plot(anchor_r, reachable, "D", mfc="none", mec=color, ms=9, mew=1.5)
        for anchor in anchors:
            ax_mid.axhline(anchor, color="0.85", lw=0.7, zorder=0)
        ax_mid.axvline(r_lcfs, color="k", lw=1.0, label="LCFS")
        ax_mid.set_xlabel("outboard midplane R [m]")
        ax_mid.set_ylabel("coordinate value")
        ax_mid.set_ylim(0.0, anchors.max() + 0.1)
        ax_mid.set_title(
            f"Thomson t={t_ts:.3f} s: channels (dots) and anchors (diamonds)"
        )
        ax_mid.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        figure_path = COMPARISON_DIR / f"{device}_{shot}_coordinates.png"
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)

        assert figure_path.exists()

    def test_draw_ramp(self):
        """How far rho_pol_norm and rho_tor_norm move through a MAST current ramp.

        The plasma grows through the ramp, so both are compared at fixed outboard midplane r/a,
        not at fixed R.
        Top: Ip and q(0), the q profiles, and the LCFS, colored by time.
        Bottom: each coordinate against r/a,
        and its spread at fixed r/a over the ramp and over the flat top after it.
        """
        ds = cached_input(
            "mast", RAMP_SHOT, RAMP_SIGNALS, RAMP_WINDOW, f"mast_{RAMP_SHOT}_ramp"
        )
        r_grid = ds["r_grid"].values
        z_grid = ds["z_grid"].values
        qpsi_all = ds["qpsi"].transpose("time", "psi_idx").values
        mask_reconstructed = np.isfinite(ds["simagx"].values)
        mask_has_q = np.isfinite(qpsi_all).all(axis=1)
        eq_rows = np.flatnonzero(mask_reconstructed & mask_has_q)
        ds_eq = ds.isel(time=eq_rows)
        eq_times = ds_eq["time"].values
        qpsi = qpsi_all[eq_rows]
        psi_n_levels = np.linspace(0.0, 1.0, qpsi.shape[1])

        ip_kA = np.abs(ds["ip"].values) / 1e3
        ip_eq_kA = np.abs(ds_eq["ip"].values) / 1e3
        mask_ip_reached = ip_eq_kA >= RAMP_END_IP_FRACTION * ip_eq_kA.max()
        ramp_end = float(eq_times[np.argmax(mask_ip_reached)])
        mask_ramp = eq_times <= ramp_end

        # Each coordinate at fixed fractions r/a of the outboard midplane minor radius
        r_over_a = np.linspace(0.0, 1.0, 101)
        rho_pol_norm = np.full((eq_times.size, r_over_a.size), np.nan)
        rho_tor_norm = np.full((eq_times.size, r_over_a.size), np.nan)
        for i in range(eq_times.size):
            ds_row = ds_eq.isel(time=i)
            psi_n_grid = psi_n_map(ds_row)
            zmagx = float(ds_row["zmagx"])
            r_axis, r_lcfs, r_table, psi_n_table = outboard_midplane_table(
                psi_n_grid, r_grid, z_grid, zmagx
            )
            r_at_fraction = r_axis + r_over_a * (r_lcfs - r_axis)
            psi_n_at_fraction = np.interp(r_at_fraction, r_table, psi_n_table)
            rho_pol_norm[i] = np.sqrt(psi_n_at_fraction)
            # Inside the LCFS the two SOL extensions agree
            rho_tor_norm[i] = rho_tor_norm_from_psi_n(
                psi_n_at_fraction, qpsi[i], "secant"
            )

        fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
        (ax_ip, ax_q, ax_lcfs), (ax_pol, ax_tor, ax_spread) = axes
        time_norm = plt.Normalize(eq_times.min(), eq_times.max())
        time_cmap = plt.get_cmap("viridis")
        time_colors = time_cmap(time_norm(eq_times))

        ax_ip.plot(ds["time"].values, ip_kA, "k-", lw=1.2)
        ax_ip.axvspan(eq_times[0], ramp_end, color="0.9", label="ramp")
        ax_ip.set_xlabel("time [s]")
        ax_ip.set_ylabel("|Ip| [kA]")
        ax_ip.legend(loc="lower right", fontsize=8)
        ax_q0 = ax_ip.twinx()
        ax_q0.scatter(eq_times, qpsi[:, 0], c=time_colors, s=14)
        ax_q0.set_ylabel("q(0), dots")
        ax_ip.set_title(f"MAST {RAMP_SHOT}")

        rbdry_all = ds_eq["rbdry"].transpose("time", "boundary_idx").values
        zbdry_all = ds_eq["zbdry"].transpose("time", "boundary_idx").values
        for i in range(eq_times.size):
            color = time_colors[i]
            ax_q.plot(psi_n_levels, qpsi[i], color=color, lw=0.8)
            rbdry = rbdry_all[i]
            zbdry = zbdry_all[i]
            on_boundary = np.isfinite(rbdry) & np.isfinite(zbdry) & (rbdry > 0.0)
            ax_lcfs.plot(rbdry[on_boundary], zbdry[on_boundary], color=color, lw=0.8)
            ax_pol.plot(r_over_a, rho_pol_norm[i], color=color, lw=0.8)
            ax_tor.plot(r_over_a, rho_tor_norm[i], color=color, lw=0.8)
        ax_q.set_yscale("log")
        ax_q.set_xlabel("psi_N")
        ax_q.set_ylabel("q")
        ax_q.set_title("q profiles")
        ax_lcfs.set_aspect("equal")
        ax_lcfs.set_xlabel("R [m]")
        ax_lcfs.set_ylabel("Z [m]")
        ax_lcfs.set_title("LCFS")
        for ax, name in [(ax_pol, "rho_pol_norm"), (ax_tor, "rho_tor_norm")]:
            ax.plot([0.0, 1.0], [0.0, 1.0], color="0.6", ls=":", lw=1.0)
            ax.set_xlabel("outboard midplane r/a")
            ax.set_ylabel(name)
            ax.set_title(f"{name} at fixed r/a")

        phases = [("ramp", mask_ramp, "--"), ("flat top", ~mask_ramp, "-")]
        coordinates = [
            ("rho_pol_norm", rho_pol_norm, "tab:green"),
            ("rho_tor_norm", rho_tor_norm, "tab:orange"),
        ]
        for phase, mask_phase, style in phases:
            for name, values, color in coordinates:
                spread = np.ptp(values[mask_phase], axis=0)
                label = f"{name}, {phase}"
                ax_spread.plot(r_over_a, spread, style, color=color, label=label)
        ax_spread.set_xlabel("outboard midplane r/a")
        ax_spread.set_ylabel("max - min over the phase")
        ax_spread.set_title("spread at fixed r/a")
        ax_spread.legend(fontsize=8)

        time_mappable = plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
        fig.colorbar(time_mappable, ax=axes, label="time [s]", shrink=0.5)
        figure_path = COMPARISON_DIR / f"mast_{RAMP_SHOT}_ramp.png"
        fig.savefig(figure_path, dpi=110)
        plt.close(fig)

        assert figure_path.exists()
