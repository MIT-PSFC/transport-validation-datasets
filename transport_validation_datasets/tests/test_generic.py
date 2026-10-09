"""Unit tests for the device-independent signal helpers in machine/generic.py.

Synthetic arrays only, no MDSplus and no network, so these run anywhere.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.constants import mu_0
from scipy.integrate import quad

from transport_validation_datasets.machine.generic import (
    cocos_from_signs,
    hold_from_usable_reconstructions,
    lcfs_voltage,
    make_uniform_1kHz_timebase,
    normalized_beta,
    ohmic_power,
    poloidal_field_energy,
    signal_on_grid,
    smoothed_power,
    snap_to_grid,
    usable_reconstructions,
)

SHOT = 12345


def grid_ms(n: int) -> np.ndarray:
    # n grid points on the 1 kHz timebase, 0 ms .. (n-1) ms
    return np.round(np.arange(n) * 1e-3, 3).astype("float32")


def efit_dataset(times, values=None) -> xr.Dataset:
    # Minimal stand-in for a reconstruction: dim "idx" with time/shot coords
    times = np.asarray(times, dtype=float)
    if values is None:
        values = np.arange(1.0, times.size + 1.0)
    return xr.Dataset(
        {"simagx": ("idx", np.asarray(values, dtype=float))},
        coords={
            "time": ("idx", times),
            "shot": ("idx", np.repeat(SHOT, times.size)),
        },
    )


class TestSnapToGrid:
    def test_exact_grid_times_land_on_themselves(self):
        grid = grid_ms(11)
        ds = efit_dataset([0.002, 0.005, 0.009], [1.0, 2.0, 3.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[2] == 1.0
        assert out[5] == 2.0
        assert out[9] == 3.0

    def test_grid_times_without_slice_are_nan(self):
        grid = grid_ms(11)
        ds = efit_dataset([0.005], [7.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[5] == 7.0
        assert np.isnan(np.delete(out, 5)).all()

    def test_sub_step_jitter_absorbed(self):
        # Within half a step of a grid time, so it belongs to that grid time
        grid = grid_ms(11)
        ds = efit_dataset([0.00251, 0.00749], [1.0, 2.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[3] == 1.0  # 2.51 ms -> 3 ms
        assert out[7] == 2.0  # 7.49 ms -> 7 ms

    def test_ties_break_toward_later_grid_point(self):
        # An exact half-step must never feed the earlier grid time a future
        # reconstruction, so 0.5 ms on a 1 ms grid lands at 1 ms
        grid = grid_ms(11)
        ds = efit_dataset([0.0005], [4.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[1] == 4.0
        assert np.isnan(out[0])

    def test_slices_outside_grid_dropped(self):
        # A reconstruction from before or after the shot window is not a
        # measurement of either end of it,
        # so it must not pile onto grid[0] or grid[-1]
        grid = grid_ms(11)  # 0 .. 10 ms
        ds = efit_dataset([-0.02, 0.005, 0.05, 0.30], [1.0, 2.0, 3.0, 4.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[5] == 2.0
        assert np.isnan(out[0])
        assert np.isnan(out[-1])
        assert np.count_nonzero(np.isfinite(out)) == 1

    def test_half_step_outside_still_kept(self):
        # The drop is at half a grid step, so the boundary case still snaps in
        grid = grid_ms(11)
        ds = efit_dataset([-0.0005, 0.0105], [1.0, 2.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[0] == 1.0
        assert out[10] == 2.0

    def test_every_slice_outside_grid_all_nan(self):
        grid = grid_ms(11)
        ds = efit_dataset([-1.0, 2.0], [1.0, 2.0])

        out = snap_to_grid(ds, grid)

        assert np.isnan(out["simagx"].values).all()
        assert out.sizes["idx"] == grid.size

    def test_duplicate_slots_keep_last_slice(self):
        # Two reconstructions inside one grid step: the later one is the one
        # that grid time carries
        grid = grid_ms(11)
        ds = efit_dataset([0.0048, 0.0052], [1.0, 2.0])

        out = snap_to_grid(ds, grid)["simagx"].values

        assert out[5] == 2.0

    def test_time_and_shot_coords_rebuilt_on_grid(self):
        grid = grid_ms(11)
        ds = efit_dataset([0.005], [1.0])

        out = snap_to_grid(ds, grid)

        assert np.array_equal(out["time"].values, grid)
        assert (out["shot"].values == SHOT).all()
        assert out.sizes["idx"] == grid.size


class TestMakeUniform1kHzTimebase:
    def test_steps_exactly_one_millisecond(self):
        times = make_uniform_1kHz_timebase(0.01)

        assert times[0] == 0.0
        assert times[-1] == pytest.approx(0.01)
        assert np.allclose(np.diff(times), 1e-3)

    def test_max_time(self):
        # Rounds up, so the last time is at or past max_time
        times = make_uniform_1kHz_timebase(0.0104)

        assert times[-1] >= 0.0104


class TestCocosFromSigns:
    @pytest.mark.parametrize(
        "ip_sign, b0_sign, psi_sign, cocos",
        [(-1, -1, 1, 7), (1, 1, 1, 1), (1, -1, -1, 3)],
        ids=["cmod_normal_field", "cmod_reversed_field", "mast"],
    )
    def test_signs_give_the_device_cocos(self, ip_sign, b0_sign, psi_sign, cocos):
        # Three reconstructions, one non-converged, and q > 0 as both EFITs write it
        current = ip_sign * np.array([8e5, np.nan, 7e5])
        bcentr = b0_sign * np.array([5.4, np.nan, 5.3])
        simagx = np.array([-0.05, np.nan, -0.04])
        sibdry = simagx + psi_sign * 0.11
        qpsi = np.tile(np.linspace(1.0, 4.0, 5), (3, 1))
        qpsi[1] = np.nan

        assert cocos_from_signs(current, bcentr, simagx, sibdry, qpsi) == cocos

    @pytest.mark.filterwarnings("ignore:All-NaN slice")
    def test_no_reconstruction_falls_back_to_1(self):
        nan = np.full(3, np.nan)

        assert cocos_from_signs(nan, nan, nan, nan, np.full((3, 5), np.nan)) == 1


# The three classes below hold the causality of the stored signals:
# no grid value may draw on a later sample.
class TestSignalOnGrid:
    def test_takes_the_last_sample_at_or_before_and_leaves_gaps_nan(self):
        # 5 ms source with a missing sample at 15 ms and a 20 ms gap after 25 ms
        grid = grid_ms(60)
        source_times = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.045, 0.050])
        values = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])

        values_on_grid = signal_on_grid(source_times, values, grid)

        # 12 ms holds the 10 ms sample, never the 15 ms or 20 ms one
        assert values_on_grid[12] == 3.0
        # The missing 15 ms sample leaves 10 ms held to 17.5 ms, then NaN until 20 ms
        assert values_on_grid[17] == 3.0
        assert np.isnan(values_on_grid[18])
        assert values_on_grid[20] == 5.0
        # Past 1.5 periods after 25 ms nothing is held, the 45 ms sample is not pulled back
        assert values_on_grid[32] == 6.0
        assert np.isnan(values_on_grid[33:45]).all()
        assert values_on_grid[45] == 7.0

    def test_float64_source_on_a_float32_grid_is_held_at_its_own_grid_time(self):
        # A float32 grid time can sit just below a float64 sample at the same millisecond,
        # which must not leave it holding the sample before
        grid = grid_ms(1001)
        source_times = np.arange(1001) * 1e-3
        values = np.arange(1001, dtype=float)

        values_on_grid = signal_on_grid(source_times, values, grid)

        np.testing.assert_array_equal(values_on_grid, values)

    def test_holds_on_the_clock_of_the_finite_samples(self):
        # A 0.1 ms clock populated only every 5 ms
        source_times = np.arange(500) * 1e-4
        values = np.full(500, np.nan)
        values[::50] = np.arange(10.0)
        grid = grid_ms(50)

        values_on_grid = signal_on_grid(source_times, values, grid)

        # Each 5 ms sample is held up to the next one
        np.testing.assert_array_equal(values_on_grid, np.repeat(np.arange(10.0), 5))
        # A lone finite sample has no period to hold for
        values_lone = np.full(500, np.nan)
        values_lone[100] = 1.0
        values_lone_on_grid = signal_on_grid(source_times, values_lone, grid)
        assert np.isnan(values_lone_on_grid).all()

    def test_averages_a_faster_source_over_each_grid_step(self):
        # 0.2 ms source on the 1 kHz grid, so grid time t takes the mean of the samples in (t - 1 ms, t]
        grid = grid_ms(10)
        source_times = np.arange(50) * 2e-4
        values = np.arange(50.0)
        values[16:22] = np.nan

        values_on_grid = signal_on_grid(source_times, values, grid)

        # 1 ms averages samples 1 to 5, the one at 1 ms itself included, never the 1.2 ms one.
        # 4 ms has no finite sample, and 5 ms averages the finite 22 to 25.
        expected = np.array([0.0, 3.0, 8.0, 13.0, np.nan, 23.5, 28.0, 33.0, 38.0, 43.0])
        np.testing.assert_array_equal(values_on_grid, expected)


class TestHoldFromUsableReconstructions:
    def test_held_from_the_last_usable_reconstruction_with_no_limit(self):
        # Reconstructions at 2, 7, 12 and 40 ms, the one at 7 ms without a finite qpsi,
        # and a stray energy_mhd sample at 20 ms with no reconstruction under it
        grid = grid_ms(50)
        reconstructed = [2, 7, 12, 40]
        simagx = np.full(grid.size, np.nan)
        psirz = np.full((grid.size, 2, 2), np.nan)
        qpsi = np.full((grid.size, 3), np.nan)
        energy_mhd = np.full(grid.size, np.nan)
        for value, i_time in enumerate(reconstructed, start=1):
            simagx[i_time] = 0.0
            psirz[i_time] = 0.0
            qpsi[i_time] = np.nan if i_time == 7 else 1.0
            energy_mhd[i_time] = float(value)
        energy_mhd[20] = 99.0
        ds = xr.Dataset(
            {
                "simagx": (("shot", "time"), simagx[None]),
                "sibdry": (("shot", "time"), simagx[None] + 1.0),
                "qpsi": (("shot", "time", "psi_idx"), qpsi[None]),
                "psirz": (("shot", "time", "r_grid", "z_grid"), psirz[None]),
                "energy_mhd": (("shot", "time"), energy_mhd[None]),
            },
            coords={"shot": [SHOT], "time": grid},
        )

        usable = usable_reconstructions(ds)
        held = hold_from_usable_reconstructions(ds, usable)

        energy_held = held["energy_mhd"].squeeze("shot").values
        assert np.isnan(energy_held[:2]).all()
        # The unusable 7 ms reconstruction and the stray 20 ms sample are skipped,
        # and the 12 ms reconstruction holds across the 28 ms gap to the next usable one
        assert (energy_held[2:12] == 1.0).all()
        assert (energy_held[12:40] == 3.0).all()
        assert (energy_held[40:] == 4.0).all()


class TestSmoothedPower:
    def test_impulse_spreads_into_a_centered_triangle_and_nan_stays_nan(self):
        # An impulse far from both ends of the record, and a missing sample elsewhere
        values = np.zeros(401)
        values[200] = 1.0
        values[50] = np.nan

        smoothed = smoothed_power(values)

        # A 50 ms window is 51 samples, so two passes give a triangle 101 samples wide
        # with its peak 1 / 51 on the impulse, the same before it as after it
        n_window = 51
        offsets = np.arange(-100, 101)
        triangle = np.clip(n_window - np.abs(offsets), 0, None) / n_window**2
        np.testing.assert_allclose(smoothed[100:301], triangle, atol=1e-12)
        assert np.isnan(smoothed[50])


class TestOhmicPower:
    def test_quadratic_w_pol_matches_the_backward_difference_closed_form(self):
        # W_pol = k t^2, so its backward difference over one step is k (t_n + t_n-1)
        times = np.arange(10) * 1e-3
        current = 5e5 + 2e6 * times
        v_loop = np.full(times.size, 1.5)
        w_pol_rate = 3e7
        w_pol = w_pol_rate * times**2

        p_ohm = ohmic_power(times, current, v_loop, w_pol)

        dw_pol_dt = w_pol_rate * (times[1:] + times[:-1])
        expected = current[1:] * v_loop[1:] - dw_pol_dt
        assert np.isnan(p_ohm[0])
        np.testing.assert_allclose(p_ohm[1:], expected, rtol=1e-12)

    def test_lcfs_voltage_follows_the_cocos_sign_of_psi(self):
        # psi_boundary falling 1 mWb/rad per ms is 2 pi V through the boundary,
        # positive for sigma_Bp < 0 (COCOS 3, 7) and negative for sigma_Bp > 0 (COCOS 1, 5)
        times = np.array([0.0, 1e-3, 2e-3])
        sibdry = np.array([0.0, -1e-3, -2e-3])

        v_cocos_3 = lcfs_voltage(times, sibdry, 3)
        v_cocos_1 = lcfs_voltage(times, sibdry, 1)

        assert np.isnan(v_cocos_3[0]) and np.isnan(v_cocos_1[0])
        np.testing.assert_allclose(v_cocos_3[1:], 2 * np.pi)
        np.testing.assert_allclose(v_cocos_1[1:], -2 * np.pi)


class TestPoloidalFieldEnergy:
    def test_circular_quadratic_flux_matches_the_quadrature_reference(self):
        # psi = psi_axis + c r^2 on a circle of radius a about (R0, 0), so |grad psi| = 2 c r
        # and W = (pi / mu0) int 4 c^2 r^2 / R dR dZ = (pi / mu0) 4 c^2 int_0^a 2 pi r^3 / sqrt(R0^2 - r^2) dr
        major_radius = 0.9
        minor_radius = 0.25
        curvature = 0.4
        psi_axis = -0.1
        r_grid = np.linspace(
            major_radius - 1.4 * minor_radius, major_radius + 1.4 * minor_radius, 281
        )
        z_grid = np.linspace(-1.4 * minor_radius, 1.4 * minor_radius, 281)
        r_mesh, z_mesh = np.meshgrid(r_grid, z_grid, indexing="ij")
        distance_squared = (r_mesh - major_radius) ** 2 + z_mesh**2
        psirz = psi_axis + curvature * distance_squared
        theta = np.linspace(0.0, 2 * np.pi, 361)
        r_boundary = major_radius + minor_radius * np.cos(theta)
        z_boundary = minor_radius * np.sin(theta)
        sibdry = psi_axis + curvature * minor_radius**2

        energy = poloidal_field_energy(
            psirz[np.newaxis],
            r_grid,
            z_grid,
            np.array([psi_axis]),
            np.array([sibdry]),
            r_boundary[np.newaxis],
            z_boundary[np.newaxis],
        )

        def integrand(distance):
            return 2 * np.pi * distance**3 / np.sqrt(major_radius**2 - distance**2)

        radial_integral, _ = quad(integrand, 0.0, minor_radius)
        expected = np.pi / mu_0 * 4 * curvature**2 * radial_integral
        np.testing.assert_allclose(energy[0], expected, rtol=0.02)

    def test_padding_and_missing_contours(self):
        r_grid = np.linspace(0.5, 1.5, 21)
        z_grid = np.linspace(-0.5, 0.5, 21)
        psirz = np.zeros((2, 21, 21))
        r_boundary = np.full((2, 5), np.nan)
        z_boundary = np.full((2, 5), np.nan)
        r_boundary[0, :2] = [0.9, 1.1]
        z_boundary[0, :2] = [0.0, 0.0]

        energy = poloidal_field_energy(
            psirz, r_grid, z_grid, np.zeros(2), np.ones(2), r_boundary, z_boundary
        )

        assert np.isnan(energy).all()


class TestNormalizedBeta:
    ENERGY_MHD = 1.2e4
    VOLUME = 1.25
    MINOR_RADIUS = 0.21
    B0 = -1.41
    IP = -2.9e5

    def test_sign_invariance(self):
        energy_mhd = np.array([self.ENERGY_MHD])
        beta_both_negative = normalized_beta(
            energy_mhd, self.VOLUME, self.MINOR_RADIUS, self.B0, self.IP
        )
        beta_both_positive = normalized_beta(
            energy_mhd, self.VOLUME, self.MINOR_RADIUS, -self.B0, -self.IP
        )
        beta_mixed = normalized_beta(
            energy_mhd, self.VOLUME, self.MINOR_RADIUS, self.B0, -self.IP
        )
        np.testing.assert_allclose(beta_both_positive, beta_both_negative)
        np.testing.assert_allclose(beta_mixed, beta_both_negative)

    def test_zero_current_is_not_finite(self):
        beta_tor_norm = normalized_beta(
            np.array([self.ENERGY_MHD]),
            self.VOLUME,
            self.MINOR_RADIUS,
            self.B0,
            np.array([0.0]),
        )
        assert not np.isfinite(beta_tor_norm).any()
