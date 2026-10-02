"""Unit tests for the device-independent signal helpers in machine/generic.py.

Synthetic arrays only, no MDSplus and no network, so these run anywhere.
"""

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets.machine.generic import (
    MU0,
    cocos_from_signs,
    make_uniform_1kHz_timebase,
    ohmic_power,
    signal_on_grid,
    snap_to_grid,
    trailing_boxcar_mean,
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
class TestHeldSignalOnGrid:
    def test_takes_the_last_sample_at_or_before_and_leaves_gaps_nan(self):
        # 5 ms source with a missing sample at 15 ms and a 20 ms gap after 25 ms
        grid = np.round(np.arange(60) * 1e-3, 3)
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

    def test_hold_floor_bridges_a_missing_sample_and_no_more(self):
        # The source above, held for at least 10 ms, as every equilibrium signal is
        grid = np.round(np.arange(60) * 1e-3, 3)
        source_times = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.045, 0.050])
        values = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])

        values_on_grid = signal_on_grid(source_times, values, grid, hold_floor=10e-3)

        # The missing 15 ms sample is bridged by the 10 ms one
        assert (values_on_grid[10:20] == 3.0).all()
        assert values_on_grid[20] == 5.0
        # The 20 ms gap after 25 ms is not, the hold ends 10 ms after it
        assert values_on_grid[34] == 6.0
        assert np.isnan(values_on_grid[36:45]).all()
        assert values_on_grid[45] == 7.0

    def test_float64_source_on_a_float32_grid_is_held_at_its_own_grid_time(self):
        # A float32 grid time can sit just below a float64 sample at the same millisecond,
        # which must not leave it holding the sample before
        grid = np.round(np.arange(1001) * 1e-3, 3).astype("float32")
        source_times = np.arange(1001) * 1e-3
        values = np.arange(1001, dtype=float)

        values_on_grid = signal_on_grid(source_times, values, grid)

        np.testing.assert_array_equal(values_on_grid, values)

    def test_holds_on_the_clock_of_the_finite_samples(self):
        # A 0.1 ms clock populated only every 5 ms, as the MAST esm group stores pphix
        source_times = np.arange(500) * 1e-4
        values = np.full(500, np.nan)
        values[::50] = np.arange(10.0)
        grid = np.round(np.arange(50) * 1e-3, 3)

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
        grid = np.round(np.arange(10) * 1e-3, 3).astype("float32")
        source_times = np.arange(50) * 2e-4
        values = np.arange(50.0)
        values[16:22] = np.nan

        values_on_grid = signal_on_grid(source_times, values, grid)

        # 1 ms averages samples 1 to 5, the one at 1 ms itself included, never the 1.2 ms one
        # 4 ms has no finite sample, and 5 ms averages the finite 22 to 25
        expected = np.array([0.0, 3.0, 8.0, 13.0, np.nan, 23.5, 28.0, 33.0, 38.0, 43.0])
        np.testing.assert_array_equal(values_on_grid, expected)


class TestTrailingBoxcarMean:
    def test_a_later_sample_does_not_change_earlier_ones(self):
        values = np.ones(30)
        values_spiked = values.copy()
        values_spiked[20] = 100.0

        smoothed = trailing_boxcar_mean(values, 5e-3, 1e-3)
        smoothed_spiked = trailing_boxcar_mean(values_spiked, 5e-3, 1e-3)

        np.testing.assert_array_equal(smoothed_spiked[:20], smoothed[:20])
        # The spike enters at its own sample and leaves 5 samples later
        assert smoothed_spiked[20] == pytest.approx((4.0 + 100.0) / 5.0)
        assert smoothed_spiked[25] == 1.0


class TestOhmicPower:
    def test_current_ramp_matches_the_backward_difference_closed_form(self):
        # Linear Ip ramp at fixed li and R, so dW_pol/dt over one step is
        # mu0 R li / 4 * dIp/dt * (Ip_n + Ip_n-1)
        times = np.arange(10) * 1e-3
        ip_ramp_rate = 2e6
        ip = 5e5 + ip_ramp_rate * times
        v_loop = np.full(times.size, 1.5)
        li = np.full(times.size, 1.2)
        major_radius = np.full(times.size, 0.68)

        p_ohm = ohmic_power(times, ip, v_loop, li, major_radius)

        dw_pol_dt = MU0 * 0.68 * 1.2 / 4.0 * ip_ramp_rate * (ip[1:] + ip[:-1])
        expected = ip[1:] * v_loop[1:] - dw_pol_dt
        assert np.isnan(p_ohm[0])
        np.testing.assert_allclose(p_ohm[1:], expected, rtol=1e-12)
