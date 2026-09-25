"""Unit tests for the index math DataWorkflow uses to build the final dataset.

These are the pure helpers: no device, no source data, no disk. They decide
which grid times carry a sample and which hold an earlier one, so a silent
change here quietly changes what lands in every device's store.
"""

import numpy as np
import xarray as xr

from transport_validation_datasets import EPISODE_DIM, TIME_COORD
from transport_validation_datasets.workflow import (
    MAX_HOLD_PERIODS,
    _hold_equilibrium,
    _hold_onto_grid,
    drop_short_segments,
)


def grid_ms(n: int) -> np.ndarray:
    # n grid points on the 1 kHz timebase, 0 ms .. (n-1) ms
    return np.round(np.arange(n) * 1e-3, 3)


class TestDropShortSegments:
    def test_segment_shorter_than_the_minimum_cleared(self):
        times = grid_ms(10)
        keep = np.array(
            [False, False, True, True, True, False, False, False, False, False]
        )

        kept, dropped = drop_short_segments(keep, times, min_length=0.005)

        assert not kept.any()
        assert len(dropped) == 1

    def test_long_segment_survives(self):
        times = grid_ms(10)
        keep = np.array(
            [False, False, True, True, True, True, True, True, False, False]
        )

        kept, dropped = drop_short_segments(keep, times, min_length=0.005)

        assert np.array_equal(kept, keep)
        assert dropped == []

    def test_segment_measured_first_to_last_sample(self):
        # n samples on the 1 kHz grid span n - 1 ms,
        # so a 5 sample run is exactly 4 ms long and a 4 ms minimum keeps it
        times = grid_ms(10)
        keep = np.array(
            [False, True, True, True, True, True, False, False, False, False]
        )

        kept, dropped = drop_short_segments(keep, times, min_length=0.004)
        assert np.array_equal(kept, keep)
        assert dropped == []

        kept, dropped = drop_short_segments(keep, times, min_length=0.005)
        assert not kept.any()
        assert dropped == [0.004]

    def test_short_and_long_segments_judged_independently(self):
        times = grid_ms(12)
        keep = np.array(
            [True, True, False, False, False]
            + [True, True, True, True, True, True, True]
        )

        kept, dropped = drop_short_segments(keep, times, min_length=0.005)

        assert np.array_equal(
            kept,
            np.array(
                [False, False, False, False, False]
                + [True, True, True, True, True, True, True]
            ),
        )
        assert len(dropped) == 1

    def test_run_touching_either_end_still_seen(self):
        times = grid_ms(10)
        keep = np.array(
            [True, True, False, False, False, False, False, False, True, True]
        )

        kept, dropped = drop_short_segments(keep, times, min_length=0.005)

        assert not kept.any()
        assert len(dropped) == 2

    def test_empty_mask_drops_nothing(self):
        times = grid_ms(10)

        kept, dropped = drop_short_segments(np.zeros(10, dtype=bool), times, 0.005)

        assert not kept.any()
        assert dropped == []

    def test_input_mask_not_mutated(self):
        times = grid_ms(10)
        keep = np.array(
            [False, False, True, True, True, False, False, False, False, False]
        )
        original = keep.copy()

        drop_short_segments(keep, times, min_length=0.005)

        assert np.array_equal(keep, original)


class TestHoldOntoGrid:
    def test_no_samples_means_no_grid_time_draws_on_anything(self):
        grid = grid_ms(10)

        index, fresh = _hold_onto_grid(grid, np.array([]), True)

        assert (index == -1).all()
        assert not fresh.any()

    def test_fresh_marks_only_grid_times_carrying_sample(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        _, fresh = _hold_onto_grid(grid, samples, True)

        assert np.array_equal(np.flatnonzero(fresh), [2, 7, 12])

    def test_a_sample_is_held_forward_until_the_next_one(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        index, _ = _hold_onto_grid(grid, samples, True)

        assert (index[:2] == -1).all()  # nothing before the first sample
        assert (index[2:7] == 0).all()
        assert (index[7:12] == 1).all()

    def test_holding_stops_after_max_hold_periods(self):
        # Median sampling period is 5 ms, so the last sample at 12 ms is held
        # to 12 + 1.5 * 5 = 19.5 ms and no further
        grid = grid_ms(30)
        samples = np.array([0.002, 0.007, 0.012])

        index, _ = _hold_onto_grid(grid, samples, True)

        assert index[19] == 2
        assert (index[20:] == -1).all()

    def test_nothing_carried_across_long_gap(self):
        grid = grid_ms(40)
        samples = np.array([0.001, 0.006, 0.011, 0.035])
        period = float(np.median(np.diff(samples)))
        # The last held grid time is the last whole millisecond at or before the cutoff
        last_held = int(np.floor(1e3 * (samples[2] + MAX_HOLD_PERIODS * period) + 1e-9))

        index, _ = _hold_onto_grid(grid, samples, True)

        assert index[last_held] == 2
        assert (index[last_held + 1 : 35] == -1).all()
        assert index[35] == 3

    def test_without_forward_fill_only_fresh_grid_times_draw_sample(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        index, fresh = _hold_onto_grid(grid, samples, False)

        assert np.array_equal(np.flatnonzero(index >= 0), np.flatnonzero(fresh))
        assert np.array_equal(index[fresh], [0, 1, 2])

    def test_lone_sample_falls_back_to_grid_step(self):
        # One sample has no sampling period of its own, so it only covers
        # MAX_HOLD_PERIODS grid steps rather than the rest of the shot
        grid = grid_ms(20)

        index, fresh = _hold_onto_grid(grid, np.array([0.002]), True)

        assert np.flatnonzero(fresh).tolist() == [2]
        assert index[2] == 0
        assert index[3] == 0
        assert (index[4:] == -1).all()

    def test_float_round_off_still_counts_as_same_time(self):
        # Everything shares the 1 kHz timebase,
        # so the tolerance only has to absorb round-off
        grid = grid_ms(10)
        samples = np.array([0.005 + 1e-9])

        _, fresh = _hold_onto_grid(grid, samples, True)

        assert np.flatnonzero(fresh).tolist() == [5]


def unprocessed_with_equilibrium(grid: np.ndarray, reconstructed: list[int]):
    # One shot's unprocessed dataset as _hold_equilibrium sees it: the time
    # coordinate already dropped, a reconstruction only at the grid times that
    # have one, and NaN everywhere else
    n_t = grid.size
    simagx = np.full(n_t, np.nan)
    psirz = np.full((n_t, 2, 2), np.nan)
    for value, i_time in enumerate(reconstructed, start=1):
        simagx[i_time] = float(value)
        psirz[i_time] = float(value)
    return xr.Dataset(
        {
            "simagx": ((EPISODE_DIM, TIME_COORD), simagx[None]),
            "psirz": ((EPISODE_DIM, TIME_COORD, "r_grid", "z_grid"), psirz[None]),
        }
    )


class TestHoldEquilibrium:
    def test_fresh_marks_grid_times_reconstruction_landed_on(self):
        grid = grid_ms(20)
        ds = unprocessed_with_equilibrium(grid, [2, 7, 12])

        _, fresh = _hold_equilibrium(ds, grid, True)

        assert np.array_equal(np.flatnonzero(fresh), [2, 7, 12])

    def test_each_reconstruction_held_until_next(self):
        grid = grid_ms(20)
        ds = unprocessed_with_equilibrium(grid, [2, 7, 12])

        held, _ = _hold_equilibrium(ds, grid, True)

        simagx = held["simagx"].squeeze(EPISODE_DIM, drop=True).values
        assert np.isnan(simagx[:2]).all()
        assert (simagx[2:7] == 1.0).all()
        assert (simagx[7:12] == 2.0).all()

        psirz = held["psirz"].squeeze(EPISODE_DIM, drop=True).values
        assert (psirz[2:7] == 1.0).all()
        assert np.isnan(psirz[:2]).all()

    def test_without_forward_fill_only_reconstruction_times_finite(self):
        grid = grid_ms(20)
        ds = unprocessed_with_equilibrium(grid, [2, 7, 12])

        held, fresh = _hold_equilibrium(ds, grid, False)
        simagx = held["simagx"].squeeze(EPISODE_DIM, drop=True).values

        assert np.array_equal(
            np.flatnonzero(np.isfinite(simagx)), np.flatnonzero(fresh)
        )

    def test_device_without_equilibrium_passes_signals_through(self):
        grid = grid_ms(20)
        ds = xr.Dataset(
            {"rmagx": ((EPISODE_DIM, TIME_COORD), np.zeros((1, grid.size)))}
        )

        held, fresh = _hold_equilibrium(ds, grid, True)

        assert set(held) == {"rmagx"}
        assert not fresh.any()
        assert held["rmagx"].shape == (1, grid.size)
