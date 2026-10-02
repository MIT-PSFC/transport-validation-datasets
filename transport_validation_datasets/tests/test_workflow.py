"""Unit tests for the index math DataWorkflow uses to build the final dataset.

These are the pure helpers: no device, no source data, no disk. They decide
which grid times carry a sample and which hold an earlier one, so a silent
change here quietly changes what lands in every device's store.
"""

import numpy as np
import xarray as xr

from transport_validation_datasets import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_validation_datasets.gp_fitting.batch_io import STATUS_CULLED, STATUS_OK
from transport_validation_datasets.machine.generic import (
    MAX_HOLD_PERIODS,
    hold_onto_grid,
)
from transport_validation_datasets.workflow import (
    _hold_equilibrium,
    _trim_and_keep_longest,
    _trim_segment_starts,
    keep_longest_segment,
    usable_slice_mask,
)


def grid_ms(n: int) -> np.ndarray:
    # n grid points on the 1 kHz timebase, 0 ms .. (n-1) ms
    return np.round(np.arange(n) * 1e-3, 3)


def runs(*lengths: int) -> np.ndarray:
    # Alternating kept and cut runs of the given lengths in samples, starting with a kept run
    return np.concatenate([np.full(n, i % 2 == 0) for i, n in enumerate(lengths)])


class TestKeepLongestSegment:
    def test_only_the_longest_run_left(self):
        times = grid_ms(30)
        keep = runs(5, 2, 10, 3, 10)

        kept, dropped = keep_longest_segment(keep, times)

        # Equally long runs keep the earliest
        assert np.flatnonzero(kept).tolist() == list(range(7, 17))
        # Measured first to last sample, so n samples span n - 1 ms
        assert np.allclose(dropped, [0.004, 0.009])

    def test_empty_mask_keeps_nothing(self):
        kept, dropped = keep_longest_segment(np.zeros(10, dtype=bool), grid_ms(10))

        assert not kept.any()
        assert dropped == []


class TestTrimSegmentStarts:
    def test_each_segment_starts_on_its_first_startable_sample(self):
        # Three segments: one starting on a sample that can start it,
        # one whose first such sample is 2 samples in, and one with none
        keep = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
        can_start = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], dtype=bool)

        trimmed = _trim_segment_starts(keep, can_start)

        assert np.flatnonzero(trimmed).tolist() == [0, 1, 2, 6, 7]


class TestTrimAndKeepLongest:
    def test_earlier_segment_never_starts_a_later_one(self):
        # A 5 ms clock, so a reconstruction reaches 7.5 ms.
        # The 0 ms reconstruction of the 0-2 ms segment reaches 4-7 ms, but only one segment is kept,
        # so 4-9 ms have no equilibrium of their own and are trimmed.
        times = grid_ms(30)
        keep = np.zeros(30, dtype=bool)
        keep[0:3] = True
        keep[4:21] = True
        reconstruction_usable = np.zeros(30, dtype=bool)
        reconstruction_usable[[0, 10, 15, 20]] = True

        kept, dropped, n_trimmed = _trim_and_keep_longest(
            keep, times, reconstruction_usable, 0.005
        )

        assert np.flatnonzero(kept).tolist() == list(range(10, 21))
        assert len(dropped) == 1
        assert n_trimmed == 6

    def test_longest_judged_after_every_trim(self):
        # 0-45 ms starts on its own 0 ms reconstruction.
        # 47-95 ms is reached at 47 only by the 45 ms reconstruction before it,
        # so on its own it starts at 58 and spans 37 ms, shorter than 0-45 ms.
        times = grid_ms(100)
        keep = np.zeros(100, dtype=bool)
        keep[0:46] = True
        keep[47:96] = True
        reconstruction_usable = np.zeros(100, dtype=bool)
        reconstruction_usable[0:46:5] = True
        reconstruction_usable[58:96:5] = True

        kept, dropped, n_trimmed = _trim_and_keep_longest(
            keep, times, reconstruction_usable, 0.005
        )

        assert np.flatnonzero(kept).tolist() == list(range(0, 46))
        assert np.allclose(dropped, [0.037])
        assert n_trimmed == 11


class TestHoldOntoGrid:
    def test_no_samples_means_no_grid_time_draws_on_anything(self):
        grid = grid_ms(10)

        index, fresh = hold_onto_grid(grid, np.array([]), True)

        assert (index == -1).all()
        assert not fresh.any()

    def test_fresh_marks_only_grid_times_carrying_sample(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        _, fresh = hold_onto_grid(grid, samples, True)

        assert np.array_equal(np.flatnonzero(fresh), [2, 7, 12])

    def test_a_sample_is_held_forward_until_the_next_one(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        index, _ = hold_onto_grid(grid, samples, True)

        assert (index[:2] == -1).all()  # nothing before the first sample
        assert (index[2:7] == 0).all()
        assert (index[7:12] == 1).all()

    def test_holding_stops_after_max_hold_periods(self):
        # Median sampling period is 5 ms, so the last sample at 12 ms is held
        # to 12 + 1.5 * 5 = 19.5 ms and no further
        grid = grid_ms(30)
        samples = np.array([0.002, 0.007, 0.012])

        index, _ = hold_onto_grid(grid, samples, True)

        assert index[19] == 2
        assert (index[20:] == -1).all()

    def test_nothing_carried_across_long_gap(self):
        grid = grid_ms(40)
        samples = np.array([0.001, 0.006, 0.011, 0.035])
        period = float(np.median(np.diff(samples)))
        # The last held grid time is the last whole millisecond at or before the cutoff
        last_held = int(np.floor(1e3 * (samples[2] + MAX_HOLD_PERIODS * period) + 1e-9))

        index, _ = hold_onto_grid(grid, samples, True)

        assert index[last_held] == 2
        assert (index[last_held + 1 : 35] == -1).all()
        assert index[35] == 3

    def test_without_forward_fill_only_fresh_grid_times_draw_sample(self):
        grid = grid_ms(20)
        samples = np.array([0.002, 0.007, 0.012])

        index, fresh = hold_onto_grid(grid, samples, False)

        assert np.array_equal(np.flatnonzero(index >= 0), np.flatnonzero(fresh))
        assert np.array_equal(index[fresh], [0, 1, 2])

    def test_lone_sample_falls_back_to_grid_step(self):
        # One sample has no sampling period of its own, so it only covers
        # MAX_HOLD_PERIODS grid steps rather than the rest of the shot
        grid = grid_ms(20)

        index, fresh = hold_onto_grid(grid, np.array([0.002]), True)

        assert np.flatnonzero(fresh).tolist() == [2]
        assert index[2] == 0
        assert index[3] == 0
        assert (index[4:] == -1).all()

    def test_float_round_off_still_counts_as_same_time(self):
        # Everything shares the 1 kHz timebase,
        # so the tolerance only has to absorb round-off
        grid = grid_ms(10)
        samples = np.array([0.005 + 1e-9])

        _, fresh = hold_onto_grid(grid, samples, True)

        assert np.flatnonzero(fresh).tolist() == [5]


def unprocessed_with_equilibrium(
    grid: np.ndarray, reconstructed: list[int], nan_qpsi: tuple[int, ...] = ()
):
    # One shot's unprocessed dataset as _hold_equilibrium sees it: the time
    # coordinate already dropped, a reconstruction only at the grid times that
    # have one, and NaN everywhere else
    n_t = grid.size
    simagx = np.full(n_t, np.nan)
    psirz = np.full((n_t, 2, 2), np.nan)
    qpsi = np.full((n_t, 3), np.nan)
    for value, i_time in enumerate(reconstructed, start=1):
        simagx[i_time] = float(value)
        psirz[i_time] = float(value)
        if i_time not in nan_qpsi:
            qpsi[i_time] = 1.0
    return xr.Dataset(
        {
            "simagx": ((EPISODE_DIM, TIME_COORD), simagx[None]),
            "sibdry": ((EPISODE_DIM, TIME_COORD), simagx[None] + 1.0),
            "qpsi": ((EPISODE_DIM, TIME_COORD, "psi_idx"), qpsi[None]),
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

    def test_unusable_reconstruction_held_over_for_the_clock_period(self):
        # A 5 ms clock with no qpsi at 7 ms,
        # so the 2 ms reconstruction holds for 1.5 clock periods, to 9.5 ms,
        # not for 1.5 of the 10 ms the usable ones are apart
        grid = grid_ms(20)
        ds = unprocessed_with_equilibrium(grid, [2, 7, 12], nan_qpsi=(7,))

        held, fresh = _hold_equilibrium(ds, grid, True)

        simagx = held["simagx"].squeeze(EPISODE_DIM, drop=True).values
        assert np.flatnonzero(fresh).tolist() == [2, 12]
        assert (simagx[2:10] == 1.0).all()
        assert np.isnan(simagx[10:12]).all()
        assert (simagx[12:] == 3.0).all()

    def test_device_without_equilibrium_passes_signals_through(self):
        grid = grid_ms(20)
        ds = xr.Dataset(
            {"rmagx": ((EPISODE_DIM, TIME_COORD), np.zeros((1, grid.size)))}
        )

        held, fresh = _hold_equilibrium(ds, grid, True)

        assert set(held) == {"rmagx"}
        assert not fresh.any()
        assert held["rmagx"].shape == (1, grid.size)


class TestUsableSliceMask:
    def test_flat_te_and_band_wider_than_profile_inside_lcfs_rejected(self):
        rho_tor_norm = np.linspace(0.0, 1.1, 23)
        i_mid = 10  # rho_tor_norm 0.5
        i_sol = 21  # rho_tor_norm 1.05
        n_t = 6
        te = np.tile(1000.0 * (1.0 - (rho_tor_norm / 1.2) ** 2), (n_t, 1))
        ne = np.tile(1.0e20 * (1.0 - (rho_tor_norm / 1.2) ** 2), (n_t, 1))
        te_error = np.full((n_t, rho_tor_norm.size), 50.0)
        ne_error = np.full((n_t, rho_tor_norm.size), 5.0e18)
        status = np.full(n_t, STATUS_OK)
        # 0 is a healthy slice
        te[1] = 500.0  # flat, te at the LCFS equals its peak
        te_error[2, i_mid] = 2000.0
        ne_error[3, i_mid] = 2.0e20
        # Past the LCFS a blown band says nothing about the profile inside
        te_error[4, i_sol] = 2000.0
        status[5] = STATUS_CULLED
        profile_dims = (EPISODE_DIM, TIME_DIM, "rho_tor_norm")
        ds_fit = xr.Dataset(
            {
                "t_e": (profile_dims, te[None]),
                "t_e_error": (profile_dims, te_error[None]),
                "n_e": (profile_dims, ne[None]),
                "n_e_error": (profile_dims, ne_error[None]),
                "t_e_fit_status": ((EPISODE_DIM, TIME_DIM), status[None]),
                "n_e_fit_status": (
                    (EPISODE_DIM, TIME_DIM),
                    np.full((1, n_t), STATUS_OK),
                ),
            },
            coords={EPISODE_DIM: [1], "rho_tor_norm": rho_tor_norm},
        )

        usable = usable_slice_mask(ds_fit)

        assert np.flatnonzero(usable).tolist() == [0, 4]
