"""Unit tests for the device-independent signal helpers in machine/generic.py.

Synthetic arrays only, no MDSplus and no network, so these run anywhere.
"""

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets.machine.generic import (
    make_uniform_1kHz_timebase,
    snap_to_grid,
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
