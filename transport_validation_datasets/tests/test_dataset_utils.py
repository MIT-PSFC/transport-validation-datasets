"""Assembling episodes into one Zarr store: padding, the block-wise rechunk, and the refusals."""

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets.dataset_utils import (
    add_to_zarr_store,
    build_tensorized_dataset,
)

TIME_DIM = "time_idx"
EPISODE_DIM = "shot"
R_GRID = np.linspace(0.4, 0.9, 5)
# Two slots of slack along time, which the rechunking pass trims
DIM_SIZES = {TIME_DIM: 6, "r_grid": 5}


def _episode(shot: int, n_time: int, r_grid: np.ndarray = R_GRID) -> xr.Dataset:
    time = 0.001 * np.arange(n_time)
    return xr.Dataset(
        {
            "ip": (TIME_DIM, shot + np.arange(n_time, dtype=float)),
            "psi": ((TIME_DIM, "r_grid"), np.full((n_time, r_grid.size), float(shot))),
            "status": (TIME_DIM, np.zeros(n_time, dtype=np.int8)),
        },
        coords={EPISODE_DIM: shot, "r_grid": r_grid, "time": (TIME_DIM, time)},
    )


def _build(zarr_path, shots, mb_per_chunk, process_fn=None) -> xr.Dataset:
    if process_fn is None:

        def process_fn(shot):
            return _episode(shot, n_time=shot)

    return build_tensorized_dataset(
        process_fn,
        identifiers=shots,
        zarr_path=zarr_path,
        time_dim=TIME_DIM,
        episode_dim=EPISODE_DIM,
        dim_sizes=DIM_SIZES,
        mb_per_chunk=mb_per_chunk,
    )


def test_shorter_episode_pads_with_nan_and_integer_padding_stays_nan(tmp_path):
    # The first episode fills the time bound, so the store holds status as int8
    # and the shorter one appended after it pads through the fill value
    ds = _build(tmp_path / "store.zarr", [6, 3], mb_per_chunk=None)

    assert ds[EPISODE_DIM].values.tolist() == [6, 3]
    ip = ds["ip"].values
    assert ip[1, :3].tolist() == [3.0, 4.0, 5.0]
    assert np.isnan(ip[1, 3:]).all()
    assert np.isnan(ds["psi"].values[1, 3:]).all()
    status = ds["status"].values
    assert (status[0] == 0).all()
    assert np.isnan(status[1, 3])


def test_rechunk_keeps_values_trims_slack_and_sizes_each_variable(tmp_path):
    shots = [1, 2, 3, 4]
    ds_unchunked = _build(tmp_path / "unchunked.zarr", shots, mb_per_chunk=None)
    # ip is 32 bytes per episode once trimmed to 4 grid times, psi 160, so ~100 bytes
    # put 3 episodes of ip in a chunk and 1 of psi
    mb_per_chunk = 100 / (1024 * 1024)

    ds_rechunked = _build(tmp_path / "rechunked.zarr", shots, mb_per_chunk)

    assert ds_unchunked.sizes[TIME_DIM] == DIM_SIZES[TIME_DIM]
    assert ds_rechunked.sizes[TIME_DIM] == 4
    for name in ("ip", "psi", "status"):
        values_trimmed = ds_unchunked[name].isel({TIME_DIM: slice(0, 4)}).values
        np.testing.assert_array_equal(ds_rechunked[name].values, values_trimmed)
    assert ds_rechunked["ip"].chunks[0] == (3, 1)
    assert ds_rechunked["psi"].chunks[0] == (1, 1, 1, 1)


def test_episode_on_another_grid_refused(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    add_to_zarr_store(_episode(1, 3), zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)

    shifted_grid = _episode(2, 3, r_grid=R_GRID + 0.01)
    with pytest.raises(ValueError, match="Coordinate r_grid"):
        add_to_zarr_store(shifted_grid, zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)
    # A shorter grid pads with NaN, which is not the store's values either
    shorter_grid = _episode(3, 3, r_grid=R_GRID[:4])
    with pytest.raises(ValueError, match="Coordinate r_grid"):
        add_to_zarr_store(shorter_grid, zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)

    with xr.open_zarr(zarr_path, consolidated=True) as ds_store:
        assert ds_store[EPISODE_DIM].values.tolist() == [1]


def test_episode_that_raises_is_skipped_and_the_rest_written(tmp_path):
    def process_fn(shot):
        if shot == 2:
            raise OSError("unreadable")
        return _episode(shot, n_time=3)

    ds = _build(tmp_path / "store.zarr", [1, 2, 3], None, process_fn=process_fn)

    assert ds[EPISODE_DIM].values.tolist() == [1, 3]
