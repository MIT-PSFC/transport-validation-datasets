"""Assembling episodes into one Zarr store: padding, refusals, and the skip accounting."""

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
DIM_SIZES = {TIME_DIM: 4, "r_grid": 5}


def _episode(shot: int, n_time: int, r_grid: np.ndarray = R_GRID) -> xr.Dataset:
    time = 0.001 * np.arange(n_time)
    return xr.Dataset(
        {
            "ip": (TIME_DIM, float(shot) + np.arange(n_time, dtype=float)),
            "psi": ((TIME_DIM, "r_grid"), np.ones((n_time, r_grid.size))),
            "status": (TIME_DIM, np.zeros(n_time, dtype=np.int8)),
        },
        coords={EPISODE_DIM: shot, "r_grid": r_grid, "time": (TIME_DIM, time)},
    )


def test_episodes_are_padded_to_the_bounds_and_the_store_reopens(tmp_path):
    zarr_path = tmp_path / "store.zarr"

    ds = build_tensorized_dataset(
        lambda shot: _episode(shot, n_time=2 + shot),
        identifiers=[1, 2],
        zarr_path=zarr_path,
        time_dim=TIME_DIM,
        episode_dim=EPISODE_DIM,
        dim_sizes=DIM_SIZES,
        mb_per_chunk=None,
    )

    assert ds.sizes == {EPISODE_DIM: 2, TIME_DIM: 4, "r_grid": 5}
    assert ds[EPISODE_DIM].values.tolist() == [1, 2]
    ip = ds["ip"].values
    assert ip[0].tolist()[:3] == [1.0, 2.0, 3.0] and np.isnan(ip[0, 3])
    assert np.isfinite(ip[1]).all()
    # The int8 status pads as NaN, not as a clean 0
    assert np.isnan(ds["status"].values[0, 3])
    ds.close()
    with xr.open_zarr(zarr_path, consolidated=True) as reopened:
        assert reopened.sizes[EPISODE_DIM] == 2


def test_rechunk_trims_the_padding_slack(tmp_path):
    ds = build_tensorized_dataset(
        lambda shot: _episode(shot, n_time=2),
        identifiers=[1, 2],
        zarr_path=tmp_path / "store.zarr",
        time_dim=TIME_DIM,
        episode_dim=EPISODE_DIM,
        dim_sizes=DIM_SIZES,
        mb_per_chunk=1,
    )

    assert ds.sizes[TIME_DIM] == 2
    ds.close()


def test_coordinate_mismatch_with_the_store_raises(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    add_to_zarr_store(_episode(1, 3), zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)

    other_grid = _episode(2, 3, r_grid=R_GRID + 0.01)
    with pytest.raises(ValueError, match="Coordinate r_grid"):
        add_to_zarr_store(other_grid, zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)

    # A shorter grid pads with NaN, which is not the store's values either
    shorter_grid = _episode(3, 3, r_grid=R_GRID[:4])
    with pytest.raises(ValueError, match="Coordinate r_grid"):
        add_to_zarr_store(shorter_grid, zarr_path, TIME_DIM, EPISODE_DIM, DIM_SIZES)

    with xr.open_zarr(zarr_path, consolidated=True) as ds_store:
        assert ds_store.sizes[EPISODE_DIM] == 1


def test_episode_over_the_bound_raises(tmp_path):
    with pytest.raises(ValueError, match="over the dim_sizes bound"):
        add_to_zarr_store(
            _episode(1, 5), tmp_path / "store.zarr", TIME_DIM, EPISODE_DIM, DIM_SIZES
        )


def test_raising_episodes_are_skipped_and_counted(tmp_path):
    def process(shot):
        if shot == 2:
            raise OSError("unreadable")
        return _episode(shot, 2)

    ds = build_tensorized_dataset(
        process,
        identifiers=[1, 2, 3],
        zarr_path=tmp_path / "store.zarr",
        time_dim=TIME_DIM,
        episode_dim=EPISODE_DIM,
        dim_sizes=DIM_SIZES,
        mb_per_chunk=None,
    )

    assert ds[EPISODE_DIM].values.tolist() == [1, 3]
    ds.close()


def test_nothing_written_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No episodes were written"):
        build_tensorized_dataset(
            lambda shot: None,
            identifiers=[1, 2],
            zarr_path=tmp_path / "store.zarr",
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            dim_sizes=DIM_SIZES,
        )
