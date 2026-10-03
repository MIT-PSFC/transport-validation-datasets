"""Tests for the MAST workflow.

Split in two. The helper tests are synthetic and fast: they build the store
groups by hand and never touch the network. The workflow tests read the real
public level 1 and level 2 stores over S3, which costs ~60-90 s per shot, so
they carry the slow marker and are deselected by default:

    uv run pytest -m slow transport_validation_datasets/tests/test_mast_workflow.py
"""

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets.machine.mast.mast_dataset import (
    LEVEL2_PATH,
    TS_CHANNEL_Z,
    MASTDataWorkflow,
    _store_path_exists,
    _thomson_dataset,
)

# Carries Thomson, an equilibrium, and a limiter contour, and is the shot the
# hollow-ne calibration notes in gp_fitting/zk/quality.py refer to
TEST_SHOT = 30097


def skip_without_store(shot: int = TEST_SHOT):
    if not _store_path_exists(f"{LEVEL2_PATH}/{shot}.zarr"):
        pytest.skip("MAST level 2 store not reachable")


def mast_workflow(test_dir, shotlist=None, **kwargs) -> MASTDataWorkflow:
    test_dir.mkdir(parents=True, exist_ok=True)
    shotlist_file = None
    if shotlist:
        shotlist_file = test_dir / "shotlist.txt"
        shotlist_file.write_text("".join(f"{shot}\n" for shot in shotlist))
    return MASTDataWorkflow(
        ds_name="mast_test",
        shotlist_file=shotlist_file,
        data_assembly_dir=test_dir,
        **kwargs,
    )


def thomson_group(times, radius, te, ne, error_frac=0.1) -> xr.Dataset:
    # Stand-in for the level 1 ayc group, on dims (time, radial_dim)
    te = np.asarray(te, dtype=float)
    ne = np.asarray(ne, dtype=float)
    return xr.Dataset(
        {
            "radius": (("time", "radial_dim"), np.asarray(radius, dtype=float)),
            "te": (("time", "radial_dim"), te),
            "te_error": (("time", "radial_dim"), error_frac * np.abs(te)),
            "ne": (("time", "radial_dim"), ne),
            "ne_error": (("time", "radial_dim"), error_frac * np.abs(ne)),
        },
        coords={"time": np.asarray(times, dtype=float)},
    )


class TestThomsonDataset:
    def test_slices_outside_shot_window_dropped(self):
        timebase = np.round(np.arange(101) * 1e-3, 3)  # 0 to 100 ms
        times = [-0.05, 0.02, 0.05, 0.20]
        radius = np.tile([1.0, 1.2, 1.4], (4, 1))
        te = np.tile([500.0, 300.0, 100.0], (4, 1))
        ne = np.tile([2e19, 1e19, 5e18], (4, 1))

        ds = _thomson_dataset(TEST_SHOT, thomson_group(times, radius, te, ne), timebase)

        assert np.allclose(ds["time"].values, [0.02, 0.05])

    def test_non_positive_values_and_errors_become_nan(self):
        timebase = np.round(np.arange(101) * 1e-3, 3)
        radius = np.array([[1.0, 1.2, 1.4]])
        te = np.array([[500.0, -1.0, 100.0]])
        ne = np.array([[2e19, 1e19, 0.0]])

        ds = _thomson_dataset(
            TEST_SHOT, thomson_group([0.02], radius, te, ne), timebase
        )

        assert np.isnan(ds["ts_channel_t_e"].values[0, 1])
        assert np.isnan(ds["ts_channel_t_e_error"].values[0, 1])
        assert np.isnan(ds["ts_channel_n_e"].values[0, 2])
        assert ds["ts_channel_t_e"].values[0, 0] == 500.0

    def test_slices_with_no_usable_channel_dropped(self):
        # Some shots publish every other slice empty, radial basis included
        timebase = np.round(np.arange(101) * 1e-3, 3)
        radius = np.tile([1.0, 1.2, 1.4], (2, 1))
        te = np.array([[500.0, 300.0, 100.0], [0.0, 0.0, 0.0]])
        ne = np.array([[2e19, 1e19, 5e18], [0.0, 0.0, 0.0]])

        ds = _thomson_dataset(
            TEST_SHOT, thomson_group([0.02, 0.03], radius, te, ne), timebase
        )

        assert ds.sizes["idx"] == 1
        assert ds["time"].values[0] == 0.02

    def test_every_channel_sits_on_the_midplane_chord(self):
        timebase = np.round(np.arange(101) * 1e-3, 3)
        radius = np.array([[1.0, 1.2, 1.4]])

        ds = _thomson_dataset(
            TEST_SHOT,
            thomson_group(
                [0.02], radius, [[500.0, 300.0, 100.0]], [[2e19, 1e19, 5e18]]
            ),
            timebase,
        )

        assert (ds["ts_channel_z"].values == TS_CHANNEL_Z).all()
        assert np.allclose(ds["ts_channel_r"].values, radius)


@pytest.mark.slow  # reads the public S3 stores, ~60-90 s per shot
class TestMakeUnprocessedDataFiles:
    def test_one_shot(self, tmp_path):
        skip_without_store()
        workflow = mast_workflow(tmp_path, shotlist=[TEST_SHOT])

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == [TEST_SHOT]
        with xr.open_dataset(workflow.unprocessed_data_dir / f"{TEST_SHOT}.nc") as ds:
            # Standardized names, on the 1 kHz grid, with Thomson to fit
            for name in ("ip", "b0", "power_ohm", "n_e_line_average", "psirz"):
                assert name in ds, name
            assert "ts_channel_t_e" in ds
            times = ds["time"].values
            assert np.allclose(np.diff(times), 1e-3, atol=1e-6)

    def test_rerun_reuses_saved(self, tmp_path):
        skip_without_store()
        workflow = mast_workflow(tmp_path, shotlist=[TEST_SHOT])
        workflow.make_unprocessed_data_files()
        written = (workflow.unprocessed_data_dir / f"{TEST_SHOT}.nc").stat().st_mtime

        workflow.make_unprocessed_data_files()

        assert (
            workflow.unprocessed_data_dir / f"{TEST_SHOT}.nc"
        ).stat().st_mtime == written


@pytest.mark.slow
class TestPrepareFitInput:
    def test_channels_mapped_on_rho_tor_norm_in_fit_units(self, tmp_path):
        skip_without_store()
        workflow = mast_workflow(tmp_path, shotlist=[TEST_SHOT])
        workflow.make_unprocessed_data_files()

        with xr.open_dataset(workflow.unprocessed_data_dir / f"{TEST_SHOT}.nc") as ds:
            fit_input = workflow.prepare_fit_input(TEST_SHOT, ds)

        assert fit_input is not None
        assert fit_input.has_fittable_points()
        assert fit_input.x.shape == fit_input.te_y.shape == fit_input.ne_y.shape
        assert fit_input.time.size == fit_input.te_y.shape[0]
        finite_rho_tor_norm = fit_input.x[np.isfinite(fit_input.x)]
        assert finite_rho_tor_norm.size > 0
        assert finite_rho_tor_norm.min() >= 0.0
        # Te [keV] and ne [1e20 m^-3], not the SI values in the stored file
        assert np.nanmax(fit_input.te_y) < 100.0
        assert np.nanmax(fit_input.ne_y) < 100.0
