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
    _branch_disagreement_errors,
    _inboard_channels,
    _ohmic_power,
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


class TestOhmicPower:
    def test_resistive_power_is_ip_times_resistive_voltage(self):
        # Flat current, so the inductive term vanishes and P = Ip * V_loop
        timebase = np.linspace(0.0, 0.1, 11)
        ip = np.full(timebase.size, 4.0e5)
        v_loop = np.full(timebase.size, 2.0)

        power = _ohmic_power(
            summary_time=timebase,
            ip=ip,
            eq_time=timebase,
            li=np.full(timebase.size, 1.0),
            r_axis=np.full(timebase.size, 0.9),
            v_loop=v_loop,
            timebase=timebase,
        )

        assert np.allclose(power, 4.0e5 * 2.0)

    def test_negative_power_clipped_to_zero(self):
        # A negative loop voltage would give negative ohmic power,
        # which means the inductive term overshot
        timebase = np.linspace(0.0, 0.1, 11)

        power = _ohmic_power(
            summary_time=timebase,
            ip=np.full(timebase.size, 4.0e5),
            eq_time=timebase,
            li=np.full(timebase.size, 1.0),
            r_axis=np.full(timebase.size, 0.9),
            v_loop=np.full(timebase.size, -2.0),
            timebase=timebase,
        )

        assert (power == 0.0).all()

    def test_rising_current_costs_inductive_power(self):
        timebase = np.linspace(0.0, 0.1, 51)
        rising = np.linspace(1.0e5, 5.0e5, timebase.size)
        flat = np.full(timebase.size, 3.0e5)
        args = dict(
            eq_time=timebase,
            li=np.full(timebase.size, 1.0),
            r_axis=np.full(timebase.size, 0.9),
            v_loop=np.full(timebase.size, 5.0),
            timebase=timebase,
        )

        ramp = _ohmic_power(summary_time=timebase, ip=rising, **args)
        steady = _ohmic_power(summary_time=timebase, ip=flat, **args)

        # Compared at the point the two currents cross, so only dIp/dt differs
        i_mid = timebase.size // 2
        assert ramp[i_mid] < steady[i_mid]


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


def two_branch_slice(inboard_offset: float = 0.0, n_outboard: int = 60):
    # One slice of a chord through the axis at R = 0.85 m, with rho linear in |R - 0.85|.
    # 60 inboard channels, the MAST core spacing, and n_outboard outboard ones, interleaved in rho.
    r_inboard = np.linspace(0.30, 0.845, 60)
    r_outboard = np.linspace(0.852, 1.40, n_outboard)
    r_channel = np.concatenate([r_inboard, r_outboard])[None, :]
    rho = np.abs(r_channel - 0.85) / 0.55
    profile = 1.0 - 0.8 * rho**2
    y = np.where(r_channel < 0.85, profile + inboard_offset, profile)
    err = np.full(y.shape, 0.01)
    return rho, r_channel, y, err


class TestBranchDisagreementErrors:
    def test_offset_branches_get_half_the_offset_in_quadrature(self):
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1)
        inboard = _inboard_channels(rho, r_channel)

        err_out = _branch_disagreement_errors(rho, y, err, inboard)

        assert (inboard == (r_channel < 0.85)).all()
        # The two ends of the overlap have fewer than BRANCH_MIN_CHANNELS estimates in reach
        interior = (rho > 0.05) & (rho < 0.95)
        expected = np.hypot(0.01, 0.05)
        assert np.allclose(err_out[interior], expected, rtol=1e-2)

    def test_sparse_branch_is_inflated_like_the_dense_one(self):
        # 20 outboard channels ~0.05 apart in rho, like the MAST outboard edge,
        # too few for a window of their own branch
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1, n_outboard=20)
        inboard = _inboard_channels(rho, r_channel)

        err_out = _branch_disagreement_errors(rho, y, err, inboard)

        interior_outboard = ~inboard & (rho > 0.05) & (rho < 0.95)
        expected = np.hypot(0.01, 0.05)
        assert np.allclose(err_out[interior_outboard], expected, rtol=1e-2)

    def test_channels_past_the_overlap_take_the_nearest_disagreement(self):
        # The inboard branch stops at rho 0.7, the outboard one runs on to rho 1
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1)
        inboard = _inboard_channels(rho, r_channel)
        y[inboard & (rho > 0.7)] = np.nan

        err_out = _branch_disagreement_errors(rho, y, err, inboard)

        # They share the nearest estimate, so the added error is one fraction of each value
        just_past = ~inboard & (rho > 0.75) & (rho < 0.85)
        added = np.sqrt(err_out[just_past] ** 2 - 0.01**2)
        added_over_value = added / y[just_past]
        assert (added > 0.01).all()
        assert np.allclose(added_over_value, added_over_value[0])
        far_past = ~inboard & (rho > 0.95)
        assert np.allclose(err_out[far_past], 0.01)

    def test_agreeing_branches_keep_their_errors(self):
        rho, r_channel, y, err = two_branch_slice()
        inboard = _inboard_channels(rho, r_channel)

        err_out = _branch_disagreement_errors(rho, y, err, inboard)

        assert np.allclose(err_out, err, rtol=1e-2)

    def test_one_spike_inflates_nothing(self):
        # The running median outvotes the spike, on its own branch and on the other,
        # and leaves it to the spike screen with its own error
        rho, r_channel, y, err = two_branch_slice()
        i_spike = 90
        y[0, i_spike] += 0.5
        inboard = _inboard_channels(rho, r_channel)

        err_out = _branch_disagreement_errors(rho, y, err, inboard)

        assert np.allclose(err_out, err, rtol=1e-2)


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
