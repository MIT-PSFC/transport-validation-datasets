"""End-to-end tests of the DataWorkflow stages on a synthetic device.

DummyWorkflow fakes the source:
a shot is a flat current on the 1 kHz grid with a Thomson sample every SAMPLE_PERIOD_MS.
Its channels sit at known rho_tor_norm and carry a synthetic shape scaled per shot.
The fits go through the linear interpolation method in linear_worker.py,
registered as "linear" for these tests,
so every stage runs in well under a second and the fitted profiles are predictable.
"""

import json
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import transport_validation_datasets.workflow as workflow_module
from transport_validation_datasets import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_validation_datasets.gp_fitting.batch_io import (
    FIT_MODE_SAMPLE,
    FIT_MODE_WINDOW_AVERAGE,
    FIT_MODE_WINDOW_SAMPLE,
    STATUS_OK,
    STATUS_SKIPPED,
    ShotFitInput,
    read_batch_setting,
    unpack_fit_batch,
)
from transport_validation_datasets.machine.generic import (
    channel_rows_at_times,
    make_uniform_1kHz_timebase,
    ts_channel_fit_rows,
)
from transport_validation_datasets.workflow import DataWorkflow

DURATION = 0.3  # s of source data per shot
SAMPLE_PERIOD_MS = 20
FIRST_SAMPLE_MS = 10
TE_AXIS = 1000.0  # eV
NE_AXIS = 1.0e20  # m^-3
# Dense enough to resolve the pedestal, and reaching into the SOL
RHO_TOR_NORM_CH = np.linspace(0.05, 1.1, 24)
# Where make_source_dataset injects the bad readings
MISFIRED_TE_CHANNEL = 3  # rho_tor_norm 0.19
NOISY_NE_CHANNEL = 7  # rho_tor_norm 0.37

# Dummy shot numbers used throughout the tests
BLACKLISTED_SHOT = 9
UNREADABLE_SHOT = 5  # the source returns None, a transient failure
SHORT_SHOT = 7  # too short to pass the filters


# Synthetic shapes, each 1 on the axis and positive on every channel


def parabola(rho_tor_norm: np.ndarray) -> np.ndarray:
    # Zero at 1.2, past the outermost channel
    return 1.0 - (rho_tor_norm / 1.2) ** 2


def pedestal(rho_tor_norm: np.ndarray) -> np.ndarray:
    # A broad core and a steep tanh pedestal at 0.95, as in H-mode
    core = 1.0 - 0.3 * rho_tor_norm**2
    tanh_edge = np.tanh((rho_tor_norm - 0.95) / 0.02)
    return core * 0.5 * (1.0 - tanh_edge)


def shot_scale(shot: int) -> float:
    # Shots differ in amplitude, so a mixed-up shot shows in the profiles
    return 1.0 + 0.1 * (shot % 10)


def make_source_dataset(
    shot: int,
    shape: Callable[[np.ndarray], np.ndarray],
    duration: float,
    broken_sample: int | None,
    bad_readings: bool,
    extra_signals: dict[str, float | np.ndarray] | None = None,
) -> xr.Dataset:
    grid = make_uniform_1kHz_timebase(duration)
    n_t = grid.size
    n_ch = RHO_TOR_NORM_CH.size
    is_sample = np.zeros(n_t, dtype=bool)
    is_sample[FIRST_SAMPLE_MS::SAMPLE_PERIOD_MS] = True
    sample_rows = np.flatnonzero(is_sample)
    shape_ch = shape(RHO_TOR_NORM_CH) * shot_scale(shot)
    te = np.full((n_t, n_ch), np.nan)
    te[is_sample] = TE_AXIS * shape_ch
    ne = np.full((n_t, n_ch), np.nan)
    ne[is_sample] = NE_AXIS * shape_ch
    # Errors are 10% of the clean readings
    te_error = 0.1 * te
    ne_error = 0.1 * ne
    if broken_sample is not None:
        # All but two channels of one sample lost, below fit_min_points
        row = sample_rows[broken_sample]
        te[row, 2:] = np.nan
        ne[row, 2:] = np.nan
    if bad_readings:
        # First sample: a Te reading at a tenth of its neighbours, and an ne error 20 times theirs
        te[sample_rows[0], MISFIRED_TE_CHANNEL] *= 0.1
        ne_error[sample_rows[0], NOISY_NE_CHANNEL] *= 20.0
        # Second sample: every channel inside rho_tor_norm 0.3 lost
        lost_in_core = RHO_TOR_NORM_CH < 0.3
        te[sample_rows[1], lost_in_core] = np.nan
        ne[sample_rows[1], lost_in_core] = np.nan

    def channel(values):
        return ((EPISODE_DIM, TIME_COORD, "ts_channel"), values[None])

    zero_d = {"ip": 1.0e6, **(extra_signals or {})}
    return xr.Dataset(
        {
            **{
                name: (
                    (EPISODE_DIM, TIME_COORD),
                    np.full((1, n_t), values, dtype=float),
                )
                for name, values in zero_d.items()
            },
            # rho_tor_norm stands in for R, see DummyWorkflow.prepare_fit_input
            "ts_channel_r": channel(np.tile(RHO_TOR_NORM_CH, (n_t, 1))),
            "ts_channel_z": channel(np.zeros((n_t, n_ch))),
            "ts_channel_t_e": channel(te),
            "ts_channel_t_e_error": channel(te_error),
            "ts_channel_n_e": channel(ne),
            "ts_channel_n_e_error": channel(ne_error),
        },
        coords={EPISODE_DIM: [shot], TIME_COORD: grid, "ts_channel": np.arange(n_ch)},
    )


class DummyWorkflow(DataWorkflow):
    valid_filter = {"ip": {"min_abs": 1.0}}
    transient_filter = {}
    end_margin = 0.01
    min_pulse_length = 0.1
    min_usable_time = 0.05
    min_segment_length = 0.01
    shot_blacklist = [BLACKLISTED_SHOT]
    fit_rho_tor_norm = np.linspace(0.0, 1.4, 29)
    fit_min_points = 3

    def __init__(self, *args, **kwargs):
        self.source_reads: list[int] = []
        self.shape = parabola
        self.broken_samples: dict[int, int] = {}
        self.bad_readings = False
        # 0D signals a shot's source carries beyond ip, a value or an array on the grid
        self.extra_signals: dict[int, dict[str, float | np.ndarray]] = {}
        super().__init__(*args, **kwargs)

    def get_shotlist_from_source(self) -> list[int]:
        return [1, 2]

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        self.source_reads.append(shot)
        if shot == UNREADABLE_SHOT:
            return None
        duration = 0.05 if shot == SHORT_SHOT else DURATION
        return make_source_dataset(
            shot,
            self.shape,
            duration,
            broken_sample=self.broken_samples.get(shot),
            bad_readings=self.bad_readings,
            extra_signals=self.extra_signals.get(shot),
        )

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        ds_shot = ds.squeeze(EPISODE_DIM, drop=True)
        has_sample = (
            ds_shot["ts_channel_t_e"]
            .notnull()
            .any(dim="ts_channel")
            .transpose(TIME_COORD)
            .values
        )
        ts_times = ds_shot[TIME_COORD].values[has_sample]
        if ts_times.size == 0:
            return None
        # The synthetic source stores each channel's rho_tor_norm in ts_channel_r,
        # so there is no equilibrium to map through
        rho_tor_norm = channel_rows_at_times(ds_shot["ts_channel_r"], ts_times)
        te_y, te_err, ne_y, ne_err = ts_channel_fit_rows(ds_shot, ts_times)
        return ShotFitInput(
            x=rho_tor_norm,
            te_y=te_y,
            te_err=te_err,
            ne_y=ne_y,
            ne_err=ne_err,
            time=ts_times,
        )


def make_workflow(tmp_path, shots=None, windows=None, **kwargs) -> DummyWorkflow:
    shotlist_file = None
    if windows is not None:
        shotlist_file = tmp_path / "shotlist.csv"
        rows = "".join(
            f"{shot},{start},{end}\n"
            for shot, shot_windows in windows.items()
            for start, end in shot_windows
        )
        shotlist_file.write_text("shot,t_start,t_end\n" + rows)
    elif shots is not None:
        shotlist_file = tmp_path / "shotlist.txt"
        shotlist_file.write_text("".join(f"{shot}\n" for shot in shots))
    kwargs.setdefault("ds_name", "dummy")
    return DummyWorkflow(
        data_assembly_dir=tmp_path,
        shotlist_file=shotlist_file,
        fit_method="linear",
        **kwargs,
    )


def run_all(workflow: DummyWorkflow) -> xr.Dataset:
    workflow.make_unprocessed_data_files()
    workflow.run_gp_fitting()
    return xr.open_zarr(workflow.stack_internal_dataset(), consolidated=True)


def shot_times(store: xr.Dataset, shot: int) -> tuple[int, np.ndarray]:
    i = int(np.flatnonzero(store[EPISODE_DIM].values == shot)[0])
    times = store[TIME_COORD].values[i]
    return i, times[np.isfinite(times)]


def expected_te(shot: int, rho_tor_norm: np.ndarray) -> np.ndarray:
    return TE_AXIS * parabola(rho_tor_norm) * shot_scale(shot)


@pytest.fixture(autouse=True)
def no_plots(monkeypatch):
    # Plotting is the slow part of both stages, test_cmod_workflow plots for real
    monkeypatch.setattr(workflow_module, "plot_unprocessed_data", lambda *a, **k: None)
    monkeypatch.setattr(workflow_module, "plot_ts_fits", lambda *a, **k: 0)


class TestMakeUnprocessedDataFiles:
    def test_writes_one_filtered_file_per_shot(self, tmp_path):
        workflow = make_workflow(tmp_path)

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == [1, 2]
        with xr.open_dataset(workflow.unprocessed_data_dir / "1.nc") as ds:
            times = ds[TIME_COORD].values
            # The end margin cuts the last 10 ms
            cutoff = DURATION - workflow.end_margin
            assert cutoff - 1.5e-3 < times.max() <= cutoff + 1e-6
            assert np.allclose(np.diff(times), 1e-3, atol=1e-6)
            n_samples = int(ds["ts_channel_t_e"].notnull().any(dim="ts_channel").sum())
            assert n_samples == 15

    def test_resumes_without_reading_the_source_again(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.make_unprocessed_data_files()
        reads = list(workflow.source_reads)

        workflow.make_unprocessed_data_files()

        assert workflow.source_reads == reads

    def test_blacklisted_never_read_unreadable_retried(self, tmp_path):
        workflow = make_workflow(tmp_path, shots=[1, BLACKLISTED_SHOT, UNREADABLE_SHOT])

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == [1]
        assert BLACKLISTED_SHOT not in workflow.source_reads
        # A None from the source is transient: not recorded, read again next run
        assert not workflow.shot_already_failed(UNREADABLE_SHOT)
        workflow.make_unprocessed_data_files()
        assert workflow.source_reads.count(UNREADABLE_SHOT) == 2

    def test_rejected_shot_recorded_and_skipped_next_run(self, tmp_path):
        workflow = make_workflow(tmp_path, shots=[SHORT_SHOT])

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == []
        assert workflow.shot_already_failed(SHORT_SHOT)
        workflow.make_unprocessed_data_files()
        assert workflow.source_reads == [SHORT_SHOT]

    def test_max_num_shots_stops_early(self, tmp_path):
        workflow = make_workflow(tmp_path, max_num_shots=1)

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == [1]

    def test_broken_record_rejected_and_recorded(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.extra_signals = {1: {"power_radiated": np.nan}}

        workflow.make_unprocessed_data_files()

        assert workflow.unprocessed_shots() == [2]
        assert workflow.shot_already_failed(1)


class TestShotRejectionReason:
    grid = make_uniform_1kHz_timebase(DURATION).astype(float)

    def kept_dataset(self, **signals) -> xr.Dataset:
        return make_source_dataset(
            1, parabola, DURATION, None, False, extra_signals=signals
        )

    def test_all_nan_signal_named_absent_signal_skipped(self, tmp_path):
        workflow = make_workflow(tmp_path)

        reason = workflow.shot_rejection_reason(
            self.kept_dataset(power_ic=0.0, power_radiated=np.nan)
        )

        assert "power_radiated" in reason
        assert workflow.shot_rejection_reason(self.kept_dataset()) is None

    def test_radiated_power_floor_only_when_set(self, tmp_path):
        workflow = make_workflow(tmp_path)
        ds = self.kept_dataset(power_radiated=3e3)
        assert workflow.shot_rejection_reason(ds) is None

        workflow.min_mean_power_radiated = 5e3

        assert "power_radiated" in workflow.shot_rejection_reason(ds)
        healthy = self.kept_dataset(power_radiated=2e5)
        assert workflow.shot_rejection_reason(healthy) is None

    def test_energy_rise_beyond_heating_rejected(self, tmp_path):
        workflow = make_workflow(tmp_path)
        # 100 kJ stored over the 0.3 s shot, peaking at the end
        energy_mhd = 1e5 * self.grid / self.grid[-1]
        # 0.3 s at 0.1 MW is 30 kJ, and the NaN half of a power record puts nothing in
        power_nbi_half_nan = np.where(self.grid < 0.15, np.nan, 0.0)
        unheated = self.kept_dataset(
            energy_mhd=energy_mhd, power_ohm=1e5, power_nbi=power_nbi_half_nan
        )
        # 0.3 s at 0.1 + 0.3 MW is 120 kJ, and a negative power takes nothing out
        heated = self.kept_dataset(
            energy_mhd=energy_mhd, power_ohm=1e5, power_nbi=3e5, power_lh=-1e6
        )

        assert "energy_mhd" in workflow.shot_rejection_reason(unheated)
        assert workflow.shot_rejection_reason(heated) is None

    def test_rise_from_first_kept_time_against_input_up_to_peak(self, tmp_path):
        workflow = make_workflow(tmp_path)
        # 0.3 MW puts in 30 kJ by the peak at 0.1 s and 90 kJ over the shot.
        # Both shots start at 200 kJ, so judging the peak itself would reject both,
        # and integrating past the peak would pass both.
        rise_shape = np.minimum(self.grid, 0.2 - self.grid) / 0.1
        small_rise = self.kept_dataset(energy_mhd=2e5 + 2e4 * rise_shape, power_ohm=3e5)
        large_rise = self.kept_dataset(energy_mhd=2e5 + 4e4 * rise_shape, power_ohm=3e5)

        assert workflow.shot_rejection_reason(small_rise) is None
        assert "energy_mhd" in workflow.shot_rejection_reason(large_rise)


class TestStageFitBatches:
    def test_plain_mode_stages_every_thomson_sample(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.make_unprocessed_data_files()

        batches = workflow.stage_fit_batches(workflow.unprocessed_shots())

        assert sorted(sum(batches.values(), [])) == [1, 2]
        batch_id = next(b for b, shots in batches.items() if shots == [1])
        batch = unpack_fit_batch(workflow._batch_in_path(batch_id))
        si = batch.shot_inputs[1]
        assert batch.fit_mode == FIT_MODE_SAMPLE
        assert si.time.size == 15
        assert np.allclose(si.x, RHO_TOR_NORM_CH)
        # Staged in the fit units: keV and 1e20 m^-3
        assert np.allclose(si.te_y[0], expected_te(1, RHO_TOR_NORM_CH) * 1e-3)
        assert np.allclose(si.ne_y[0], parabola(RHO_TOR_NORM_CH) * shot_scale(1))
        assert si.windows.shape == (0, 2)
        assert (si.window_index == -1).all()

    def test_bad_readings_drop_both_variables_and_pedestal_survives(self, tmp_path):
        workflow = make_workflow(tmp_path, shots=[1])
        workflow.shape = pedestal
        workflow.bad_readings = True
        workflow.make_unprocessed_data_files()

        batches = workflow.stage_fit_batches([1])

        (batch_id,) = batches
        si = unpack_fit_batch(workflow._batch_in_path(batch_id)).shot_inputs[1]
        # The misfired Te and the noisy ne each take the other variable's reading along,
        # and nothing else in the sample goes, the steep pedestal included
        for y in (si.te_y[0], si.ne_y[0]):
            dropped = np.flatnonzero(np.isnan(y))
            assert dropped.tolist() == [MISFIRED_TE_CHANNEL, NOISY_NE_CHANNEL]
        # A sample with no channel in the core is not fit at all
        assert np.isnan(si.te_y[1]).all()
        assert np.isnan(si.ne_y[1]).all()
        pedestal_ch = pedestal(RHO_TOR_NORM_CH)
        pedestal_te_keV = TE_AXIS * 1e-3 * pedestal_ch * shot_scale(1)
        assert np.allclose(si.te_y[2:], pedestal_te_keV)

    def test_windowed_mode_keeps_only_samples_inside_windows(self, tmp_path):
        workflow = make_workflow(tmp_path, windows={1: [(0.1, 0.2)]})
        workflow.make_unprocessed_data_files()

        batches = workflow.stage_fit_batches([1])

        (batch_id,) = batches
        si = unpack_fit_batch(workflow._batch_in_path(batch_id)).shot_inputs[1]
        assert np.allclose(si.time, [0.11, 0.13, 0.15, 0.17, 0.19])
        assert (si.window_index == 0).all()
        assert np.array_equal(si.windows, [[0.1, 0.2]])
        assert read_batch_setting(workflow._batch_in_path(batch_id), "fit_mode") == (
            FIT_MODE_WINDOW_SAMPLE
        )

    def test_shot_without_window_not_staged(self, tmp_path):
        # Both shots have unprocessed files, the windowed shotlist only lists one
        make_workflow(tmp_path).make_unprocessed_data_files()
        workflow = make_workflow(tmp_path, windows={1: [(0.1, 0.2)]})

        batches = workflow.stage_fit_batches([1, 2])

        assert sum(batches.values(), []) == [1]
        assert not workflow.fit_already_failed(2)

    def test_average_mode_pools_samples_of_each_window(self, tmp_path):
        workflow = make_workflow(
            tmp_path, windows={1: [(0.1, 0.2), (0.2, 0.28)]}, average_windows=True
        )
        workflow.make_unprocessed_data_files()

        batches = workflow.stage_fit_batches([1])

        (batch_id,) = batches
        batch = unpack_fit_batch(workflow._batch_in_path(batch_id))
        si = batch.shot_inputs[1]
        assert batch.fit_mode == FIT_MODE_WINDOW_AVERAGE
        assert np.allclose(si.time, [0.15, 0.24])
        # Five samples in the first window, four in the second, padded to five
        n_ch = RHO_TOR_NORM_CH.size
        assert si.x.shape == (2, 5 * n_ch)
        assert np.isfinite(si.x[0]).all()
        assert np.isfinite(si.x[1, : 4 * n_ch]).all()
        assert np.isnan(si.x[1, 4 * n_ch :]).all()
        assert si.window_index.tolist() == [0, 1]

    def test_shot_with_no_sample_in_windows_skipped(self, tmp_path):
        workflow = make_workflow(tmp_path, windows={1: [(0.0, 0.005)]})
        workflow.make_unprocessed_data_files()

        batches = workflow.stage_fit_batches([1])

        assert batches == {}
        assert not workflow.fit_already_failed(1)

    def test_average_without_windows_refused(self, tmp_path):
        with pytest.raises(ValueError, match="average_windows"):
            make_workflow(tmp_path, shots=[1], average_windows=True)

    def test_existing_batch_in_another_mode_refused(self, tmp_path):
        plain = make_workflow(tmp_path)
        plain.make_unprocessed_data_files()
        plain.stage_fit_batches([1])
        windowed = make_workflow(tmp_path, windows={1: [(0.1, 0.2)]})

        with pytest.raises(ValueError, match="fit mode"):
            windowed.run_gp_fitting()


class TestRunGpFitting:
    def test_writes_one_fit_file_per_shot_matching_profiles(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.make_unprocessed_data_files()

        workflow.run_gp_fitting()

        for shot in (1, 2):
            with xr.open_dataset(workflow.fit_shots_dir / f"{shot}.nc") as ds:
                # The fit grid runs to 1.4 and is cut at 1.1 when written
                assert ds.sizes == {EPISODE_DIM: 1, TIME_DIM: 15, "rho_tor_norm": 23}
                assert np.isclose(ds["rho_tor_norm"].values[-1], 1.1)
                assert (ds["t_e_fit_status"].values == STATUS_OK).all()
                assert (ds["n_e_fit_status"].values == STATUS_OK).all()
                assert ds.attrs["fit_method"] == "linear"
                assert ds.attrs["fit_mode"] == FIT_MODE_SAMPLE
                assert json.loads(ds.attrs["windows"]) == []
                assert (ds["window_index"].values == -1).all()
                rho_tor_norm = ds["rho_tor_norm"].values
                inside = (rho_tor_norm >= RHO_TOR_NORM_CH[0]) & (
                    rho_tor_norm <= RHO_TOR_NORM_CH[-1]
                )
                # Linear interpolation of a parabola between the channels
                assert np.allclose(
                    ds["t_e"].values[0][:, inside],
                    expected_te(shot, rho_tor_norm[inside]),
                    atol=0.01 * TE_AXIS,
                )

    def test_rerun_neither_refits_nor_rewrites(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()
        outputs = sorted(workflow.fit_batches_dir.glob("*_out_linear.npz"))
        fits = sorted(workflow.fit_shots_dir.glob("*.nc"))
        before = [p.stat().st_mtime_ns for p in outputs + fits]

        workflow.run_gp_fitting()

        assert [p.stat().st_mtime_ns for p in outputs + fits] == before

    def test_clean_fit_state_removes_batches_and_results(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        workflow.clean_fit_state()

        assert not workflow.fit_batches_dir.exists()
        assert not workflow.fit_shots_dir.exists()
        assert workflow.unprocessed_shots() == [1, 2]

    def test_windowed_fit_file_carries_windows(self, tmp_path):
        windows = {1: [(0.1, 0.2), (0.24, 0.289)]}
        workflow = make_workflow(tmp_path, windows=windows)
        workflow.make_unprocessed_data_files()

        workflow.run_gp_fitting()

        with xr.open_dataset(workflow.fit_shots_dir / "1.nc") as ds:
            assert ds.attrs["fit_mode"] == FIT_MODE_WINDOW_SAMPLE
            assert json.loads(ds.attrs["windows"]) == [[0.1, 0.2], [0.24, 0.289]]
            assert ds["window_index"].values[0].tolist() == [0] * 5 + [1] * 2
            assert np.allclose(
                ds[TIME_COORD].values[0], [0.11, 0.13, 0.15, 0.17, 0.19, 0.25, 0.27]
            )


class TestStackInternalDataset:
    def test_plain_store_holds_every_shot_on_grid(self, tmp_path):
        workflow = make_workflow(tmp_path)

        store = run_all(workflow)

        assert store.sizes[EPISODE_DIM] == 2
        assert store.attrs["fit_mode"] == FIT_MODE_SAMPLE
        for shot in (1, 2):
            i, times = shot_times(store, shot)
            with xr.open_dataset(workflow.unprocessed_data_dir / f"{shot}.nc") as ds:
                assert np.array_equal(times, ds[TIME_COORD].values)
            fresh = store["fresh_profile"].values[i, : times.size]
            assert np.allclose(times[fresh > 0], np.arange(0.01, 0.30, 0.02))
            te = store["t_e"].values[i, : times.size]
            # Every sample is held to the next one, so the profile is there
            # from the first sample to the end of the grid
            assert np.isfinite(te).any(axis=-1).tolist() == (times >= 0.01).tolist()
            rho_tor_norm = store["rho_tor_norm"].values
            inside = (rho_tor_norm >= RHO_TOR_NORM_CH[0]) & (
                rho_tor_norm <= RHO_TOR_NORM_CH[-1]
            )
            assert np.allclose(
                te[-1, inside],
                expected_te(shot, rho_tor_norm[inside]),
                atol=0.01 * TE_AXIS,
            )
        assert (store["fresh_equilibrium"].values == 0).all()

    def test_windowed_store_holds_stop_at_edges(self, tmp_path):
        windows = {1: [(0.1, 0.2), (0.24, 0.289)], 2: [(0.05, 0.1)]}
        workflow = make_workflow(tmp_path, windows=windows)

        store = run_all(workflow)

        assert store.attrs["fit_mode"] == FIT_MODE_WINDOW_SAMPLE
        i, times = shot_times(store, 1)
        assert np.allclose(
            times, np.concatenate([np.arange(100, 201), np.arange(240, 290)]) * 1e-3
        )
        has_profile = np.isfinite(store["t_e"].values[i, : times.size]).any(axis=-1)
        # First window: no sample before 0.11, the 0.19 sample is held to the
        # window's end. Second window: nothing crosses in from before 0.24,
        # the 0.25 and 0.27 samples fill the rest
        expected = ((times > 0.1095) & (times < 0.2005)) | (times > 0.2495)
        assert has_profile.tolist() == expected.tolist()
        assert int(store["fresh_profile"].values[i, : times.size].sum()) == 7
        i, times = shot_times(store, 2)
        assert np.allclose(times, np.arange(50, 101) * 1e-3)
        assert np.isfinite(store["t_e"].values[i, : times.size]).any(axis=-1).all()
        assert int(store["fresh_profile"].values[i, : times.size].sum()) == 3

    def test_average_store_fills_window_fresh_at_center(self, tmp_path):
        workflow = make_workflow(
            tmp_path, windows={1: [(0.1, 0.2)]}, average_windows=True
        )

        store = run_all(workflow)

        assert store.attrs["fit_mode"] == FIT_MODE_WINDOW_AVERAGE
        i, times = shot_times(store, 1)
        assert np.allclose(times, np.arange(100, 201) * 1e-3)
        te = store["t_e"].values[i, : times.size]
        assert np.isfinite(te).all()
        assert (te == te[0]).all()
        fresh = store["fresh_profile"].values[i, : times.size]
        assert np.allclose(times[fresh > 0], [0.15])
        rho_tor_norm = store["rho_tor_norm"].values
        inside = (rho_tor_norm >= RHO_TOR_NORM_CH[0]) & (
            rho_tor_norm <= RHO_TOR_NORM_CH[-1]
        )
        assert np.allclose(
            te[0, inside], expected_te(1, rho_tor_norm[inside]), atol=0.01 * TE_AXIS
        )
        assert "center" in store["fresh_profile"].attrs["description"]

    def test_unfit_slices_dropped_by_default_and_kept_on_request(self, tmp_path):
        workflow = make_workflow(tmp_path, shots=[1])
        workflow.broken_samples = {1: 3}  # the sample at 0.07 s has 2 channels
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        dropped = xr.open_zarr(workflow.stack_internal_dataset(), consolidated=True)
        i, times = shot_times(dropped, 1)
        at = int(np.flatnonzero(np.isclose(times, 0.07))[0])
        assert int(dropped["fresh_profile"].values[i, : times.size].sum()) == 14
        assert dropped["fresh_profile"].values[i, at] == 0
        # Held from the 0.05 s sample, as though there were no sample at 0.07 s
        assert np.isfinite(dropped["t_e"].values[i, at]).all()
        assert dropped["t_e_fit_status"].values[i, at] == STATUS_OK

        kept = xr.open_zarr(
            workflow.stack_internal_dataset(drop_unfit_slices=False), consolidated=True
        )
        assert int(kept["fresh_profile"].values[i, : times.size].sum()) == 15
        assert kept["t_e_fit_status"].values[i, at] == STATUS_SKIPPED
        assert np.isnan(kept["t_e"].values[i, at]).all()

    def test_check_added_after_unprocessed_stage_drops_shot(self, tmp_path):
        workflow = make_workflow(tmp_path)
        workflow.extra_signals = {
            1: {"power_radiated": 3e3},
            2: {"power_radiated": 2e5},
        }
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        # The files on disk were written without a floor, the stack applies it now
        workflow.min_mean_power_radiated = 5e3
        store = xr.open_zarr(workflow.stack_internal_dataset(), consolidated=True)

        assert store[EPISODE_DIM].values.tolist() == [2]

    def test_shots_excluded_after_files_written_left_out(self, tmp_path):
        workflow = make_workflow(tmp_path, shots=[1, 2, 3])
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        workflow.first_shot = 2
        workflow.shot_blacklist = [3]
        store = xr.open_zarr(workflow.stack_internal_dataset(), consolidated=True)

        assert store[EPISODE_DIM].values.tolist() == [2]

    def test_density_far_off_interferometer_drops_shot(self, tmp_path):
        workflow = make_workflow(tmp_path)
        # The fitted parabola averages ~0.77 of its axis value over rho_tor_norm 0-1,
        # so shot 2's interferometer reads ~5x its Thomson
        workflow.extra_signals = {
            1: {"n_e_line_average": NE_AXIS * shot_scale(1)},
            2: {"n_e_line_average": 4.0 * NE_AXIS * shot_scale(2)},
        }
        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        workflow.density_ratio_bounds = (0.5, 1.3)
        store = xr.open_zarr(workflow.stack_internal_dataset(), consolidated=True)

        assert store[EPISODE_DIM].values.tolist() == [1]

    def test_publish_strips_raw_channels(self, tmp_path):
        workflow = make_workflow(tmp_path)
        run_all(workflow)

        published = xr.open_zarr(workflow.publish_dataset(), consolidated=True)

        assert "ts_channel_t_e" not in published
        assert "ts_channel" not in published.dims
        assert "t_e" in published
        assert published.attrs["stripped_signals"].startswith("ts_channel")

    def test_fit_results_in_another_mode_refused(self, tmp_path):
        plain = make_workflow(tmp_path)
        plain.make_unprocessed_data_files()
        plain.run_gp_fitting()
        windowed = make_workflow(tmp_path, windows={1: [(0.1, 0.2)]})

        with pytest.raises(ValueError, match="fit mode"):
            windowed.stack_internal_dataset()

    def test_fit_results_for_other_windows_refused(self, tmp_path):
        first = make_workflow(tmp_path, windows={1: [(0.1, 0.2)]})
        first.make_unprocessed_data_files()
        first.run_gp_fitting()
        edited = make_workflow(tmp_path, windows={1: [(0.1, 0.22)]})

        with pytest.raises(ValueError, match="other time windows"):
            edited.run_gp_fitting()
        with pytest.raises(ValueError, match="other time windows"):
            edited.stack_internal_dataset()
