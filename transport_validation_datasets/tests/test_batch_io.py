"""Round trips of the staged batch and result files, and their settings strings."""

import numpy as np
import pytest

from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_OK,
    STATUS_SKIPPED,
    FitAnchors,
    FitBatch,
    FitBounds,
    ShotFitInput,
    ShotFitOutput,
    batch_settings,
    fit_batch_settings,
    pack_fit_batch,
    pack_fit_results,
    read_batch_setting,
    read_batch_shots,
    read_batch_windows,
    unpack_fit_batch,
    unpack_fit_results,
)

X_STAR = np.linspace(0.0, 1.3, 14)


def _anchors(scale: float) -> FitAnchors:
    return FitAnchors(
        value=np.array([[1.3, 0.0, 0.05 * scale]]),
        grad=np.array([[1.3, 0.0, 0.5 * scale]]),
    )


def _batch() -> FitBatch:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.1, (3, 5))
    windowed = ShotFitInput(
        x=x,
        te_y=rng.uniform(0.1, 2.0, (3, 5)),
        te_err=np.full((3, 5), 0.1),
        ne_y=rng.uniform(0.1, 1.0, (3, 5)),
        ne_err=np.full((3, 5), 0.05),
        time=np.array([0.5, 0.6, 0.7]),
        windows=np.array([[0.45, 0.65], [0.65, 0.75]]),
        window_index=np.array([0, 0, 1]),
    )
    plain = ShotFitInput(
        x=x[:2],
        te_y=np.full((2, 5), np.nan),
        te_err=np.full((2, 5), 0.1),
        ne_y=rng.uniform(0.1, 1.0, (2, 5)),
        ne_err=np.full((2, 5), 0.05),
        time=np.array([1.0, 1.1]),
    )
    return FitBatch(
        shot_inputs={7: windowed, 5: plain},
        x_star=X_STAR,
        min_points=4,
        scale_per_slice=True,
        bounds={"te": FitBounds(l1_min=0.3), "ne": FitBounds(var_max=5.0)},
        anchors={"te": _anchors(1.0), "ne": _anchors(0.2)},
        pedestal_rho_tor_norm=0.95,
        sol_extension="secant",
        fit_mode="window_sample",
    )


def test_batch_round_trip(tmp_path):
    path = tmp_path / "batch_abc.npz"
    batch = _batch()

    pack_fit_batch(path, batch)
    unpacked = unpack_fit_batch(path)

    assert not list(tmp_path.glob("*.tmp"))
    assert set(unpacked.shot_inputs) == {5, 7}
    assert unpacked.bounds == batch.bounds
    assert (unpacked.min_points, unpacked.scale_per_slice) == (4, True)
    assert (unpacked.fit_mode, unpacked.sol_extension) == ("window_sample", "secant")
    assert unpacked.pedestal_rho_tor_norm == 0.95
    np.testing.assert_array_equal(unpacked.x_star, X_STAR)
    for var in ("te", "ne"):
        np.testing.assert_array_equal(
            unpacked.anchors[var].value, batch.anchors[var].value
        )
        np.testing.assert_array_equal(
            unpacked.anchors[var].grad, batch.anchors[var].grad
        )
    windowed = unpacked.shot_inputs[7]
    np.testing.assert_allclose(windowed.te_y, batch.shot_inputs[7].te_y, rtol=1e-6)
    np.testing.assert_array_equal(windowed.windows, batch.shot_inputs[7].windows)
    assert windowed.window_index.tolist() == [0, 0, 1]
    plain = unpacked.shot_inputs[5]
    assert np.isnan(plain.te_y).all()
    assert plain.windows.shape == (0, 2)
    assert plain.window_index.tolist() == [-1, -1]

    assert read_batch_shots(path) == [5, 7]
    assert read_batch_setting(path, "fit_mode") == "window_sample"
    windows = read_batch_windows(path)
    assert windows[5].shape == (0, 2)
    np.testing.assert_array_equal(windows[7], batch.shot_inputs[7].windows)


def test_batch_settings_match_the_packed_batch(tmp_path):
    path = tmp_path / "batch_abc.npz"
    batch = _batch()
    pack_fit_batch(path, batch)

    settings = batch_settings(path)

    assert settings == fit_batch_settings(batch)
    assert set(settings) == {
        "fit_mode",
        "sol_extension",
        "pedestal_rho_tor_norm",
        "anchors",
        "bounds",
        "x_star",
        "min_points",
        "scale_per_slice",
    }
    assert all(isinstance(value, str) for value in settings.values())
    # A changed knob shows up in the string, a repacked identical batch does not
    changed = FitBatch(**{**vars(batch), "bounds": {**batch.bounds, "te": FitBounds()}})
    assert fit_batch_settings(changed)["bounds"] != settings["bounds"]
    assert fit_batch_settings(changed)["anchors"] == settings["anchors"]


def test_missing_bounds_key_is_an_error(tmp_path):
    path = tmp_path / "batch_abc.npz"
    pack_fit_batch(path, _batch())
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != "bounds:te:l1_min"}
    np.savez(path, **arrays)

    with pytest.raises(KeyError, match="bounds:te:l1_min"):
        unpack_fit_batch(path)


def test_results_round_trip(tmp_path):
    path = tmp_path / "batch_abc_out_zk.npz"
    with_hyps = ShotFitOutput.empty(2, X_STAR.size, np.array([0.5, 0.6]))
    with_hyps.te_fit[0] = 1.0
    with_hyps.te_status[0] = STATUS_OK
    with_hyps.te_hyps = np.full((2, 4), np.nan)
    with_hyps.te_hyps[0] = [2.0, 0.5, 0.3, 0.1]
    without_hyps = ShotFitOutput.empty(1, X_STAR.size, np.array([1.0]))

    pack_fit_results(path, {7: with_hyps, 5: without_hyps}, X_STAR)
    unpacked = unpack_fit_results(path)

    assert not list(tmp_path.glob("*.tmp"))
    assert set(unpacked) == {5, 7}
    assert unpacked[7].te_status.tolist() == [STATUS_OK, STATUS_SKIPPED]
    assert (unpacked[7].te_fit[0] == 1.0).all()
    assert np.isnan(unpacked[7].te_fit[1]).all()
    np.testing.assert_allclose(unpacked[7].te_hyps[0], [2.0, 0.5, 0.3, 0.1], rtol=1e-6)
    assert unpacked[7].ne_hyps is None
    assert unpacked[5].te_hyps is None
    assert unpacked[5].time.tolist() == [1.0]
