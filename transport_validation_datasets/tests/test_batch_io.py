"""The fit settings a staged batch is compared on, and the mode of an atomic write."""

import os

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    FitAnchors,
    FitBatch,
    FitBounds,
    ShotFitInput,
    UnprocessedStamp,
    atomic_write,
    batch_settings,
    fit_batch_settings,
    pack_fit_batch,
)


def _anchors(scale: float) -> FitAnchors:
    return FitAnchors(
        value=np.array([[1.3, 0.0, 0.05 * scale]]),
        grad=np.array([[1.3, 0.0, 0.5 * scale]]),
    )


def _batch() -> FitBatch:
    rng = np.random.default_rng(0)
    shot_input = ShotFitInput(
        x=rng.uniform(0.0, 1.1, (3, 5)),
        te_y=rng.uniform(0.1, 2.0, (3, 5)),
        te_err=np.full((3, 5), 0.1),
        ne_y=rng.uniform(0.1, 1.0, (3, 5)),
        ne_err=np.full((3, 5), 0.05),
        time=np.array([0.5, 0.6, 0.7]),
    )
    return FitBatch(
        shot_inputs={7: shot_input},
        x_star=np.linspace(0.0, 1.3, 14),
        min_points=4,
        scale_per_slice=True,
        bounds={"te": FitBounds(l1_min=0.3), "ne": FitBounds(var_max=5.0)},
        anchors={"te": _anchors(1.0), "ne": _anchors(0.2)},
        # An integer, as TOML gives it, against the float the file stores
        pedestal_rho_tor_norm=1,
        sol_extension="secant",
        fit_mode="window_sample",
    )


def test_staged_settings_match_the_batch_and_a_changed_knob_does_not(tmp_path):
    path = tmp_path / "batch_abc.npz"
    batch = _batch()
    unprocessed_stamps = {7: UnprocessedStamp(digest="0" * 64, mtime_ns=0)}
    pack_fit_batch(path, batch, unprocessed_stamps)

    settings = batch_settings(path)

    assert settings == fit_batch_settings(batch)
    changed_bounds = {**batch.bounds, "te": FitBounds()}
    changed = FitBatch(**{**vars(batch), "bounds": changed_bounds})
    changed_settings = fit_batch_settings(changed)
    differing = [key for key in settings if changed_settings[key] != settings[key]]
    assert differing == ["bounds"]


def test_atomic_write_mode_matches_plain_write(tmp_path):
    # Under umask 022 a plain write gives 0o644, mkstemp's temporary would leave 0o600
    umask_before = os.umask(0o022)
    try:
        plain_path = tmp_path / "plain.txt"
        plain_path.write_text("x")
        atomic_path = tmp_path / "atomic.txt"
        atomic_write(atomic_path, lambda tmp_file: tmp_file.write_text("x"))
    finally:
        os.umask(umask_before)
    assert atomic_path.stat().st_mode == plain_path.stat().st_mode
    assert list(tmp_path.glob("*.tmp")) == []
