"""Round-trip checks for batch_io.py's optional alternate-coordinate fields
(ShotFitInput.psi_norm/qpsi, FitBatch.fit_coordinate) added for
worker_akho.py's coordinate-substitution step (see its module docstring)."""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    FitBatch,
    ShotFitInput,
    default_fit_bounds,
    pack_fit_batch,
    unpack_fit_batch,
)


def _shot_input(with_alt_coords: bool) -> ShotFitInput:
    kwargs = dict(
        x=np.array([[0.1, 0.5, 0.9]]),
        te_y=np.array([[1.0, 2.0, 3.0]]),
        te_err=np.array([[0.1, 0.1, 0.1]]),
        ne_y=np.array([[1.0, 2.0, 3.0]]),
        ne_err=np.array([[0.1, 0.1, 0.1]]),
        time=np.array([0.0]),
    )
    if with_alt_coords:
        kwargs["psi_norm"] = np.array([[0.05, 0.3, 0.85]])
        kwargs["qpsi"] = np.array([[1.0, 2.0, 3.0, 4.0]])
    return ShotFitInput(**kwargs)


def test_shot_fit_input_defaults_to_no_alt_coordinates():
    si = _shot_input(with_alt_coords=False)
    assert si.psi_norm is None
    assert si.qpsi is None


def test_fit_batch_defaults_to_rho():
    batch = FitBatch(
        shot_inputs={1: _shot_input(False)},
        x_star=np.linspace(0, 1, 5),
        min_points=3,
        scale_per_slice=True,
        bounds=default_fit_bounds(),
    )
    assert batch.fit_coordinate == "rho"


def test_pack_unpack_round_trips_alt_coordinates(tmp_path):
    si_with = _shot_input(with_alt_coords=True)
    si_without = _shot_input(with_alt_coords=False)
    batch = FitBatch(
        shot_inputs={1: si_with, 2: si_without},
        x_star=np.linspace(0, 1.1, 10),
        min_points=3,
        scale_per_slice=True,
        bounds=default_fit_bounds(),
        fit_coordinate="psi_norm",
    )
    path = tmp_path / "batch.npz"
    pack_fit_batch(path, batch)
    back = unpack_fit_batch(path)

    assert back.fit_coordinate == "psi_norm"
    np.testing.assert_allclose(back.shot_inputs[1].psi_norm, si_with.psi_norm)
    np.testing.assert_allclose(back.shot_inputs[1].qpsi, si_with.qpsi)
    assert back.shot_inputs[2].psi_norm is None
    assert back.shot_inputs[2].qpsi is None


def test_unpack_missing_fit_coordinate_key_defaults_to_rho(tmp_path):
    """A batch file written before this schema addition has no fit_coordinate
    key at all -- it must still read back as "rho", not raise."""
    batch = FitBatch(
        shot_inputs={1: _shot_input(False)},
        x_star=np.linspace(0, 1, 5),
        min_points=3,
        scale_per_slice=True,
        bounds=default_fit_bounds(),
    )
    path = tmp_path / "batch.npz"
    pack_fit_batch(path, batch)
    back = unpack_fit_batch(path)
    assert back.fit_coordinate == "rho"
