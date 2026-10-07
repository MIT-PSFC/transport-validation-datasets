"""The zk fit checks on synthetic slices, and one real MAST slice through every fit method.

The fast tests run the zk quality checks (zk/quality.py)
and the mkgp blending fix (zk/gp.py MKGP_BLEND_DX) on synthetic channels.
The slow test fits MAST 28956 t=0.179 s with every registered method's fit_batch.
That slice is the regression case for the envelope check's channel span,
whose SOL ringing past the outermost ne channel was culled as an overshoot,
and for the short-l1 basin ~15 percent below the innermost Te channels.

The fixture holds the staged (float32) channel arrays, so the data-derived
restart seeds reproduce the production fit bit for bit. It is generated on
first use: the shot is pulled from the public MAST store and staged by the
workflow itself (stage_fit_batches, the shared cleaning included), and the
slice is read back from the batch file and cached in test_outputs/.
"""

import numpy as np
import pytest

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_OK,
    STATUS_REPAIRED,
    FitBatch,
    FitBounds,
    ShotFitInput,
    unpack_fit_batch,
)
from transport_validation_datasets.gp_fitting.registry import (
    WORKER_MODULES,
    load_worker,
)
from transport_validation_datasets.gp_fitting.zk.gp import run_gp
from transport_validation_datasets.gp_fitting.zk.quality import (
    fit_ignores_data,
    nonphysical_peak,
    rise_is_data_supported,
)
from transport_validation_datasets.machine.mast.mast_dataset import (
    MASTDataWorkflow,
    MASTSettings,
)
from transport_validation_datasets.workflow import (
    STORED_RHO_TOR_NORM_MAX,
    _fit_anchors,
)

FIXTURE = PACKAGE_ROOT / "tests" / "test_outputs" / "mast_28956_t0p179.npz"
SHOT = 28956
SLICE_TIME = 0.179

# The MAST staging knobs the production fit of this slice ran under.
MAST_BOUNDS = MASTDataWorkflow.fit_bounds
MAST_SETTINGS = MASTSettings()
MAST_ANCHORS = _fit_anchors(MAST_SETTINGS)
X_STAR = MASTDataWorkflow.fit_rho_tor_norm
MIN_POINTS = 10


def ensure_fixture():
    # Build the fixture from the public MAST store on first use. float32, the
    # dtype batch staging writes, so the data-derived restart seeds match the
    # production fit exactly.
    if FIXTURE.exists():
        return
    from transport_validation_datasets.machine.mast.mast_dataset import (
        LEVEL2_PATH,
        _store_path_exists,
    )

    if not _store_path_exists(f"{LEVEL2_PATH}/{SHOT}.zarr"):
        pytest.skip("no cached fixture and the MAST level 2 store is not reachable")
    staging_dir = FIXTURE.parent / "fixture_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    shotlist_file = staging_dir / "shotlist.txt"
    shotlist_file.write_text(f"{SHOT}\n")
    workflow = MASTDataWorkflow(
        ds_name="fit_methods_fixture",
        data_assembly_dir=staging_dir,
        shotlist_file=shotlist_file,
    )
    workflow.make_unprocessed_data_files()
    (batch_id,) = workflow.stage_fit_batches([SHOT])
    staged_batch = unpack_fit_batch(workflow._batch_in_path(batch_id))
    fit_input = staged_batch.shot_inputs[SHOT]
    i = int(np.argmin(np.abs(fit_input.time - SLICE_TIME)))
    np.savez_compressed(
        FIXTURE,
        x=np.asarray(fit_input.x[i], dtype=np.float32),
        te_y=np.asarray(fit_input.te_y[i], dtype=np.float32),
        te_err=np.asarray(fit_input.te_err[i], dtype=np.float32),
        ne_y=np.asarray(fit_input.ne_y[i], dtype=np.float32),
        ne_err=np.asarray(fit_input.ne_err[i], dtype=np.float32),
        time=np.float32(fit_input.time[i]),
    )


def fixture_batch() -> FitBatch:
    ensure_fixture()
    data = np.load(FIXTURE)
    one = lambda name: np.asarray(data[name], dtype=np.float64)[None]  # noqa: E731
    shot_input = ShotFitInput(
        x=one("x"),
        te_y=one("te_y"),
        te_err=one("te_err"),
        ne_y=one("ne_y"),
        ne_err=one("ne_err"),
        time=np.array([float(data["time"])]),
    )
    return FitBatch(
        shot_inputs={SHOT: shot_input},
        x_star=X_STAR,
        min_points=MIN_POINTS,
        scale_per_slice=True,
        bounds=MAST_BOUNDS,
        anchors=MAST_ANCHORS,
        pedestal_rho_tor_norm=MAST_SETTINGS.pedestal_rho_tor_norm,
        sol_extension=MAST_SETTINGS.sol_extension,
    )


class TestEnvelopeCheckSpan:
    # A flat profile with an edge drop: channels end at rho 1.0 with the SOL
    # values near zero, the way the fixture slice's ne looks
    x_ch = np.linspace(0.0, 1.0, 21)
    y_ch = np.where(x_ch < 0.8, 1.0, 1.0 - 4.5 * (x_ch - 0.8))
    err_ch = np.full_like(x_ch, 0.05)

    def test_sol_ringing_past_outermost_channel_not_flagged(self):
        # The fit rings up to several times the local (near-zero) envelope,
        # but only beyond the outermost channel, where there is no data left
        x = np.linspace(0.0, 1.1, 45)
        y = np.interp(x, self.x_ch, self.y_ch)
        y[x > 1.0] = 0.06  # SOL tail: tiny in absolute terms, huge relatively

        assert nonphysical_peak(y, x, self.x_ch, self.y_ch, self.err_ch) is None

    def test_hump_between_channels_still_flagged(self):
        # An invented interpolation hump inside the channel span must still
        # be caught (the C-Mod 1160503008 t=1.311 class)
        x = np.linspace(0.0, 1.1, 45)
        y = np.interp(x, self.x_ch, self.y_ch)
        y[(x > 0.35) & (x < 0.55)] = 2.0

        peak = nonphysical_peak(y, x, self.x_ch, self.y_ch, self.err_ch)

        assert peak is not None
        assert 0.35 < peak < 0.55


class TestCoreBiasCheck:
    # Four core channels over a peaked profile, and a fit of the same shape scaled down
    x_star = np.linspace(0.0, 1.1, 45)
    x_ch = np.array([0.05, 0.15, 0.25, 0.35])
    y_ch = 1.0 - 0.8 * x_ch**2

    def fit_scaled(self, factor):
        return factor * (1.0 - 0.8 * self.x_star**2)

    def test_collapse_flagged(self):
        err_ch = np.full(4, 0.1)
        fit = self.fit_scaled(0.6)

        assert fit_ignores_data(self.x_ch, self.y_ch, err_ch, self.x_star, fit)

    def test_small_miss_of_precise_channels_not_flagged(self):
        # ~9 sigma per channel, but only 3 percent of the value
        err_ch = np.full(4, 0.003)
        fit = self.fit_scaled(0.97)

        assert not fit_ignores_data(self.x_ch, self.y_ch, err_ch, self.x_star, fit)

    def test_large_miss_of_noisy_channels_not_flagged(self):
        # 30 percent under, but within one error bar of each channel
        err_ch = np.full(4, 0.5)
        fit = self.fit_scaled(0.7)

        assert not fit_ignores_data(self.x_ch, self.y_ch, err_ch, self.x_star, fit)

    def test_huge_error_channel_does_not_hide_collapse(self):
        # Unweighted, the mean z is ~2.2 with the huge-error channel near 0.
        # Weighted by 1/err^2, the other three carry the significance.
        err_ch = np.array([0.13, 5.0, 0.13, 0.13])
        fit = self.fit_scaled(0.6)

        assert fit_ignores_data(self.x_ch, self.y_ch, err_ch, self.x_star, fit)


class TestRiseSupport:
    # A linear rise of slope 0.5 through rho 0.7, as on the flank of a hollow ne profile
    rho = 0.7

    @pytest.mark.parametrize(
        ("x_ch", "err", "supported"),
        [
            # Eight precise channels within reach of rho
            (np.linspace(0.55, 0.85, 16), 0.02, True),
            # The same channels, the rise well inside their errors
            (np.linspace(0.55, 0.85, 16), 0.5, False),
            # Three channels within reach, a pedestal-shoulder bump
            (np.array([0.3, 0.5, 0.66, 0.7, 0.74, 0.9]), 0.02, False),
        ],
    )
    def test_rise_needs_many_channels_and_significance(self, x_ch, err, supported):
        y_ch = 0.5 + 0.5 * x_ch
        err_ch = np.full_like(x_ch, err)

        assert rise_is_data_supported(x_ch, y_ch, err_ch, self.rho) == supported


class TestPairedChannels:
    def test_paired_channels_keep_their_weight(self):
        # Two branches 0.003 apart in rho, as MAST's inboard and outboard channels interleave.
        # mkgp's blending would merge each pair with an error near the data maximum
        # and pull the core toward the data mean (see zk/gp.py MKGP_BLEND_DX).
        x_branch = np.linspace(0.05, 1.0, 40)
        x_ch = np.concatenate([x_branch, x_branch + 0.003])
        y_ch = 1.0 - 0.9 * x_ch**2
        err_ch = np.full(x_ch.shape, 0.02)
        x_eval = np.array([0.05, 0.3])
        hyperparams = np.array([1.0, 0.3, 0.3, 0.1])

        gp = run_gp(
            x_ch,
            y_ch,
            err_ch,
            x_eval,
            FitBounds(),
            MAST_ANCHORS["te"],
            MAST_SETTINGS.pedestal_rho_tor_norm,
            hyperparams=hyperparams,
            optimize=False,
        )
        mean = np.asarray(gp.get_gp_mean(), dtype=float).ravel()

        expected = 1.0 - 0.9 * x_eval**2
        assert np.allclose(mean, expected, atol=0.03)


@pytest.mark.slow  # full GP fit of the slice, minutes per method
@pytest.mark.parametrize("method", sorted(WORKER_MODULES))
def test_regression_slice_fits_usable(method):
    worker = load_worker(method)
    batch = fixture_batch()

    try:
        outputs = worker.fit_batch(batch, num_workers=1)
    except NotImplementedError:
        pytest.skip(f"fit method '{method}' is not implemented yet")

    so = outputs[SHOT]
    # Only what is written is held to non-negative values.
    # Past it akho's SOL may dip a hair below zero toward the outer anchors.
    stored = X_STAR <= STORED_RHO_TOR_NORM_MAX + 1e-9
    for name in ("te", "ne"):
        status = int(getattr(so, f"{name}_status")[0])
        assert status in (STATUS_OK, STATUS_REPAIRED), f"{name} status {status}"
        fit = getattr(so, f"{name}_fit")[0]
        assert np.isfinite(fit).all()
        assert (fit[stored] >= 0.0).all()
