"""Tests for the TCV workflow.

The fast tests cover the fringe-jump correction of the FIR density,
the placement of the measured 0D signals,
reading the MATLAB v7.3 layout of a DEFUSE export (0D signals and raw Thomson),
and the LIUQE GEQDSK block, all on synthetic data.
The slow test reads one real shot, so it needs the DEFUSE exports and the MEQ databases:

    uv run pytest -m slow transport_validation_datasets/tests/test_tcv_workflow.py
"""

import h5py
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from transport_validation_datasets.machine.generic import (
    geqdsk_psi_n_grid,
    make_uniform_1kHz_timebase,
)
from transport_validation_datasets.machine.tcv.sources import (
    LIUQE_FLUX_PER_RADIAN,
    DefuseSignal,
    defuse_path,
    liuqe_geqdsk_dataset,
    meqdb_path,
    read_defuse_signals,
    read_defuse_thomson,
    read_liuqe,
)
from transport_validation_datasets.machine.tcv.tcv_dataset import (
    TCVSettings,
    _remove_fringe_jumps,
    _zero_d_dataset,
)

LIVE_SHOT = 72920

FIR_SAMPLE_STEP = 4e-5
FIR_DENSITY = 3e19


def _fir_trace(
    steps: list[tuple[float, float]], seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 s of NEavg at 25 kHz with 2e17 noise: (times, true density, trace with a fringe jump of each size at each time)."""
    sample_time = np.arange(0.0, 1.0, FIR_SAMPLE_STEP)
    rng = np.random.default_rng(seed)
    density_true = FIR_DENSITY + 2e17 * rng.standard_normal(sample_time.size)
    density_trace = density_true.copy()
    for jump_time, jump_size in steps:
        density_trace[sample_time >= jump_time] += jump_size
    return sample_time, density_true, density_trace


def test_fringe_jumps_removed_and_real_changes_kept():
    """A fringe slip is removed, a dropout that recovers and a spike that decays leave the level where it was,
    and a real 1.5e19 drop over 3 ms is kept."""
    sample_time, density_true, density_trace = _fir_trace([(0.2, -2e19)])
    mask_dropout = (sample_time >= 0.400) & (sample_time < 0.404)
    density_trace[mask_dropout] = -1e19
    time_since_spike = sample_time - 0.6
    mask_spike = time_since_spike >= 0
    spike_decay = np.exp(-time_since_spike[mask_spike] / 1e-3)
    density_trace[mask_spike] += 1.5e19 * spike_decay
    drop_fraction = np.clip((sample_time - 0.8) / 3e-3, 0.0, 1.0)
    real_drop = 1.5e19 * drop_fraction
    density_true = density_true - real_drop
    density_trace = density_trace - real_drop

    density_corrected, cut_time = _remove_fringe_jumps(sample_time, density_trace)

    assert cut_time is None
    mask_off_spike = ~((sample_time >= 0.5995) & (sample_time < 0.605))
    np.testing.assert_allclose(
        density_corrected[mask_off_spike], density_true[mask_off_spike], atol=1.5e18
    )


def test_fringe_burst_cuts_the_rest_of_the_record():
    """Three slips within FRINGE_BURST_WINDOW_S mean the interferometer lost count, NaN from the first on."""
    sample_time, density_true, density_trace = _fir_trace(
        [(0.2, -2e19), (0.22, -2e19), (0.24, 2e19)]
    )

    density_corrected, cut_time = _remove_fringe_jumps(sample_time, density_trace)

    assert cut_time == pytest.approx(0.2, abs=1e-3)
    mask_before = sample_time < 0.199
    np.testing.assert_allclose(
        density_corrected[mask_before], density_true[mask_before]
    )
    assert np.isnan(density_corrected[sample_time >= 0.2]).all()


def test_zero_d_signals_placed_causally_with_heating_summed():
    """Both beamlines sum and a missing ECRH is zero,
    PradTot bridges a 40 ms skip in its ~17 ms cadence,
    and NEavg is NaN from where the interferometer lost count."""
    timebase = make_uniform_1kHz_timebase(0.5)
    fast_time = np.arange(0.0, 0.5, 1e-4)
    ip = DefuseSignal(time=fast_time, values=np.full(fast_time.size, -2e5))
    b0 = DefuseSignal(time=fast_time, values=np.full(fast_time.size, -1.4))
    prad_time = np.array([0.017, 0.034, 0.051, 0.091, 0.108, 0.125])
    prad = DefuseSignal(time=prad_time, values=np.full(prad_time.size, 1e5))
    density_time, density_true, density_trace = _fir_trace(
        [(0.2, -2e19), (0.22, -2e19), (0.24, 2e19)]
    )
    density = DefuseSignal(time=density_time, values=density_trace)
    beam_time = np.arange(0.1, 0.3, 1e-3)
    nbi = DefuseSignal(time=beam_time, values=np.full(beam_time.size, 0.5))
    nbi2 = DefuseSignal(time=beam_time, values=np.full(beam_time.size, 0.25))
    signals = {
        "I_P": ip,
        "BZERO": b0,
        "PradTot": prad,
        "NEavg": density,
        "NBI": nbi,
        "NBI2": nbi2,
    }

    ds = _zero_d_dataset(70000, signals, timebase)

    mask_beams = (timebase >= 0.1) & (timebase <= 0.299)
    np.testing.assert_allclose(ds["power_nbi"].values[mask_beams], 0.75e6)
    assert (ds["power_nbi"].values[timebase > 0.3] == 0.0).all()
    assert (ds["power_ec"].values == 0.0).all()
    mask_skip = (timebase >= 0.051) & (timebase < 0.091)
    assert np.isfinite(ds["power_radiated"].values[mask_skip]).all()
    mask_lost_count = timebase >= 0.2005
    assert np.isnan(ds["n_e_line_average"].values[mask_lost_count]).all()
    assert np.isfinite(ds["n_e_line_average"].values[timebase < 0.199]).all()


def _write_matlab(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    matlab_class: str = "single",
    empty: bool = False,
):
    """One dataset with the attributes MATLAB v7.3 writes."""
    group[name] = data
    group[name].attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if empty:
        group[name].attrs["MATLAB_empty"] = np.uint8(1)


def test_read_defuse_signals_handles_the_matlab_layout(tmp_path):
    """MATLAB v7.3 rows come back flat, per-gyrotron ECRH rows are summed, repeated times are dropped,
    and a flagged empty array reads as absent."""
    path = tmp_path / "TCVno70000.h5"
    time = np.array([[0.0, 0.1, 0.1, 0.2, 0.3]])
    with h5py.File(path, "w") as defuse_file:
        signal_root = defuse_file.create_group("SIG")
        ip_group = signal_root.create_group("I_P")
        _write_matlab(ip_group, "signal", -np.array([[1e5, 2e5, 2e5, 3e5, 4e5]]))
        _write_matlab(ip_group, "time", time)
        ecrh_group = signal_root.create_group("ECRH")
        ecrh_rows = np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [np.nan, 0.25, 0.25, 0.25, np.nan],
            ]
        )
        _write_matlab(ecrh_group, "signal", ecrh_rows)
        _write_matlab(ecrh_group, "time", time)
        nbi_group = signal_root.create_group("NBI")
        for key in ["signal", "time"]:
            _write_matlab(nbi_group, key, np.array([1, 0], dtype=np.uint64), empty=True)

    signals = read_defuse_signals(path, ("I_P", "ECRH", "NBI", "NBI2"))

    assert set(signals) == {"I_P", "ECRH"}
    np.testing.assert_array_equal(signals["I_P"].time, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(signals["I_P"].values, [-1e5, -2e5, -3e5, -4e5])
    np.testing.assert_allclose(signals["ECRH"].values, [0.5, 1.75, 1.75, 1.5])


def _write_raw_thomson(
    signal_root: h5py.Group, group_name: str, readings: np.ndarray, errors: np.ndarray
):
    """A DEFUSE raw Thomson group, readings stored channel-first as some exports do."""
    raw = signal_root.create_group(group_name).create_group("signal/raw")
    _write_matlab(raw, "t", np.array([[0.017, 0.034, 0.051]]))
    _write_matlab(raw, "z", readings.T)
    _write_matlab(raw, "error_bar", errors.T)
    los = raw.create_group("los")
    _write_matlab(los, "rchord", np.array([[0.9, 0.9, 0.9, 0.9]]))
    _write_matlab(los, "zchord", np.array([[-0.3, -0.1, 0.1, 0.3]]))


def test_read_defuse_thomson_reads_both_variables_on_one_layout(tmp_path):
    """Readings come back time-first with DEFUSE's exclusion flags untouched, and a missing ne group reads as None."""
    te = np.full((3, 4), 500.0)
    te_error = np.full((3, 4), 50.0)
    te_error[1, 2] = -1.0
    ne = np.full((3, 4), 3e19)
    ne_error = np.full((3, 4), 3e18)
    ne[0, 0] = ne_error[0, 0] = -1.03
    path = tmp_path / "TCVno70000.h5"
    with h5py.File(path, "w") as defuse_file:
        signal_root = defuse_file.create_group("SIG")
        _write_raw_thomson(signal_root, "Te_rho", te, te_error)
        _write_raw_thomson(signal_root, "Ne_rho", ne, ne_error)
    path_no_ne = tmp_path / "TCVno70001.h5"
    with h5py.File(path_no_ne, "w") as defuse_file:
        signal_root = defuse_file.create_group("SIG")
        _write_raw_thomson(signal_root, "Te_rho", te, te_error)

    thomson = read_defuse_thomson(path)

    assert thomson.te.shape == (3, 4)
    np.testing.assert_array_equal(thomson.z_channel, [-0.3, -0.1, 0.1, 0.3])
    assert thomson.te_error[1, 2] == -1.0
    assert thomson.ne[0, 0] == pytest.approx(-1.03)
    assert read_defuse_thomson(path_no_ne) is None


def _synthetic_liuqe(n_t: int = 3) -> dict[str, np.ndarray]:
    """A LIUQE struct of circular, up-down symmetric equilibria in LIUQE's own conventions.

    psi = FA + (FB - FA) rho^2 in Wb per 2 pi radians with rho the normalized minor radius,
    q = 1 + 2 rho_pol^2 diverging at the LCFS of the last slice,
    and the middle slice without a boundary (lB 0).
    """
    r_grid = np.linspace(0.6, 1.2, 7)
    z_grid = np.linspace(-0.4, 0.4, 9)
    rho_pol_surfaces = np.linspace(0.0, 1.0, 11)
    r_axis, minor_radius = 0.9, 0.25
    flux_axis, flux_boundary = -0.1, 0.2
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    rho_squared = ((r_mesh - r_axis) ** 2 + z_mesh**2) / minor_radius**2
    flux_zr = flux_axis + (flux_boundary - flux_axis) * rho_squared
    q_surfaces = 1.0 + 2.0 * rho_pol_surfaces**2
    inverse_q = np.tile((1.0 / q_surfaces)[:, None], (1, n_t))
    inverse_q[-1, -1] = 0.0
    psi_n_surfaces = rho_pol_surfaces**2
    # F at the LCFS is rBt, the vacuum field times r0
    f_surfaces = -1.25 - 0.01 * (psi_n_surfaces - 1.0)
    flux_range = flux_boundary - flux_axis
    # dF/dpsi with psi per 2 pi radians, F dF/dpsi in LIUQE units
    df_dpsi = -0.01 / flux_range
    pressure = 2e4 * (1.0 - psi_n_surfaces)
    dp_dpsi = -2e4 / flux_range
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    contour_r = r_axis + minor_radius * np.cos(theta)
    contour_z = minor_radius * np.sin(theta)
    surfaces_shape = (rho_pol_surfaces.size, n_t)
    return {
        "t": np.array([0.1, 0.2, 0.3])[:n_t],
        "pQ": rho_pol_surfaces,
        "rx": r_grid,
        "zx": z_grid,
        "rl": np.array([0.6, 1.2, 1.2, 0.6]),
        "zl": np.array([-0.4, -0.4, 0.4, 0.4]),
        "r0": 0.88,
        "Fx": np.repeat(flux_zr[:, :, None], n_t, axis=2),
        "FA": np.full(n_t, flux_axis),
        "FB": np.full(n_t, flux_boundary),
        "rA": np.full(n_t, r_axis),
        "zA": np.zeros(n_t),
        "Ip": np.full(n_t, -2e5),
        "rBt": np.full(n_t, -1.25),
        "lB": np.array([1.0, 0.0, 1.0])[:n_t],
        "TQ": np.tile(f_surfaces[:, None], (1, n_t)),
        "PQ": np.tile(pressure[:, None], (1, n_t)),
        "TTpQ": np.tile((f_surfaces * df_dpsi)[:, None], (1, n_t)),
        "PpQ": np.full(surfaces_shape, dp_dpsi),
        "iqQ": inverse_q,
        "rB_lcfs": np.repeat(contour_r[:, None], n_t, axis=1),
        "zB_lcfs": np.repeat(contour_z[:, None], n_t, axis=1),
    }


def test_liuqe_geqdsk_block_is_per_radian_on_the_geqdsk_grid():
    """The flux map lands (t, r, z) in Wb/rad, the d/dpsi profiles scale the other way,
    q diverges where 1/q is 0, and a slice without a boundary is NaN."""
    liuqe = _synthetic_liuqe()

    ds = liuqe_geqdsk_dataset(liuqe, shot=70000)

    flux_scale = LIUQE_FLUX_PER_RADIAN
    assert ds["psirz"].dims == ("idx", "r_grid", "z_grid")
    psirz_expected = np.transpose(liuqe["Fx"][:, :, 0]) * flux_scale
    np.testing.assert_allclose(ds["psirz"].values[0], psirz_expected, rtol=1e-6)
    assert float(ds["simagx"][0]) == pytest.approx(-0.1 * flux_scale)
    psi_n_grid = geqdsk_psi_n_grid(ds.sizes["r_grid"])
    assert ds.sizes["psi_idx"] == psi_n_grid.size
    # FF' and p' against finite differences of the resampled F and p on psi per radian
    psi_grid = float(ds["simagx"][0]) + psi_n_grid * float(
        ds["sibdry"][0] - ds["simagx"][0]
    )
    fpol = ds["fpol"].values[0]
    pres = ds["pres"].values[0]
    f_squared_half = 0.5 * fpol**2
    ffprime_expected = np.gradient(f_squared_half, psi_grid)
    pprime_expected = np.gradient(pres, psi_grid)
    np.testing.assert_allclose(ds["ffprime"].values[0], ffprime_expected, rtol=1e-3)
    np.testing.assert_allclose(ds["pprime"].values[0], pprime_expected, rtol=1e-6)
    assert float(ds["fpol"][0, -1]) == pytest.approx(
        float(ds["bcentr"][0]) * float(ds["rcentr"])
    )
    assert np.isfinite(ds["qpsi"].values[0]).all()
    assert np.isinf(ds["qpsi"].values[2, -1])
    assert np.isnan(ds["psirz"].values[1]).all()
    assert np.isnan(ds["simagx"].values[1])
    assert ds.attrs["cocos"] == 7


SETTINGS = TCVSettings()
LIVE_DATA_PRESENT = (
    meqdb_path(SETTINGS.meqdb_dir, LIVE_SHOT).exists()
    and defuse_path(SETTINGS.defuse_dir, LIVE_SHOT).exists()
)


@pytest.mark.slow
@pytest.mark.skipif(
    not LIVE_DATA_PRESENT, reason="needs the DEFUSE exports and the TCV MEQ databases"
)
def test_live_liuqe_block_is_consistent_and_places_the_thomson_channels():
    """One real shot: the flux map hits simagx at the axis, F at the LCFS is bcentr rcentr,
    FF' and p' match the resampled F and p, and the channels land where DEFUSE puts them in rho_pol."""
    liuqe = read_liuqe(meqdb_path(SETTINGS.meqdb_dir, LIVE_SHOT))
    ds = liuqe_geqdsk_dataset(liuqe, LIVE_SHOT)
    times = ds["time"].values
    i_mid = int(np.argmin(np.abs(times - 0.5 * (times[0] + times[-1]))))
    eq_slice = ds.isel(idx=i_mid)
    r_grid = eq_slice["r_grid"].values
    z_grid = eq_slice["z_grid"].values
    flux_interpolator = RegularGridInterpolator(
        (r_grid, z_grid), eq_slice["psirz"].values.astype(float)
    )
    axis_position = [[float(eq_slice["rmagx"]), float(eq_slice["zmagx"])]]
    flux_at_axis = flux_interpolator(axis_position)[0]
    flux_axis = float(eq_slice["simagx"])
    flux_boundary = float(eq_slice["sibdry"])
    flux_range = flux_boundary - flux_axis
    assert abs(flux_at_axis - flux_axis) < 0.02 * abs(flux_range)
    assert float(eq_slice["fpol"][-1]) == pytest.approx(
        float(eq_slice["bcentr"]) * float(eq_slice["rcentr"])
    )
    psi_n_grid = geqdsk_psi_n_grid(eq_slice.sizes["psi_idx"])
    psi_grid = flux_axis + psi_n_grid * flux_range
    fpol = eq_slice["fpol"].values
    f_squared_half = 0.5 * fpol**2
    ffprime_finite_difference = np.gradient(f_squared_half, psi_grid)
    ffprime_ratio = eq_slice["ffprime"].values[3:-3] / ffprime_finite_difference[3:-3]
    assert np.median(ffprime_ratio) == pytest.approx(1.0, abs=0.02)

    thomson = read_defuse_thomson(defuse_path(SETTINGS.defuse_dir, LIVE_SHOT))
    with h5py.File(defuse_path(SETTINGS.defuse_dir, LIVE_SHOT), "r") as defuse_file:
        rho_pol_defuse = np.asarray(defuse_file["SIG/Te_rho/signal/raw/x"][()])
    i_slice = int(np.argmin(np.abs(thomson.time - times[i_mid])))
    channel_positions = np.column_stack([thomson.r_channel, thomson.z_channel])
    eq_at_slice = ds.isel(idx=int(np.argmin(np.abs(times - thomson.time[i_slice]))))
    slice_interpolator = RegularGridInterpolator(
        (r_grid, z_grid), eq_at_slice["psirz"].values.astype(float), bounds_error=False
    )
    flux_channels = slice_interpolator(channel_positions)
    psi_n_channels = (flux_channels - float(eq_at_slice["simagx"])) / float(
        eq_at_slice["sibdry"] - eq_at_slice["simagx"]
    )
    rho_pol_channels = np.sqrt(np.clip(psi_n_channels, 0.0, None))
    rho_pol_difference = rho_pol_channels - rho_pol_defuse[i_slice]
    assert np.nanmedian(np.abs(rho_pol_difference)) < 0.01
