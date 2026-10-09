"""Tests for the DIII-D workflow.

The fast tests cover the IDA database priority and file reader, the strict DISPY EFIT selection and timebase,
the EFIT, GEQDSK, density and heating physics methods on stub connections,
and the mapping of IDA slices onto rho_tor_norm on a synthetic equilibrium.
The slow test pulls one real shot, so it needs the IDA databases and the DIII-D data servers (omega):

    uv run pytest -m slow transport_validation_datasets/tests/test_d3d_workflow.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSettingParams
from disruption_py.settings.nickname_setting import NicknameSettingParams

from transport_validation_datasets.dispy_utils import _register_verbose_level
from transport_validation_datasets.machine.d3d.d3d_dataset import (
    D3DDataWorkflow,
)
from transport_validation_datasets.machine.d3d.dispy_methods import (
    D3DMethods,
    DispyEfitNicknameSetting,
    Uniform1kHzTimeSetting,
)
from transport_validation_datasets.machine.d3d.ida import (
    IdaDatabase,
    find_ida_path,
    find_ida_shots,
    ida_dataset,
)
from transport_validation_datasets.machine.generic import make_uniform_1kHz_timebase

IDA_DIR = Path("/fusion/projects/results/ida-results/HBP_database")


def test_ida_databases_priority_wildcards_shotlists_and_union(tmp_path):
    """The first database with a file wins, wildcards match VVUQ-style names, a database with a shotlist
    serves only its shots, and the default shotlist is the union of what every database serves."""
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    wild = tmp_path / "wild"
    general = tmp_path / "general"
    for directory in [primary, fallback, wild, general]:
        directory.mkdir()
    databases = [
        IdaDatabase(str(primary / "IDA_{shot}_.cdf")),
        IdaDatabase(str(fallback / "ida{shot}.nc")),
        IdaDatabase(str(wild / "IDA_{shot}_*_.cdf")),
        IdaDatabase(str(general / "IDA_{shot}_.cdf"), shots=frozenset({2, 5})),
    ]
    files = [
        primary / "IDA_2_.cdf",
        fallback / "ida1.nc",
        fallback / "ida2.nc",
        wild / "IDA_4_0.2_4.0_.cdf",
        general / "IDA_2_.cdf",
        general / "IDA_5_.cdf",
        general / "IDA_6_.cdf",
    ]
    for path in files:
        path.touch()

    assert find_ida_path(2, databases) == primary / "IDA_2_.cdf"
    assert find_ida_path(1, databases) == fallback / "ida1.nc"
    assert find_ida_path(3, databases) is None
    assert find_ida_path(4, databases) == wild / "IDA_4_0.2_4.0_.cdf"
    assert find_ida_path(5, databases) == general / "IDA_5_.cdf"
    assert find_ida_path(6, databases) is None
    assert find_ida_shots(databases) == [1, 2, 4, 5]


class _StubDatabase:
    """Answers the code_rundb query with fixed rows and records it."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def query(self, query, use_pandas=True):
        self.queries.append(query)
        return self.rows


def _nickname_params(database) -> NicknameSettingParams:
    return NicknameSettingParams(
        shot_id=199051,
        mds_conn=None,
        database=database,
        disruption_time=None,
        tokamak=Tokamak.D3D,
    )


def test_dispy_nickname_takes_latest_run_and_raises_without_one():
    """The latest run of the runtag is the EFIT tree, and a shot without one fails instead of falling back to efit01."""
    database = _StubDatabase([("EFIT04",), ("EFIT07",)])

    tree = DispyEfitNicknameSetting("DISPY").get_tree_name(_nickname_params(database))

    assert tree == "EFIT07"
    assert "runtag = 'DISPY'" in database.queries[0]
    with pytest.raises(ValueError, match="no EFIT run under runtag DISPY"):
        DispyEfitNicknameSetting("DISPY").get_tree_name(
            _nickname_params(_StubDatabase([]))
        )


class _StubEfitConnection:
    """Serves one atime array [ms] for every get_data call."""

    def __init__(self, atime_ms):
        self.atime_ms = atime_ms

    def get_data(self, path, tree_name=None):
        return self.atime_ms


def _time_params(atime_ms) -> TimeSettingParams:
    connection = _StubEfitConnection(atime_ms)
    return TimeSettingParams(
        shot_id=199051,
        mds_conn=connection,
        database=None,
        disruption_time=None,
        tokamak=Tokamak.D3D,
    )


def test_time_setting_is_1khz_to_the_efit_end_and_rejects_slow_efit():
    """A 1 kHz EFIT gives the uniform timebase out to its last slice, a 50 Hz one (not DISPY) fails the shot."""
    atime_1khz_ms = np.arange(100.0, 5001.0)
    times = Uniform1kHzTimeSetting().get_times(_time_params(atime_1khz_ms))

    assert times[0] == 0.0
    assert times[-1] == pytest.approx(5.0)
    np.testing.assert_allclose(np.diff(times), 1e-3, atol=1e-6)
    atime_50hz_ms = np.arange(100.0, 5001.0, 20.0)
    with pytest.raises(ValueError, match="not a 1 kHz reconstruction"):
        Uniform1kHzTimeSetting().get_times(_time_params(atime_50hz_ms))


class _StubEfitScalarConnection:
    """Serves 1 kHz a-eqdsk nodes over 100-200 ms, with wmhd counting the slices.

    Slice 20 fails chisq.
    """

    def __init__(self):
        n_slices = 101
        self.nodes = {
            "atime": np.arange(100.0, 201.0),
            "chisq": np.full(n_slices, 5.0),
            "wmhd": np.arange(n_slices, dtype=float),
            "volume": np.full(n_slices, 20.0),
            "aminor": np.full(n_slices, 0.6),
            "bcentr": np.full(n_slices, -2.0),
            "ipmhd": np.full(n_slices, 1.2e6),
        }
        self.nodes["chisq"][20] = 100.0

    def get_data(self, path, tree_name=None):
        node = path.split(":")[-1]
        return self.nodes[node].copy()


def test_efit_scalars_land_on_their_slices_and_invalid_ones_are_nan():
    """Each slice lands on its own grid time, and a slice failing chisq is NaN there.
    The workflow holds the last usable reconstruction over it (DataWorkflow.add_equilibrium_signals)."""
    _register_verbose_level()
    connection = _StubEfitScalarConnection()
    times = connection.nodes["atime"] / 1e3
    params = PhysicsMethodParams(
        shot_id=199051,
        tokamak=Tokamak.D3D,
        disruption_time=None,
        mds_conn=connection,
        times=times,
    )

    wmhd = D3DMethods.get_efit_scalars(params=params)["wmhd"]

    wmhd_expected = np.arange(101.0)
    wmhd_expected[20] = np.nan
    np.testing.assert_allclose(wmhd, wmhd_expected)


class _StubDensityConnection:
    """Serves \\density [cm^-3] (or raises TreeNODATA when it is None) and PTDATA dssdenest [1e19 m^-3], both on 0-1000 ms."""

    def __init__(self, density_cm3):
        self.density_cm3 = density_cm3
        self.time_ms = np.linspace(0.0, 1000.0, 11)

    def get_data_with_dims(self, path, tree_name=None):
        if path == r"\density":
            if self.density_cm3 is None:
                raise mdsExceptions.TreeNODATA()
            return np.full(self.time_ms.size, self.density_cm3), self.time_ms
        assert "dssdenest" in path
        return np.full(self.time_ms.size, 5.0), self.time_ms


def _line_average_density(density_cm3) -> np.ndarray:
    # disruption-py's physics_method decorator logs at the VERBOSE level
    _register_verbose_level()
    times = np.linspace(0.1, 0.9, 5)
    connection = _StubDensityConnection(density_cm3)
    params = PhysicsMethodParams(
        shot_id=199264,
        tokamak=Tokamak.D3D,
        disruption_time=None,
        mds_conn=connection,
        times=times,
    )
    return D3DMethods.get_line_average_density(params=params)["n_e_line_average"]


def test_line_average_density_falls_back_to_pcs_estimate():
    """\\density [cm^-3] where the DISPY tree has it, else dssdenest [1e19 m^-3] (199264 has no \\density)."""
    np.testing.assert_allclose(_line_average_density(4e13), 4e19)
    np.testing.assert_allclose(_line_average_density(None), 5e19)
    np.testing.assert_allclose(_line_average_density(np.nan), 5e19)


class _StubGeqdskConnection:
    """Serves a DISPY EFIT of four 1 kHz slices over 100-103 ms, the third failing chisq.

    psi = r + 10 z, so the flux map's orientation shows,
    the boundary is padded with (0, 0) points as EFIT pads it,
    and the limiter is stored as two rows of R and Z with one padding point.
    """

    def __init__(self):
        n_t, n_psi = 4, 5
        self.r_grid = np.linspace(1.0, 2.4, 3)
        self.z_grid = np.linspace(-1.2, 1.2, 4)
        z_mesh, r_mesh = np.meshgrid(self.z_grid, self.r_grid, indexing="ij")
        # MDS returns psirz (T, z, r)
        psi_zr = r_mesh + 10.0 * z_mesh
        boundary_r = np.array([1.2, 2.2, 2.2, 1.2, 0.0, 0.0])
        boundary_z = np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0])
        geqdsk = {
            "rmaxis": np.full(n_t, 1.7),
            "zmaxis": np.zeros(n_t),
            "ssimag": np.full(n_t, -0.5),
            "ssibry": np.full(n_t, 0.1),
            "bcentr": np.full(n_t, -2.0),
            "cpasma": np.full(n_t, 1.2e6),
            "fpol": np.full((n_t, n_psi), -3.4),
            "pres": np.tile(np.linspace(1e5, 0.0, n_psi), (n_t, 1)),
            "ffprim": np.zeros((n_t, n_psi)),
            "pprime": np.full((n_t, n_psi), -1e5),
            "qpsi": np.tile(np.linspace(1.0, 4.0, n_psi), (n_t, 1)),
            "psirz": np.repeat(psi_zr[None], n_t, axis=0),
            "rbbbs": np.tile(boundary_r, (n_t, 1)),
            "zbbbs": np.tile(boundary_z, (n_t, 1)),
        }
        self.nodes = {
            r"\efit_a_eqdsk:atime": np.arange(100.0, 104.0),
            r"\efit_a_eqdsk:chisq": np.array([5.0, 5.0, 100.0, 5.0]),
            r"\top.results.geqdsk:r": self.r_grid,
            r"\top.results.geqdsk:z": self.z_grid,
            r"\top.results.geqdsk:rzero": np.full(n_t, 1.6955),
            r"\top.results.geqdsk:lim": np.array(
                [[1.0, 2.4, 2.4, 1.0, 0.0], [-1.3, -1.3, 1.3, 1.3, 0.0]]
            ),
            **{
                rf"\top.results.geqdsk:{name}": values
                for name, values in geqdsk.items()
            },
        }

    def get_data(self, path, tree_name=None):
        return np.array(self.nodes[path], dtype=float)


def test_geqdsk_block_masks_failed_slices_and_contour_padding():
    """The flux map lands (t, r, z), the slice failing chisq is NaN,
    and the padding of the boundary and the limiter is NaN with the limiter rows read as R and Z."""
    _register_verbose_level()
    connection = _StubGeqdskConnection()
    params = PhysicsMethodParams(
        shot_id=199051,
        tokamak=Tokamak.D3D,
        disruption_time=None,
        mds_conn=connection,
        times=make_uniform_1kHz_timebase(0.103),
    )

    ds = D3DMethods.get_geqdsk_parameters(params=params)

    slice_rows = [
        int(np.argmin(np.abs(params.times - t))) for t in (0.1, 0.101, 0.102, 0.103)
    ]
    simagx = ds["simagx"].values[slice_rows]
    np.testing.assert_allclose(simagx[[0, 1, 3]], -0.5)
    assert np.isnan(simagx[2])
    assert int(np.isfinite(ds["simagx"].values).sum()) == 3
    psirz = ds["psirz"].values[slice_rows[0]]
    r_mesh, z_mesh = np.meshgrid(connection.r_grid, connection.z_grid, indexing="ij")
    np.testing.assert_allclose(psirz, r_mesh + 10.0 * z_mesh)
    rbdry = ds["rbdry"].values[slice_rows[0]]
    assert np.isfinite(rbdry[:4]).all()
    assert np.isnan(rbdry[4:]).all()
    np.testing.assert_allclose(ds["rlim"].values, [1.0, 2.4, 2.4, 1.0, np.nan])
    np.testing.assert_allclose(ds["zlim"].values, [-1.3, -1.3, 1.3, 1.3, np.nan])
    assert float(ds["rcentr"]) == pytest.approx(1.6955)


class _StubHeatingConnection:
    """Serves pinj [kW] over 100-300 ms and an echpwrc [W] closed by a stray t = 0 sample, both on ms clocks."""

    def get_data_with_dims(self, path, tree_name=None):
        if path == r"\top.nb:pinj":
            time_ms = np.arange(100.0, 301.0)
            return np.full(time_ms.size, 2000.0), time_ms
        time_ms = np.append(np.arange(150.0, 251.0), 0.0)
        power = np.append(np.full(time_ms.size - 1, 5e5), 9e9)
        return power, time_ms


def test_heating_powers_are_zero_outside_their_records():
    """pinj comes in W, and the stray t = 0 sample closing echpwrc is dropped rather than placed."""
    _register_verbose_level()
    times = make_uniform_1kHz_timebase(0.4)
    params = PhysicsMethodParams(
        shot_id=199051,
        tokamak=Tokamak.D3D,
        disruption_time=None,
        mds_conn=_StubHeatingConnection(),
        times=times,
    )

    powers = D3DMethods.get_heating_powers(params=params)

    mask_beams = (times >= 0.1005) & (times <= 0.3)
    np.testing.assert_allclose(powers["p_nbi"][mask_beams], 2e6)
    assert (powers["p_nbi"][times < 0.0995] == 0.0).all()
    mask_ech = (times >= 0.1505) & (times <= 0.25)
    np.testing.assert_allclose(powers["p_ech"][mask_ech], 5e5)
    assert (powers["p_ech"][times > 0.2505] == 0.0).all()
    assert (powers["p_ech"][times < 0.1495] == 0.0).all()


def test_ida_file_read_in_seconds_with_its_points_per_slice(tmp_path):
    """Slices are sorted and in seconds, the psi_N points repeat per slice, the errors ride along."""
    psi_n = np.linspace(0.0, 1.1, 12)
    time_ms = np.array([300.0, 100.0, 200.0])
    te = np.tile(np.linspace(2000.0, 10.0, psi_n.size), (time_ms.size, 1))
    te[0] *= 2.0
    ida_file = xr.Dataset(
        {
            "T_e": (("time", "psi_n"), te),
            "T_e_err": (("time", "psi_n"), 0.1 * te),
            "n_e": (("time", "psi_n"), np.full(te.shape, 5e19)),
            "n_e_err": (("time", "psi_n"), np.full(te.shape, 2e18)),
        },
        coords={"time": time_ms, "psi_n": psi_n},
    )
    path = tmp_path / "IDA_199051_.cdf"
    ida_file.to_netcdf(path)

    ds = ida_dataset(path, 199051)

    np.testing.assert_allclose(ds["time"].values, [0.1, 0.2, 0.3])
    assert ds["ida_psi_n"].dims == ("idx", "ida_point")
    np.testing.assert_allclose(ds["ida_psi_n"].values[2], psi_n)
    np.testing.assert_allclose(ds["ida_t_e"].values[2], te[0])
    np.testing.assert_allclose(ds["ida_t_e_error"].values[2], 0.1 * te[0])
    assert (ds["shot"].values == 199051).all()


# IDA slice times on the grid [s]: the third has no usable reconstruction in reach,
# the fourth an unconstrained core Te
IDA_SLICE_TIMES = np.array([0.020, 0.040, 0.060, 0.080])
UNMAPPED_TIME = 0.060
UNCONSTRAINED_TIME = 0.080


def _unprocessed_shot() -> xr.Dataset:
    """A DIII-D unprocessed shot on a 0.1 s grid: constant q = 2 equilibria and four IDA slices.

    With constant q, Phi_N = psi_N, so each IDA point sits at rho_tor_norm = sqrt(psi_N).
    """
    times = make_uniform_1kHz_timebase(0.1)
    n_t = times.size
    n_psi = 9
    r_grid = np.linspace(1.0, 2.4, 5)
    z_grid = np.linspace(-1.2, 1.2, 7)
    psi_n_points = np.linspace(0.0, 1.1, 23)
    simagx = np.full(n_t, -0.5)
    mask_no_equilibrium = np.abs(times - UNMAPPED_TIME) < 0.003
    simagx[mask_no_equilibrium] = np.nan
    te_rows = np.full((n_t, psi_n_points.size), np.nan)
    te_error_rows = np.full_like(te_rows, np.nan)
    for slice_time in IDA_SLICE_TIMES:
        i_time = int(np.argmin(np.abs(times - slice_time)))
        te_rows[i_time] = 2000.0 * (1.0 - 0.8 * psi_n_points)
        te_error_rows[i_time] = 100.0
    i_unconstrained = int(np.argmin(np.abs(times - UNCONSTRAINED_TIME)))
    te_error_rows[i_unconstrained, 0] = 1500.0
    ne_rows = np.where(np.isfinite(te_rows), 5e19, np.nan)
    ne_error_rows = np.where(np.isfinite(te_rows), 2e18, np.nan)
    psi_n_rows = np.tile(psi_n_points, (n_t, 1))
    ds = xr.Dataset(
        data_vars={
            "simagx": (("time",), simagx),
            "sibdry": (("time",), np.full(n_t, 0.1)),
            "qpsi": (("time", "psi_idx"), np.full((n_t, n_psi), 2.0)),
            "psirz": (
                ("time", "r_grid", "z_grid"),
                np.zeros((n_t, r_grid.size, z_grid.size)),
            ),
            "ida_psi_n": (("time", "ida_point"), psi_n_rows),
            "ida_t_e": (("time", "ida_point"), te_rows),
            "ida_t_e_error": (("time", "ida_point"), te_error_rows),
            "ida_n_e": (("time", "ida_point"), ne_rows),
            "ida_n_e_error": (("time", "ida_point"), ne_error_rows),
        },
        coords={"time": times, "r_grid": r_grid, "z_grid": z_grid},
    )
    return ds.expand_dims(shot=[199051])


def test_ida_slices_map_through_the_nearest_equilibrium_and_drop_an_unconstrained_core(
    tmp_path,
):
    """Constant q puts each point at sqrt(psi_N), a slice with no reconstruction in reach stays unmapped,
    and a slice whose axis Te error is over half of Te there is emptied in both variables."""
    workflow = D3DDataWorkflow(
        ds_name="d3d_test",
        data_assembly_dir=tmp_path,
        shotlist_file=_shotlist(tmp_path),
    )

    fit_input = workflow.prepare_fit_input(199051, _unprocessed_shot())

    np.testing.assert_allclose(fit_input.time, IDA_SLICE_TIMES, atol=1e-6)
    psi_n_points = np.linspace(0.0, 1.1, 23)
    np.testing.assert_allclose(fit_input.x[0], np.sqrt(psi_n_points), atol=1e-6)
    assert np.isnan(fit_input.x[2]).all()
    np.testing.assert_allclose(fit_input.te_y[0, 0], 2.0)
    assert np.isnan(fit_input.te_y[3]).all()
    assert np.isnan(fit_input.ne_y[3]).all()
    assert np.isfinite(fit_input.ne_y[1]).all()


def _shotlist(tmp_path) -> Path:
    shotlist = tmp_path / "shotlist.txt"
    shotlist.write_text("199051\n")
    return shotlist


def test_only_the_ida_method_serves_diii_d(tmp_path):
    with pytest.raises(ValueError, match="'zk'"):
        D3DDataWorkflow(
            ds_name="d3d_test",
            data_assembly_dir=tmp_path,
            shotlist_file=_shotlist(tmp_path),
            fit_method="zk",
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not IDA_DIR.exists(),
    reason="needs the /fusion IDA databases and DIII-D data server access",
)
def test_live_single_shot(tmp_path):
    """One real shot read from source: physical ranges, the GEQDSK block, and the IDA readings."""
    workflow = D3DDataWorkflow(
        ds_name="d3d_live",
        data_assembly_dir=tmp_path,
        max_num_shots=1,
        fit_method="ida",
    )
    shot = workflow.shotlist[0]

    ds = workflow.get_source_dataset(shot)

    assert ds is not None, f"shot {shot} could not be read, see the log"
    assert "mdsthin" in sys.modules or "MDSplus" in sys.modules
    ip_peak = float(np.abs(ds["ip"]).max())
    assert 3e5 < ip_peak < 2.5e6
    assert 1.0 < float(np.abs(ds["b0"]).median()) < 2.5
    assert 0.5 < float(ds["minor_radius"].median()) < 0.7
    assert 1e19 < float(ds["n_e_line_average"].median()) < 1.5e20
    assert ds.attrs["cocos"] in (1, 3, 5, 7)
    assert np.isfinite(ds["simagx"]).sum() > 1000
    assert ds["ida_t_e"].notnull().any("ida_point").sum() > 10
