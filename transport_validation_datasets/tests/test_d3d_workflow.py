"""Tests for the DIII-D workflow.

The fast tests cover the IDA database priority, the strict DISPY EFIT selection and timebase,
the EFIT and density physics methods on stub connections,
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

from transport_validation_datasets.dispy_utils import register_verbose_level
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


def test_invalid_efit_slices_are_held_over():
    """A slice failing chisq never reaches the grid, the slice before holds over it."""
    register_verbose_level()
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
    wmhd_expected[20] = 19.0
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
    register_verbose_level()
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
