"""End-to-end IMAS export chain on synthetic shots, read back through imas.

Builds a self-consistent synthetic C-Mod-like shot (unprocessed dataset with
the staged GEQDSK block plus 0D signals, and a fit-result dataset shaped
like workflow._shot_fit_dataset's output), runs the whole export chain --
geqdsk writing, per-shot COCOS identification, conversion to the DD's own
convention, IDS building and writing -- then imports every written file
back through imas.DBEntry and checks the physics-defining signs and values.

Three sign sets run, each entering under the cocos attribute its unprocessed file would carry
(see machine.generic.cocos_from_signs), and all must land on the same target convention:
C-Mod's EFIT pins psi increasing for either field polarity,
so normal-field shots are COCOS 7 and reversed-field ones COCOS 1.
MAST's EFIT has psi decreasing with Ip > 0, B0 < 0 and q > 0, COCOS 3.

Needs the `imas` extra (imas-python, eqdsk); skipped without it.
"""

import numpy as np
import pytest
import xarray as xr

imas = pytest.importorskip("imas")
pytest.importorskip("eqdsk")

from transport_validation_datasets.gp_fitting.batch_io import STATUS_OK  # noqa: E402
from transport_validation_datasets.imas_export.scenario_export import (  # noqa: E402
    DD_VERSION,
    _sigma_bp,
    _target_cocos,
    build_imas_from_shot,
    write_ids,
)
from transport_validation_datasets.machine.generic import (  # noqa: E402
    rho_tor_norm_from_psi_n,
)
from transport_validation_datasets.workflow import (  # noqa: E402
    DATASET_EQUILIBRIUM_SIGNALS,
    TIME_COORD,
    TIME_DIM,
)

R0, A_MINOR = 0.68, 0.22
N_PSI = 33
N_GRID = 33
N_BDRY = 64
N_LIM = 72
EQ_TIMES = (0.520, 0.550, 0.580)
TS_TIMES = (0.531, 0.561)
N_RHO_TOR_NORM = 56
SOL_EXTENSION = "secant"
QPSI = 1.0 + 2.0 * np.linspace(0.0, 1.0, N_PSI) ** 2


def synthetic_unprocessed(
    ip_sign: int, b0_sign: int, psi_sign: int, cocos: int
) -> xr.Dataset:
    """One shot's unprocessed dataset: 0D signals + the GEQDSK block.

    Circular flux surfaces around (R0, 0), psi running axis-to-boundary along psi_sign,
    q > 0, F on sign(B0), and pprime/ffprime on the sign that makes p and |F| fall outward.
    """
    ip0 = ip_sign * 0.8e6
    b0 = b0_sign * 5.4
    psi0, dpsi = -0.05, psi_sign * 0.11

    time = np.round(np.arange(0.500, 0.6001, 0.001), 6)
    n_t = time.size
    eq_rows = np.isin(time, np.asarray(EQ_TIMES))

    psi_n = np.linspace(0.0, 1.0, N_PSI)
    pres = 5.0e4 * (1.0 - psi_n) + 1.0e3
    pprime = np.full(N_PSI, -5.0e4 / dpsi)
    fpol = R0 * b0 * (1.0 + 0.05 * (1.0 - psi_n))
    ffprime = fpol * (R0 * b0 * (-0.05) / dpsi)

    r_grid = np.linspace(0.40, 0.96, N_GRID)
    z_grid = np.linspace(-0.28, 0.28, N_GRID)
    rr, zz = np.meshgrid(r_grid, z_grid, indexing="ij")
    psirz = psi0 + dpsi * (((rr - R0) ** 2 + zz**2) / A_MINOR**2)

    theta_b = np.linspace(0.0, 2.0 * np.pi, N_BDRY)
    theta_l = np.linspace(0.0, 2.0 * np.pi, N_LIM)

    def on_eq_times(values: np.ndarray, dims: tuple) -> tuple:
        full = np.full((n_t, *np.shape(values)), np.nan)
        full[eq_rows] = values
        return (("time", *dims), full)

    scalars = {
        "rmagx": R0,
        "zmagx": 0.0,
        "simagx": psi0,
        "sibdry": psi0 + dpsi,
        "bcentr": b0,
        "current": ip0,
        "rcentr": R0,
        "rleft": float(r_grid[0]),
        "rdim": float(r_grid[-1] - r_grid[0]),
        "zmid": 0.0,
        "zdim": float(z_grid[-1] - z_grid[0]),
    }
    profiles = {
        "fpol": (fpol, ("psi_idx",)),
        "pres": (pres, ("psi_idx",)),
        "ffprime": (ffprime, ("psi_idx",)),
        "pprime": (pprime, ("psi_idx",)),
        "qpsi": (QPSI, ("psi_idx",)),
        "psirz": (psirz, ("r_grid", "z_grid")),
        "rbdry": (R0 + A_MINOR * np.cos(theta_b), ("boundary_idx",)),
        "zbdry": (A_MINOR * np.sin(theta_b), ("boundary_idx",)),
        "rlim": (R0 + 0.245 * np.cos(theta_l), ("limiter_idx",)),
        "zlim": (0.245 * np.sin(theta_l), ("limiter_idx",)),
    }
    data_vars = {name: on_eq_times(value, ()) for name, value in scalars.items()}
    data_vars |= {
        name: on_eq_times(value, dims) for name, (value, dims) in profiles.items()
    }
    assert set(data_vars) == set(DATASET_EQUILIBRIUM_SIGNALS)
    data_vars["ip"] = (("time",), np.full(n_t, ip0))
    data_vars["b0"] = (("time",), np.full(n_t, b0))
    return xr.Dataset(
        data_vars=data_vars, coords={"time": time}, attrs={"cocos": cocos, "r0": R0}
    )


def synthetic_fit(shot: int) -> xr.Dataset:
    """One shot's fit-result dataset, shaped like _shot_fit_dataset saves."""
    rho_tor_norm = np.linspace(0.0, 1.1, N_RHO_TOR_NORM)
    n_slices = len(TS_TIMES)
    te_profile = 1400.0 * np.clip(1.0 - 0.85 * rho_tor_norm**2, 0.0, None) + 60.0
    ne_profile = (0.9 - 0.7 * rho_tor_norm**2) * 1.0e20 + 5.0e18
    te = np.tile(te_profile, (n_slices, 1))
    ne = np.tile(ne_profile, (n_slices, 1))
    statuses = np.full((1, n_slices), STATUS_OK, dtype=np.int8)

    def profile(values: np.ndarray) -> tuple:
        return (("shot", TIME_DIM, "rho_tor_norm"), values[None].astype(np.float32))

    return xr.Dataset(
        data_vars={
            "t_e": profile(te),
            "t_e_error": profile(0.1 * te),
            "n_e": profile(ne),
            "n_e_error": profile(0.1 * ne),
            "t_e_fit_status": (("shot", TIME_DIM), statuses),
            "n_e_fit_status": (("shot", TIME_DIM), statuses),
        },
        coords={
            "shot": [shot],
            TIME_DIM: np.arange(n_slices),
            "rho_tor_norm": rho_tor_norm,
            TIME_COORD: (
                ("shot", TIME_DIM),
                np.asarray(TS_TIMES, dtype=np.float32)[None],
            ),
        },
        attrs={"sol_extension": SOL_EXTENSION},
    )


@pytest.mark.parametrize(
    "ip_sign, b0_sign, psi_sign, cocos",
    [(-1, -1, 1, 7), (1, 1, 1, 1), (1, -1, -1, 3)],
    ids=["cmod_normal_field", "cmod_reversed_field", "mast"],
)
def test_imas_export_chain_reads_back(tmp_path, ip_sign, b0_sign, psi_sign, cocos):
    shot = 900000000 + cocos
    unprocessed_ds = synthetic_unprocessed(ip_sign, b0_sign, psi_sign, cocos)
    fit_ds = synthetic_fit(shot)

    ids_list = build_imas_from_shot(
        shot, fit_ds, unprocessed_ds, geqdsk_dir=tmp_path / "geqdsk"
    )
    out_dir = tmp_path / "imas"
    for ids in ids_list:
        write_ids(ids, out_dir)

    def read_back(name: str):
        path = out_dir / f"{name}.nc"
        assert path.exists(), f"{name} IDS file was not written"
        with imas.DBEntry(str(path), "r") as entry:
            return entry.get(name)

    sigma_bp = _sigma_bp(_target_cocos(DD_VERSION))

    eq = read_back("equilibrium")
    assert np.allclose(np.asarray(eq.time), EQ_TIMES)
    ts = eq.time_slice[0]
    dpsi = float(ts.global_quantities.psi_boundary - ts.global_quantities.psi_axis)
    assert np.sign(dpsi) == sigma_bp * ip_sign, "psi direction off target COCOS"
    assert np.sign(float(ts.global_quantities.ip)) == ip_sign
    p1 = ts.profiles_1d
    # sigma_rho_theta_phi is +1 in COCOS 11 and 17
    assert np.all(np.sign(np.asarray(p1.q)) == ip_sign * b0_sign)
    assert np.sign(np.median(np.asarray(p1.phi)[1:])) == b0_sign, "phi off sign(B0)"
    assert np.sign(np.median(np.asarray(p1.f))) == b0_sign
    assert np.sign(np.median(np.asarray(p1.dpressure_dpsi))) == -np.sign(dpsi)
    rho_tor = np.asarray(p1.rho_tor)
    assert np.all(np.isfinite(rho_tor)) and np.all(np.diff(rho_tor) > 0)
    assert np.asarray(eq.time_slice[0].profiles_2d[0].psi).shape == (N_GRID, N_GRID)
    assert ts.boundary.type.value == 0, "circular synthetic plasma is limited"

    cp = read_back("core_profiles")
    assert len(cp.profiles_1d) == len(TS_TIMES)
    assert np.allclose(np.asarray(cp.time), TS_TIMES)
    for i in range(len(TS_TIMES)):
        p = cp.profiles_1d[i]
        te_out = np.asarray(p.electrons.temperature)
        assert te_out.shape == (N_RHO_TOR_NORM,) and np.all(te_out > 0)
        assert np.allclose(te_out, np.asarray(fit_ds["t_e"][0, i]), rtol=1e-6)
        # The psi grid maps back onto the fit grid through the staging map,
        # past the LCFS included
        grid_psi = np.asarray(p.grid.psi)
        psi_axis = float(ts.global_quantities.psi_axis)
        psi_boundary = float(ts.global_quantities.psi_boundary)
        grid_psi_n = (grid_psi - psi_axis) / (psi_boundary - psi_axis)
        grid_rho_tor_norm = rho_tor_norm_from_psi_n(grid_psi_n, QPSI, SOL_EXTENSION)
        fit_rho_tor_norm = fit_ds["rho_tor_norm"].values
        assert np.allclose(grid_rho_tor_norm, fit_rho_tor_norm, atol=1e-6)
        assert (grid_psi_n[fit_rho_tor_norm > 1.0] > 1.0).all()

    sm = read_back("summary")
    assert np.allclose(np.asarray(sm.global_quantities.ip.value), ip_sign * 0.8e6)
    assert np.allclose(np.asarray(sm.global_quantities.b0.value), b0_sign * 5.4)
    assert sm.global_quantities.r0.value == pytest.approx(R0)

    wall = read_back("wall")
    outline = wall.description_2d[0].limiter.unit[0].outline
    assert np.asarray(outline.r).size == N_LIM
    assert np.allclose(
        (np.asarray(outline.r) - R0) ** 2 + np.asarray(outline.z) ** 2, 0.245**2
    )
