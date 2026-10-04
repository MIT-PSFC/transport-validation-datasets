"""Custom disruption-py physics methods, EFIT tree and timebase for the DIII-D dataset.

Written against the PyPI disruption-py 0.14.0 API (params.mds_conn) and passed in through RetrievalSettings,
so nothing in disruption-py is patched.
Every EFIT signal comes from the shot's DISPY run, the 1 kHz disruption-efit (DispyEfitNicknameSetting).
"""

import numpy as np
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSettingParams
from disruption_py.settings.nickname_setting import (
    NicknameSetting,
    NicknameSettingParams,
)
from loguru import logger

from transport_validation_datasets.dispy_utils import (
    UniformTimebaseSetting,
    assemble_geqdsk_dataset,
    injected_power,
)
from transport_validation_datasets.machine.generic import (
    EQUILIBRIUM_HOLD_FLOOR,
    MU0,
    normalized_beta,
    orient_signal,
    signal_on_grid,
    snap_to_grid,
)

# Major radius EFIT gives the vacuum toroidal field at (its rzero), the store's r0 [m]
R0 = 1.6955
# b0 = mu0 N I / (2 pi R0) of the N = 144 turn TF coil from its current bcoil [A], the EFIT bcentr formula.
# PTDATA bt is not used, it reads 2.4-2.8 percent above this at a reference radius nowhere documented.
TF_COIL_TURNS = 144
BCOIL_TO_B0 = MU0 * TF_COIL_TURNS / (2 * np.pi * R0)

# A shot whose EFIT median step exceeds this is not a 1 kHz reconstruction and is skipped [ms]
EFIT_MAX_STEP_MS = 1.5
# EFIT slices with a chi-squared above this are invalid, the cut of disruption-py's get_efit_parameters
EFIT_CHISQ_MAX = 50.0

# GEQDSK block name -> g-file node of the EFIT tree, each (n_t, ...) on the a-file times
GEQDSK_NODES = {
    "rmagx": r"\top.results.geqdsk:rmaxis",
    "zmagx": r"\top.results.geqdsk:zmaxis",
    "simagx": r"\top.results.geqdsk:ssimag",
    "sibdry": r"\top.results.geqdsk:ssibry",
    "bcentr": r"\top.results.geqdsk:bcentr",
    "current": r"\top.results.geqdsk:cpasma",
    "fpol": r"\top.results.geqdsk:fpol",
    "pres": r"\top.results.geqdsk:pres",
    "ffprime": r"\top.results.geqdsk:ffprim",
    "pprime": r"\top.results.geqdsk:pprime",
    "qpsi": r"\top.results.geqdsk:qpsi",
    "psirz": r"\top.results.geqdsk:psirz",
    "rbdry": r"\top.results.geqdsk:rbbbs",
    "zbdry": r"\top.results.geqdsk:zbbbs",
}


class DispyEfitNicknameSetting(NicknameSetting):
    """The shot's _efit_tree: its latest run under a code_rundb runtag (DISPY, the 1 kHz disruption-efit).

    disruption-py's own DIII-D nickname falls back to the 50 Hz efit01 when the runtag has no run,
    and forces runtag DIS under pytest. This raises instead, so no shot is built on another EFIT.
    """

    def __init__(self, runtag: str):
        """Select the runs of one runtag.

        Args:
            runtag: code_rundb runtag of the EFIT runs (D3DSettings.runtag).
        """
        self.runtag = runtag

    def _get_tree_name(self, params: NicknameSettingParams) -> str:
        efit_runs = params.database.query(
            f"select tree from code_rundb.dbo.plasmas where shot = {params.shot_id} "
            f"and runtag = '{self.runtag}' and deleted = 0 order by idx",
            use_pandas=False,
        )
        if not efit_runs:
            raise ValueError(
                f"Shot {params.shot_id} has no EFIT run under runtag {self.runtag}"
            )
        efit_tree = efit_runs[-1][0]
        logger.info(
            f"Shot {params.shot_id}: EFIT tree {efit_tree} (runtag {self.runtag})"
        )
        return efit_tree


class Uniform1kHzTimeSetting(UniformTimebaseSetting):
    """Uniform 1 kHz timebase [s] from 0 to the end of the EFIT.

    Raises when the EFIT is slower than 1 kHz or a single slice, the marks of a reconstruction that is not DISPY.
    """

    def efit_end_time(self, params: TimeSettingParams) -> float:
        """The last a-file time of the DISPY EFIT [s].

        Args:
            params: Parameters needed to retrieve the timebase.

        Returns:
            The end time [s].

        Raises:
            ValueError: If the EFIT has a single slice or a median step above EFIT_MAX_STEP_MS.
        """
        efit_time_ms = np.atleast_1d(
            params.mds_conn.get_data(r"\efit_a_eqdsk:atime", tree_name="_efit_tree")
        )
        if efit_time_ms.size < 2:
            raise ValueError(
                f"Shot {params.shot_id}: {efit_time_ms.size} EFIT slice, not a 1 kHz reconstruction"
            )
        efit_steps_ms = np.diff(efit_time_ms)
        efit_step_median_ms = np.median(efit_steps_ms)
        if efit_step_median_ms > EFIT_MAX_STEP_MS:
            raise ValueError(
                f"Shot {params.shot_id}: median EFIT step {efit_step_median_ms:.1f} ms, not a 1 kHz reconstruction"
            )
        return float(np.max(efit_time_ms)) / 1e3


def _efit_slices(params: PhysicsMethodParams) -> tuple[np.ndarray, np.ndarray]:
    """EFIT slice times and the mask of the valid slices, those with a chi-squared within EFIT_CHISQ_MAX.

    Args:
        params: disruption-py physics method parameters for the shot.

    Returns:
        (efit_time, mask_valid): the (n_eq,) slice times [s] and the (n_eq,) mask of the valid ones.
    """
    efit_time_ms = params.mds_conn.get_data(
        r"\efit_a_eqdsk:atime", tree_name="_efit_tree"
    )
    efit_time = efit_time_ms / 1e3
    chisq = params.mds_conn.get_data(r"\efit_a_eqdsk:chisq", tree_name="_efit_tree")
    mask_valid = chisq <= EFIT_CHISQ_MAX
    return efit_time, mask_valid


def _efit_signals(
    params: PhysicsMethodParams, nodes: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A-eqdsk nodes on the EFIT timebase, NaN on the invalid slices (_efit_slices).

    Reads DIII-D's own a-eqdsk node names only. The a-file-format names in the same tree are NID
    aliases, and about half of them point at a different quantity.

    Args:
        params: disruption-py physics method parameters for the shot.
        nodes: A-eqdsk node names.

    Returns:
        (efit_time, signals): the (n_eq,) slice times [s] and each node's (n_eq,) values.
    """
    efit_time, mask_valid = _efit_slices(params)
    signals = {}
    for node in nodes:
        values = params.mds_conn.get_data(
            rf"\efit_a_eqdsk:{node}", tree_name="_efit_tree"
        )
        values[~mask_valid] = np.nan
        signals[node] = values
    return efit_time, signals


class D3DMethods:
    """Signals the dataset needs that disruption-py 0.14 has no built-in for, or no usable one.

    Every 0D signal is placed on the timebase causally (signal_on_grid), never interpolated,
    so no grid time draws on a later sample.
    The disruption-py built-ins interpolate, so the EFIT scalars and the plasma current are read here too.
    The GEQDSK block is snapped onto the timebase like C-Mod's (get_geqdsk_parameters).
    """

    # Store geometry
    BOUNDARY_NODES = ["aminor", "rsurf", "kappa", "tritop", "tribot"]
    # A-eqdsk nodes the normalized beta is built from (normalized_beta): stored energy [J], volume [m^3],
    # minor radius [m], the vacuum field bcentr at rcentr = R0 [T] and the reconstructed current [A]
    NORMALIZED_BETA_NODES = ["wmhd", "volume", "aminor", "bcentr", "ipmhd"]

    @staticmethod
    @physics_method(columns=["ip"], tokamak=Tokamak.D3D)
    def get_plasma_current(params: PhysicsMethodParams):
        """Measured plasma current [A] (PTDATA ip), signed.

        The node of the built-in get_ip_parameters, which interpolates it.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"ip": (n_t,)} on the timebase.
        """
        ip, ip_time_ms = params.mds_conn.get_data_with_dims(
            f"ptdata('ip', {params.shot_id})"
        )
        ip_time = ip_time_ms / 1e3
        ip_on_timebase = signal_on_grid(ip_time, ip, params.times)
        return {"ip": ip_on_timebase}

    @staticmethod
    @physics_method(columns=["b0"], tokamak=Tokamak.D3D)
    def get_toroidal_field(params: PhysicsMethodParams):
        """Vacuum toroidal field at R0 [T], mu0 144 bcoil / (2 pi R0) from the TF coil current (PTDATA bcoil).

        That is how EFIT computes bcentr, and the two agree to 0.01 percent, but bcoil has no EFIT dropouts.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"b0": (n_t,)} on the timebase.
        """
        bcoil, bcoil_time_ms = params.mds_conn.get_data_with_dims(
            f"ptdata('bcoil', {params.shot_id})"
        )
        bcoil_time = bcoil_time_ms / 1e3
        bcoil_on_timebase = signal_on_grid(bcoil_time, bcoil, params.times)
        return {"b0": bcoil_on_timebase * BCOIL_TO_B0}

    @staticmethod
    @physics_method(columns=["wmhd", "beta_tor_norm"], tokamak=Tokamak.D3D)
    def get_efit_scalars(params: PhysicsMethodParams):
        """Stored energy and normalized beta of the DISPY EFIT, held over the invalid slices (_efit_slices).

        wmhd is the node of the built-in get_efit_parameters, which interpolates it.
        beta_tor_norm is built as IMAS defines it from the reconstruction's own energy and volume (normalized_beta),
        not read from the tree betan, which normalizes with the vacuum field at rout
        and which disruption-py's built-in serves under the column name beta_n.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"wmhd", "beta_tor_norm": (n_t,)} on the timebase.
        """
        efit_time, signals = _efit_signals(params, D3DMethods.NORMALIZED_BETA_NODES)
        energy_mhd = signals["wmhd"]
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_tor_norm = normalized_beta(
                energy_mhd,
                signals["volume"],
                signals["aminor"],
                signals["bcentr"],
                signals["ipmhd"],
            )
        energy_mhd_on_grid = signal_on_grid(
            efit_time, energy_mhd, params.times, EQUILIBRIUM_HOLD_FLOOR
        )
        beta_tor_norm_on_grid = signal_on_grid(
            efit_time, beta_tor_norm, params.times, EQUILIBRIUM_HOLD_FLOOR
        )
        return {"wmhd": energy_mhd_on_grid, "beta_tor_norm": beta_tor_norm_on_grid}

    @staticmethod
    @physics_method(columns=["n_e_line_average"], tokamak=Tokamak.D3D)
    def get_line_average_density(params: PhysicsMethodParams):
        r"""Line-averaged electron density [m^-3], \density of the DISPY EFIT tree, else the PCS estimate dssdenest.

        Replaces the built-in get_density_parameters, which falls back to \d3d::denv2,
        and that reads about 3x higher. dssdenest [1e19 m^-3] matches \density within 1 percent where both exist.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"n_e_line_average": (n_t,)} on the timebase.
        """
        try:
            density_cm3, density_time_ms = params.mds_conn.get_data_with_dims(
                r"\density", tree_name="_efit_tree"
            )
            density = density_cm3 * 1e6
        except mdsExceptions.MdsException:
            density = np.array([np.nan])
        if not np.isfinite(density).any():
            params.logger.warning("EFIT tree has no density, using PCS dssdenest")
            density_1e19, density_time_ms = params.mds_conn.get_data_with_dims(
                f"ptdata('dssdenest', {params.shot_id})"
            )
            density = density_1e19 * 1e19
        density_time = density_time_ms / 1e3
        n_e_line_average = signal_on_grid(density_time, density, params.times)
        return {"n_e_line_average": n_e_line_average}

    @staticmethod
    @physics_method(columns=["p_rad"], tokamak=Tokamak.D3D)
    def get_radiated_power(params: PhysicsMethodParams):
        r"""Total radiated power [W] including the divertor, \bolom::prad_tot of the standard bolometer analysis.

        Sampled every 4 ms, from raw channels smoothed by a centered 50 ms boxcar applied twice (non-causal),
        the kernel smoothed_power gives the other devices, so it is not smoothed further.
        Its units label reads MW, but the values are W (they match the built-in pwrmix to a few percent).
        Replaces the built-in pwrmix,
        a causal 10 ms sum of the 48 raw channels that resolves ELMs and goes negative.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"p_rad": (n_t,)} on the timebase.
        """
        p_rad, p_rad_time_ms = params.mds_conn.get_data_with_dims(
            r"\top.prad_01.prad:prad_tot", tree_name="bolom"
        )
        p_rad_time = p_rad_time_ms / 1e3
        p_rad_on_timebase = signal_on_grid(p_rad_time, p_rad, params.times)
        return {"p_rad": p_rad_on_timebase}

    @staticmethod
    @physics_method(columns=["p_nbi", "p_ech"], tokamak=Tokamak.D3D)
    def get_heating_powers(params: PhysicsMethodParams):
        """Injected neutral beam and electron cyclotron powers [W], the nodes of the built-in get_power_parameters.

        The built-in also rebuilds pwrmix from the 48 raw bolometer channels and P_oh from vloopb,
        so a bolometer failure there would NaN p_nbi and skip the shot.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {"p_nbi", "p_ech": (n_t,)} on the timebase.
        """
        p_nbi_kW = injected_power(params, r"\top.nb:pinj", "d3d", time_scale=1e-3)
        p_nbi = p_nbi_kW * 1e3
        p_ech = injected_power(params, r"\top.ech.total:echpwrc", "rf", time_scale=1e-3)
        return {"p_nbi": p_nbi, "p_ech": p_ech}

    @staticmethod
    @physics_method(columns=BOUNDARY_NODES, tokamak=Tokamak.D3D)
    def get_boundary_parameters(params: PhysicsMethodParams):
        """Plasma boundary minor radius, geometric axis R, elongation and triangularities from the DISPY EFIT.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            {BOUNDARY_NODES: (n_t,)} on the timebase.
        """
        efit_time, signals = _efit_signals(params, D3DMethods.BOUNDARY_NODES)
        return {
            node: signal_on_grid(
                efit_time, values, params.times, EQUILIBRIUM_HOLD_FLOOR
            )
            for node, values in signals.items()
        }

    @staticmethod
    @physics_method(columns=[*GEQDSK_NODES.keys()], tokamak=Tokamak.D3D)
    def get_geqdsk_parameters(params: PhysicsMethodParams):
        """The full GEQDSK reconstruction of the DISPY EFIT, snapped onto the timebase without interpolation.

        The g-file nodes share the a-file times (ms in the tree).
        A slice failing EFIT_CHISQ_MAX is NaN, so usable_reconstructions drops it.
        rzero is stored per slice but is the machine constant R0, so it is taken as one scalar.
        The limiter is one (n, 2) array of R and Z.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dataset with every GEQDSK_NODES signal plus grids and limiter,
            the COCOS number as its "cocos" attribute.
        """
        efit_time, mask_valid = _efit_slices(params)
        r_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\top.results.geqdsk:r", tree_name="_efit_tree")
        )[0]
        z_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\top.results.geqdsk:z", tree_name="_efit_tree")
        )[0]
        rcentr_slices = params.mds_conn.get_data(
            r"\top.results.geqdsk:rzero", tree_name="_efit_tree"
        )
        rcentr = float(np.nanmedian(rcentr_slices))
        limiter = np.asarray(
            params.mds_conn.get_data(
                r"\top.results.geqdsk:lim", tree_name="_efit_tree"
            ),
            dtype=float,
        )
        # (n, 2) rows of R and Z, whichever way the tree lays them out
        if limiter.ndim == 2 and limiter.shape[0] == 2 and limiter.shape[1] != 2:
            limiter = limiter.T
        limiter_rows = limiter.reshape(-1, 2)

        # Every one of these is needed to build a GEQDSK.
        # A missing node raises here, and disruption-py fills this method's columns with NaN.
        geqdsk_data = {
            name: np.asarray(
                params.mds_conn.get_data(node, tree_name="_efit_tree"), dtype=float
            )
            for name, node in GEQDSK_NODES.items()
        }
        geqdsk_data = orient_signal(geqdsk_data, efit_time)
        for values in geqdsk_data.values():
            values[~mask_valid] = np.nan

        geqdsk_data["psirz"] = geqdsk_data["psirz"].astype(np.float32)
        ds_geqdsk = assemble_geqdsk_dataset(
            params.shot_id,
            efit_time,
            r_grid,
            z_grid,
            rcentr,
            geqdsk_data,
            limiter_rows[:, 0],
            limiter_rows[:, 1],
            params.logger,
        )
        return snap_to_grid(ds_geqdsk, params.times)
