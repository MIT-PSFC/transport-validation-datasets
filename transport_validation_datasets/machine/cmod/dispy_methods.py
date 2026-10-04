"""Custom disruption-py physics methods for C-Mod."""

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSetting, TimeSettingParams

from transport_validation_datasets.machine.generic import (
    EQUILIBRIUM_HOLD_FLOOR,
    cocos_from_signs,
    injected_power_on_grid,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    normalized_beta,
    ohmic_power,
    orient_signal,
    signal_on_grid,
    smoothed_power,
    snap_to_grid,
)

# Seconds per unit of the units string an MDSplus time node reports.
_TIME_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6}

# Length of the TCI chord 4 the line-integrated density nl_04 is divided by [m], as in disruption-py get_densities
TCI_NL_04_CHORD_LENGTH = 0.6

# Factor scaling the AXUV twopi_diode onto the 2pi foil bolometer, as in disruption-py get_power.
# It was cross-calibrated in the flat-top of non-disruptive shots.
TWOPI_DIODE_CALIBRATION = 4.5


def _time_unit_scale(params, expression: str) -> float:
    """Resolve the seconds-per-unit factor of an EFIT tree time expression.

    Args:
        params: Anything carrying the shot's mds_conn and logger.
        expression: MDSplus expression whose units to read.

    Returns:
        Seconds per unit, 1.0 when the tree reports units this does not know.
    """
    unit = params.mds_conn.get_data(f"units_of({expression})", tree_name="_efit_tree")
    scale = _TIME_UNIT_SCALE.get(str(unit).strip().lower())
    if scale is not None:
        return scale
    params.logger.verbose(
        "EFIT tree '{tree}' reports time units '{unit}' for {expr}, assuming seconds.",
        tree=params.mds_conn.get_tree_name_of_nickname("_efit_tree"),
        unit=unit,
        expr=expression,
    )
    return 1.0


def efit_times_in_seconds(params, node: str) -> np.ndarray:
    """Read an EFIT tree time node and convert it to seconds.

    Every C-Mod retrieval here has to land on the same timebase, so they all
    read their EFIT times through this. The tree reports its own units, which
    are seconds in practice but not guaranteed to be.

    Args:
        params: Anything carrying the shot's mds_conn and logger.
        node: EFIT tree time node, e.g. the aeqdsk analysis time.

    Returns:
        The node's times [s].
    """
    times = params.mds_conn.get_data(node, tree_name="_efit_tree")
    return np.asarray(times, dtype=float) * _time_unit_scale(params, node)


class UniformTimeSetting(TimeSetting):
    """1 kHz uniform timebase up to the maximum time in the EFIT tree."""

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """Build the shot's 1 kHz timebase from the EFIT tree's own time range.

        Args:
            params: Parameters needed to retrieve the timebase.

        Returns:
            Times from 0 to the last EFIT time in 1 ms steps [s].
        """
        node = r"\efit_aeqdsk:ali"
        (efit_time,) = params.mds_conn.get_dims(node, tree_name="_efit_tree")
        scale = _time_unit_scale(params, f"dim_of({node})")
        return make_uniform_1kHz_timebase(float(np.max(efit_time)) * scale)


def _aeqdsk_node(params, expression: str, efit_time: np.ndarray) -> np.ndarray:
    """Read one aeqdsk node on the EFIT times, NaN when the tree lacks it.

    Args:
        params: disruption-py physics method parameters for the shot.
        expression: MDSplus expression of the node, with any unit conversion.
        efit_time: (n_eq,) the EFIT times [s], sizing the NaN fallback.

    Returns:
        (n_eq,) the node's values.
    """
    try:
        return params.mds_conn.get_data(expression, tree_name="_efit_tree")
    except mdsExceptions.MdsException as e:
        params.logger.warning(repr(e))
        params.logger.opt(exception=True).debug(e)
        return np.full(len(efit_time), np.nan)


class CmodAeqdskMethods:
    """0D C-Mod aeqdsk signals read as the tree stores them, which stock disruption-py skips or rebuilds."""

    @staticmethod
    @physics_method(columns=["rout"], tokamak=Tokamak.CMOD)
    def get_geometric_major_radius(params: PhysicsMethodParams):
        """Retrieve the geometric major radius of the LCFS.

        disruption-py exposes rmagx (magnetic axis) but not aeqdsk rout, which
        is the boundary geometric center and the radius that pairs with a_minor.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with rout [m] on the requested timebase.
        """
        efit_time = efit_times_in_seconds(params, r"\efit_aeqdsk:time")
        rout = _aeqdsk_node(params, r"\efit_aeqdsk:rout/100", efit_time)
        if not np.array_equal(params.times, efit_time):
            rout = signal_on_grid(efit_time, rout, params.times, EQUILIBRIUM_HOLD_FLOOR)
        return {"rout": rout}

    @staticmethod
    @physics_method(columns=["betan"], tokamak=Tokamak.CMOD)
    def get_normalized_beta(params: PhysicsMethodParams):
        """Retrieve the normalized beta as IMAS defines it, with the vacuum field b0 at r0 (normalized_beta).

        Built from EFIT's own stored energy and volume, wplasm = 3/2 <p> vout.
        bcentr is the vacuum field at rcencm, the fixed 0.66 m the store's r0 is.
        EFIT's betat normalizes with the vacuum field at rout instead,
        and its betan node multiplies by |btaxp|, the total field at the magnetic axis.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with betan [percent m T / MA] on the requested timebase.
        """
        efit_time = efit_times_in_seconds(params, r"\efit_aeqdsk:time")
        energy_mhd = _aeqdsk_node(params, r"\efit_aeqdsk:wplasm", efit_time)  # [J]
        volume = _aeqdsk_node(params, r"\efit_aeqdsk:vout/1e6", efit_time)  # [m^3]
        minor_radius = _aeqdsk_node(params, r"\efit_aeqdsk:aout/100", efit_time)  # [m]
        b_center = _aeqdsk_node(params, r"\efit_aeqdsk:bcentr", efit_time)  # [T]
        ip = _aeqdsk_node(params, r"\efit_aeqdsk:cpasma", efit_time)  # [A]

        with np.errstate(divide="ignore", invalid="ignore"):
            betan = normalized_beta(energy_mhd, volume, minor_radius, b_center, ip)

        if not np.array_equal(params.times, efit_time):
            betan = signal_on_grid(
                efit_time, betan, params.times, EQUILIBRIUM_HOLD_FLOOR
            )
        return {"betan": betan}


def _injected_power(params, node: str, tree_name: str) -> np.ndarray:
    """An injected heating power record on the timebase (injected_power_on_grid), in the record's units.

    0 when the shot has none (that heating system did not run).

    Args:
        params: disruption-py physics method parameters for the shot.
        node: MDSplus node of the power record.
        tree_name: Tree holding it.

    Returns:
        (n_t,) the power on params.times.
    """
    try:
        power, power_time = params.mds_conn.get_data_with_dims(
            node, tree_name=tree_name
        )
    except mdsExceptions.MdsException:
        params.logger.debug("no {node} record, taking 0", node=node)
        return np.zeros(len(params.times))
    return injected_power_on_grid(power_time, power, params.times)


class CmodPlasmaMethods:
    """C-Mod magnetics and density retrievals that replace the disruption-py built-ins.

    The built-ins interpolate onto the timebase.
    These place each record causally (signal_on_grid), so no grid time draws on a later sample.
    Every record here is sampled faster than the grid, so each grid time takes the mean of the preceding millisecond.
    """

    @staticmethod
    @physics_method(columns=["ip"], tokamak=Tokamak.CMOD)
    def get_plasma_current(params: PhysicsMethodParams):
        r"""Plasma current, magnetics \ip, signed.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with ip [A] on the requested timebase.
        """
        ip, ip_time = params.mds_conn.get_data_with_dims(r"\ip", tree_name="magnetics")
        ip_on_grid = signal_on_grid(ip_time, ip, params.times)
        return {"ip": ip_on_grid}

    @staticmethod
    @physics_method(columns=["bt"], tokamak=Tokamak.CMOD)
    def get_toroidal_field(params: PhysicsMethodParams):
        r"""Vacuum toroidal field at 0.66 m, magnetics \btor, signed.

        disruption-py reads it in get_n_equal_1_amplitude, which also needs the BP13 sensors.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with bt [T] on the requested timebase.
        """
        btor, btor_time = params.mds_conn.get_data_with_dims(
            r"\btor", tree_name="magnetics"
        )
        btor_on_grid = signal_on_grid(btor_time, btor, params.times)
        return {"bt": btor_on_grid}

    @staticmethod
    @physics_method(columns=["n_e"], tokamak=Tokamak.CMOD)
    def get_line_average_density(params: PhysicsMethodParams):
        """Line-averaged density, the TCI chord 4 line integral over TCI_NL_04_CHORD_LENGTH.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with n_e [m^-3] on the requested timebase.
        """
        nl_04, nl_04_time = params.mds_conn.get_data_with_dims(
            r".tci.results:nl_04", tree_name="electrons"
        )
        nl_04_trace = np.squeeze(nl_04)
        n_e = nl_04_trace / TCI_NL_04_CHORD_LENGTH
        n_e_on_grid = signal_on_grid(nl_04_time, n_e, params.times)
        return {"n_e": n_e_on_grid}


class CmodPowerMethods:
    """C-Mod power retrievals that replace the disruption-py built-ins.

    Each record is placed on the grid without interpolation (signal_on_grid),
    and power_ohm and power_radiated are then smoothed non-causally (smoothed_power), as on every device.
    """

    @staticmethod
    @physics_method(columns=["p_rad"], tokamak=Tokamak.CMOD)
    def get_radiated_power(params: PhysicsMethodParams):
        r"""Radiated power, the AXUV \twopi_diode in kW scaled by TWOPI_DIODE_CALIBRATION.

        Averaged over each grid step, then smoothed non-causally (smoothed_power), as on every device.
        NaN outside the record, where disruption-py fills 0 and hides a missing record.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with p_rad [W] on the requested timebase.
        """
        diode_kw, diode_time = params.mds_conn.get_data_with_dims(
            r"\twopi_diode", tree_name="spectroscopy"
        )
        p_rad = diode_kw * 1e3 * TWOPI_DIODE_CALIBRATION
        p_rad_on_grid = signal_on_grid(diode_time, p_rad, params.times)
        grid_steps = np.diff(params.times)
        dt = float(np.median(grid_steps))
        p_rad_smoothed = smoothed_power(p_rad_on_grid, dt)
        return {"p_rad": p_rad_smoothed}

    @staticmethod
    @physics_method(columns=["p_icrf", "p_lh"], tokamak=Tokamak.CMOD)
    def get_heating_powers(params: PhysicsMethodParams):
        r"""ICRF net power (\rf_power_net, MW) and lower hybrid net power (LH \top.results:netpow, kW).

        Each is 0 outside its record and on a shot without the system (_injected_power).

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with p_icrf and p_lh [W] on the requested timebase.
        """
        p_icrf_mw = _injected_power(params, r"\rf_power_net", "rf")
        p_lh_kw = _injected_power(params, r"\top.results:netpow", "lh")
        p_icrf = p_icrf_mw * 1e6
        p_lh = p_lh_kw * 1e3
        return {"p_icrf": p_icrf, "p_lh": p_lh}

    @staticmethod
    @physics_method(columns=["p_ohm"], tokamak=Tokamak.CMOD)
    def get_ohmic_power(params: PhysicsMethodParams):
        r"""Compute the ohmic power Ip V_loop - dW_pol/dt (generic.ohmic_power) on the requested timebase.

        V_loop is the flux loop voltage \top.mflux:v0 of the ANALYSIS tree and Ip the magnetics \ip.
        li and the geometric major radius rout come from the EFIT tree.
        Every input is placed causally (signal_on_grid), never interpolated:
        V_loop and Ip are averaged over each grid step,
        and li and R are held from the last reconstruction, for at least EQUILIBRIUM_HOLD_FLOOR.
        The result is smoothed non-causally (smoothed_power), as on every device.
        disruption-py's get_ohmic_parameters subtracts L_i dIp/dt instead of dW_pol/dt,
        which drops the change of li and R that DIII-D EFIT poh and MAST ESM pphix include.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dict with p_ohm [W] on the requested timebase.
        """
        v_loop, v_loop_time = params.mds_conn.get_data_with_dims(
            r"\top.mflux:v0", tree_name="analysis"
        )
        ip, ip_time = params.mds_conn.get_data_with_dims(r"\ip", tree_name="magnetics")
        efit_time = efit_times_in_seconds(params, r"\efit_aeqdsk:time")
        li = params.mds_conn.get_data(r"\efit_aeqdsk:ali", tree_name="_efit_tree")
        major_radius = params.mds_conn.get_data(
            r"\efit_aeqdsk:rout/100", tree_name="_efit_tree"
        )

        times = params.times
        v_loop_on_grid = signal_on_grid(v_loop_time, v_loop, times)
        ip_on_grid = signal_on_grid(ip_time, ip, times)
        li_on_grid = signal_on_grid(efit_time, li, times, EQUILIBRIUM_HOLD_FLOOR)
        major_radius_on_grid = signal_on_grid(
            efit_time, major_radius, times, EQUILIBRIUM_HOLD_FLOOR
        )
        p_ohm_raw = ohmic_power(
            times, ip_on_grid, v_loop_on_grid, li_on_grid, major_radius_on_grid
        )
        grid_steps = np.diff(times)
        dt = float(np.median(grid_steps))
        p_ohm = smoothed_power(p_ohm_raw, dt)
        return {"p_ohm": p_ohm}


class CmodEfitMethods:
    """C-Mod GEQDSK and EFIT-quality retrievals for stock disruption-py."""

    geqdsk_cols = {
        # 0D signals
        "rmagx": r"\efit_g_eqdsk:rmaxis",
        "zmagx": r"\efit_g_eqdsk:zmaxis",
        "simagx": r"\efit_g_eqdsk:ssimag",
        "sibdry": r"\efit_g_eqdsk:ssibry",
        "bcentr": r"\efit_g_eqdsk:bcentr",
        "current": r"\efit_g_eqdsk:cpasma",
        # 1D profiles
        "fpol": r"\efit_g_eqdsk:fpol",
        "pres": r"\efit_g_eqdsk:pres",
        "ffprime": r"\efit_g_eqdsk:ffprim",
        "pprime": r"\efit_g_eqdsk:pprime",
        "qpsi": r"\efit_g_eqdsk:qpsi",
        # 2D flux grid
        "psirz": r"\efit_g_eqdsk:psirz",
        # Boundary
        "rbdry": r"\efit_g_eqdsk:rbbbs",
        "zbdry": r"\efit_g_eqdsk:zbbbs",
    }

    @staticmethod
    @physics_method(columns=[*geqdsk_cols.keys()], tokamak=Tokamak.CMOD)
    def get_geqdsk_parameters(params: PhysicsMethodParams):
        """Retrieve the full GEQDSK reconstruction for C-Mod (COCOS-normalised).

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dataset with all geqdsk_cols signals plus grids and limiter, snapped
            onto the requested timebase without interpolation.
        """
        efit_time = efit_times_in_seconds(params, r"\efit_a_eqdsk:atime")
        # EFIT grid arrays are static, collapse any (T, n) layout to a single row.
        r_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\efit_g_eqdsk:rgrid", tree_name="_efit_tree")
        )[0]
        z_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\efit_g_eqdsk:zgrid", tree_name="_efit_tree")
        )[0]
        # RCENTR is the radius bcentr is quoted at (~0.66 m on C-Mod), stored in the tree as RZERO.
        # It is a machine constant, not the midpoint of the R grid.
        # Static like the grid arrays, so collapse it to a scalar.
        rcentr = float(
            np.atleast_1d(
                params.mds_conn.get_data(r"\efit_g_eqdsk:rzero", tree_name="_efit_tree")
            )[0]
        )

        # Every one of these is needed to build a GEQDSK.
        # A missing node raises here, and disruption-py fills this method's columns with NaN.
        geqdsk_data = {
            param: params.mds_conn.get_data(path, tree_name="_efit_tree")
            for param, path in CmodEfitMethods.geqdsk_cols.items()
        }

        rlim = zlim = None
        try:
            rlim = params.mds_conn.get_data(
                r"\efit_g_eqdsk:xlim", tree_name="_efit_tree"
            )
            zlim = params.mds_conn.get_data(
                r"\efit_g_eqdsk:ylim", tree_name="_efit_tree"
            )
        except mdsExceptions.MdsException as e:
            params.logger.warning(repr(e))
            params.logger.opt(exception=True).debug(e)

        geqdsk_data = orient_signal(geqdsk_data, efit_time)

        cocos_input = cocos_from_signs(
            geqdsk_data["current"],
            geqdsk_data["bcentr"],
            geqdsk_data["simagx"],
            geqdsk_data["sibdry"],
            geqdsk_data["qpsi"],
            params.logger,
        )

        # geqdsk_cols keys are the make_geqdsk_dataset argument names
        ds_geqdsk = make_geqdsk_dataset(
            shot_id=params.shot_id,
            times=efit_time,
            r_grid=r_grid,
            z_grid=z_grid,
            cocos_input=cocos_input,
            rcentr=rcentr,
            rlim=rlim,
            zlim=zlim,
            **geqdsk_data,
        )

        # Snap onto requested timebase without interpolation
        ds_geqdsk = snap_to_grid(ds_geqdsk, params.times)
        return ds_geqdsk


class CmodThomsonMethods:
    """Raw Thomson scattering channel retrievals for C-Mod."""

    core_nodes = {
        "z": r".yag_new.results.profiles:z_sorted",
        "ne": r".yag_new.results.profiles:ne_rz",
        "ne_error": r".yag_new.results.profiles:ne_err",
        "te": r".yag_new.results.profiles:te_rz",
        "te_error": r".yag_new.results.profiles:te_err",
    }
    edge_nodes = {
        "z": r"\fiber_z",
        "ne": r"\ts_ne",
        "ne_error": r"\ts_ne_err",
        "te": r"\ts_te",
        "te_error": r"\ts_te_err",
    }

    # Per-variable attributes, IMAS data dictionary path under "ref"
    channel_attrs = {
        "ts_channel_r": {
            "description": "Major radius of TS channel measurement locations",
            "units": "m",
            "ref": "/thomson_scattering/channel(i1)/position/r",
        },
        "ts_channel_z": {
            "description": "Height of TS channel measurement locations",
            "units": "m",
            "ref": "/thomson_scattering/channel(i1)/position/z",
        },
        "ts_channel_ne": {
            "description": "Electron density measured by TS channels",
            "units": "m^-3",
            "ref": "/thomson_scattering/channel(i1)/n_e/data",
        },
        "ts_channel_ne_error": {
            "description": "Electron density measurement error of TS channels",
            "units": "m^-3",
            "ref": "/thomson_scattering/channel(i1)/n_e/data_error_upper",
        },
        "ts_channel_te": {
            "description": "Electron temperature measured by TS channels",
            "units": "eV",
            "ref": "/thomson_scattering/channel(i1)/t_e/data",
        },
        "ts_channel_te_error": {
            "description": "Electron temperature measurement error of TS channels",
            "units": "eV",
            "ref": "/thomson_scattering/channel(i1)/t_e/data_error_upper",
        },
    }

    @staticmethod
    def _get_region_channels(params: PhysicsMethodParams, nodes: dict) -> dict:
        """Retrieve one TS system: channel z positions, ne/te profiles and errors.

        Returns:
            Dict with z (channel,), time (T,), and ne/ne_error/te/te_error as
            (T, channel) arrays with invalid (zero) points set to nan.

        Raises:
            ValueError: If the channel count does not match the z positions, or
                the te and ne timebases disagree.
        """
        region = {"z": params.mds_conn.get_data(nodes["z"], tree_name="electrons")}
        time = None
        for quant in ["ne", "te"]:
            data, quant_time = params.mds_conn.get_data_with_dims(
                nodes[quant], tree_name="electrons"
            )
            error = params.mds_conn.get_data(
                nodes[f"{quant}_error"], tree_name="electrons"
            )
            # Stored as (channel, time), cut pre-fire times and orient to (time, channel)
            valid_time_indices = quant_time >= 0
            quant_time = quant_time[valid_time_indices]
            data = data[:, valid_time_indices].astype(np.float64).T
            error = error[:, valid_time_indices].astype(np.float64).T
            # 0 marks channels with no measurement
            data[data == 0] = np.nan
            error[error == 0] = np.nan

            if data.shape[1] != len(region["z"]):
                raise ValueError(
                    f"TS {quant} channel count {data.shape[1]} does not match "
                    f"z position count {len(region['z'])}"
                )
            if time is None:
                time = quant_time
            elif not np.array_equal(quant_time, time):
                raise ValueError("TS te timebase does not match ne timebase")

            region[quant] = data
            region[f"{quant}_error"] = error
        region["time"] = time
        return region

    @staticmethod
    def _align_region(
        params: PhysicsMethodParams, region: dict, time: np.ndarray
    ) -> dict:
        """Place one TS system's samples onto another system's timebase.

        Both the edge and the core read the same laser pulses, but their clocks differ by up to ~20 us.
        A system can also skip a pulse, as the edge does late in shot 1160909025.
        The per-system time >= 0 cut can split the pulse at t ~ 0 between them, as in shot 1160913007.
        Each sample goes to the nearest time of the other system (snap_to_grid),
        so a slip costs only the samples it touches, never the whole system.
        Times with no sample come back NaN, and samples with no partner time are dropped.

        Args:
            params: disruption-py physics method parameters for the shot.
            region: One system as _get_region_channels returns it.
            time: Timebase to place it on [s].

        Returns:
            The region on time, with the same keys.
        """
        quants = ["ne", "ne_error", "te", "te_error"]
        shot_ids = np.repeat(params.shot_id, region["time"].size)
        ds_region = xr.Dataset(
            {quant: (("idx", "ts_channel"), region[quant]) for quant in quants},
            coords={"time": ("idx", region["time"]), "shot": ("idx", shot_ids)},
        )
        ds_aligned = snap_to_grid(ds_region, time)

        region_aligned = {"z": region["z"], "time": time}
        for quant in quants:
            region_aligned[quant] = ds_aligned[quant].values
        mask_data_before = np.isfinite(region["ne"]) | np.isfinite(region["te"])
        mask_data_after = np.isfinite(region_aligned["ne"])
        mask_data_after |= np.isfinite(region_aligned["te"])
        n_samples_before = int(mask_data_before.any(axis=1).sum())
        n_samples_after = int(mask_data_after.any(axis=1).sum())
        n_lost = n_samples_before - n_samples_after
        if n_lost > 0:
            params.logger.warning(
                f"{n_lost} TS samples with data had no partner time and were dropped."
            )
        return region_aligned

    @staticmethod
    @physics_method(
        columns=[
            "ts_channel_r",
            "ts_channel_z",
            "ts_channel_ne",
            "ts_channel_ne_error",
            "ts_channel_te",
            "ts_channel_te_error",
        ],
        tokamak=Tokamak.CMOD,
    )
    def get_thomson_channels(params: PhysicsMethodParams):
        """Get TS measurements for core and edge systems at their R and Z locations.

        Both systems measure along the same vertical laser chord, so R is the
        beam radius for every channel. Data stays on the native TS timebase
        (~20 Hz), not params.times.
        The edge samples are placed on the core timebase sample by sample (_align_region).
        Both systems are required, and a failed or inconsistent read of either raises.

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dataset with ne [m^-3] and te [eV] plus errors on (idx, ts_channel),
            channel positions ts_channel_r and ts_channel_z [m] on (ts_channel,).
        """
        core = CmodThomsonMethods._get_region_channels(
            params, CmodThomsonMethods.core_nodes
        )
        # Core te is stored in keV, convert to eV to match IMAS and edge system
        core["te"] = core["te"] * 1000.0
        core["te_error"] = core["te_error"] * 1000.0

        # Without the edge the data ends at rho_tor_norm ~0.8
        edge = CmodThomsonMethods._get_region_channels(
            params, CmodThomsonMethods.edge_nodes
        )
        edge = CmodThomsonMethods._align_region(params, edge, core["time"])

        regions = {"core": core, "edge": edge}

        # Radius of the vertical laser beam, shared by all channels
        r_beam = float(
            np.atleast_1d(
                params.mds_conn.get_data(r".yag.results.param:r", tree_name="electrons")
            )[0]
        )

        z_pos = np.concatenate([r["z"] for r in regions.values()])
        n_channels = len(z_pos)
        ts_array = np.concatenate(
            [np.repeat(name, len(r["z"])) for name, r in regions.items()]
        )

        data_vars = {
            "ts_channel_r": (
                "ts_channel",
                np.full(n_channels, r_beam, dtype=np.float32),
            ),
            "ts_channel_z": ("ts_channel", z_pos.astype(np.float32)),
        }
        for quant in ["ne", "ne_error", "te", "te_error"]:
            combined = np.concatenate([r[quant] for r in regions.values()], axis=1)
            data_vars[f"ts_channel_{quant}"] = (
                ("idx", "ts_channel"),
                combined.astype(np.float32),
            )

        ds_thomson = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": ("idx", core["time"]),
                "shot": ("idx", np.repeat(params.shot_id, len(core["time"]))),
                "ts_channel": np.arange(n_channels),
                "ts_array": ("ts_channel", ts_array),
            },
            attrs={
                "description": "Raw Thomson scattering measurement channels from core and edge systems at their R and Z locations",
            },
        )
        for var, attrs in CmodThomsonMethods.channel_attrs.items():
            ds_thomson[var].attrs.update(attrs)

        return ds_thomson
