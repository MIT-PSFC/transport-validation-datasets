"""Custom disruption-py physics methods for C-Mod."""

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.core.utils.math import interp1
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSetting, TimeSettingParams

from transport_validation_datasets.machine.generic import (
    efit_cocos_from_signs,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    orient_signal,
    snap_to_grid,
)

# Seconds per unit of the units string an MDSplus time node reports.
_TIME_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


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


class CmodGeometryMethods:
    """Geometry signals from the C-Mod aeqdsk that stock disruption-py skips."""

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
        try:
            rout = params.mds_conn.get_data(
                r"\efit_aeqdsk:rout/100", tree_name="_efit_tree"
            )
        except mdsExceptions.MdsException as e:
            params.logger.warning(repr(e))
            params.logger.opt(exception=True).debug(e)
            rout = np.full(len(efit_time), np.nan)

        if not np.array_equal(params.times, efit_time):
            rout = interp1(efit_time, rout, params.times)
        return {"rout": rout}


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

        cocos_input = efit_cocos_from_signs(
            geqdsk_data["current"], geqdsk_data["bcentr"], params.logger
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
    # The core Thomson readout was mid-upgrade during this shot range: both
    # the legacy "yag" electronics/tree and the newer "yag_new" one
    # digitized the same laser pulses (identical timebases, confirmed per
    # shot), but with different channel counts/positions.
    # Taken from C-Mod_Analysis routines
    legacy_core_shot_range = (1030000000, 1040000000)
    legacy_core_nodes = {
        "z": r".yag.results.global.profile:z_sorted",
        "ne": r".yag.results.global.profile:ne_rz_t",
        "ne_error": r".yag.results.global.profile:ne_err_zt",
        "te": r".yag.results.global.profile:te_rz_t",
        "te_error": r".yag.results.global.profile:te_err_zt",
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
        beam radius for every channel. For shots in legacy_core_shot_range,
        the legacy "yag" core channels are merged in alongside "yag_new"
        (same ts_array label "core" for both -- see legacy_core_nodes).
        Data stays on the native TS timebase
        (~20 Hz), not params.times.
        The edge samples are placed on the core timebase sample by sample (_align_region).
        An inconsistent core read raises (see _get_region_channels),
        an edge read failure is logged and the edge skipped.

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

        shot_min, shot_max = CmodThomsonMethods.legacy_core_shot_range
        if shot_min < params.shot_id < shot_max:
            try:
                legacy = CmodThomsonMethods._get_region_channels(
                    params, CmodThomsonMethods.legacy_core_nodes
                )
                legacy["te"] = legacy["te"] * 1000.0
                legacy["te_error"] = legacy["te_error"] * 1000.0
                if not np.allclose(legacy["time"], core["time"], atol=1e-4):
                    raise ValueError(
                        "Legacy core TS timebase does not match yag_new core timebase"
                    )
                core["z"] = np.concatenate([core["z"], legacy["z"]])
                for quant in ["ne", "ne_error", "te", "te_error"]:
                    core[quant] = np.concatenate([core[quant], legacy[quant]], axis=1)
            except Exception as e:
                params.logger.warning(
                    "Legacy core Thomson scattering data not found/merged, "
                    "continuing with yag_new core channels only."
                )
                params.logger.warning(repr(e))
                params.logger.opt(exception=True).debug(e)

        edge = None
        try:
            edge = CmodThomsonMethods._get_region_channels(
                params, CmodThomsonMethods.edge_nodes
            )
        except Exception as e:
            params.logger.warning(
                "Edge Thomson scattering data not found, continuing with core only."
            )
            params.logger.warning(repr(e))
            params.logger.opt(exception=True).debug(e)
        if edge is not None:
            edge = CmodThomsonMethods._align_region(params, edge, core["time"])

        regions = {"core": core} if edge is None else {"core": core, "edge": edge}

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
