"""Custom disruption-py physics methods for C-Mod."""

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.core.utils.math import interp1
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSetting, TimeSettingParams
from disruption_py.settings.time_setting import _postprocess

from transport_validation_datasets.machine.generic import (
    efit_cocos_from_signs,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    orient_signal,
    snap_to_grid,
)


class UniformTimeSetting(TimeSetting):
    """1 kHz uniform timebase up to the maximum time in the EFIT tree."""

    def __init__(self):
        """Initialize with tokamak overrides."""

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """Retrieve the EFIT timebase for the tested tokamaks.

        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns:
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )
        efit_time_unit = params.mds_conn.get_data(
            r"units_of(dim_of(\efit_aeqdsk:ali))", tree_name="_efit_tree"
        )
        if efit_time_unit not in {"s", "ms", "us"}:
            params.logger.verbose(
                "Failed to get the time units of EFIT tree '{tree}', assuming seconds.",
                tree=params.mds_conn.get_tree_name_of_nickname("_efit_tree"),
            )
        efit_times = _postprocess(times=efit_time, units=efit_time_unit)
        max_time = efit_times.max()
        timebase = make_uniform_1kHz_timebase(max_time)
        return timebase


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
        efit_time = params.mds_conn.get_data(
            r"\efit_aeqdsk:time", tree_name="_efit_tree"
        )
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
        efit_time = params.mds_conn.get_data(
            r"\efit_a_eqdsk:atime", tree_name="_efit_tree"
        )
        # EFIT grid arrays are static, collapse any (T, n) layout to a single row.
        r_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\efit_g_eqdsk:rgrid", tree_name="_efit_tree")
        )[0]
        z_grid = np.atleast_2d(
            params.mds_conn.get_data(r"\efit_g_eqdsk:zgrid", tree_name="_efit_tree")
        )[0]

        geqdsk_data = {}
        for param, path in CmodEfitMethods.geqdsk_cols.items():
            try:
                geqdsk_data[param] = params.mds_conn.get_data(
                    path, tree_name="_efit_tree"
                )
            except mdsExceptions.MdsException as e:
                params.logger.warning(repr(e))
                params.logger.opt(exception=True).debug(e)
                geqdsk_data[param] = None

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

        ds_geqdsk = make_geqdsk_dataset(
            shot_id=params.shot_id,
            times=efit_time,
            r_grid=r_grid,
            z_grid=z_grid,
            rmagx=geqdsk_data["rmagx"],
            zmagx=geqdsk_data["zmagx"],
            simagx=geqdsk_data["simagx"],
            sibdry=geqdsk_data["sibdry"],
            bcentr=geqdsk_data["bcentr"],
            current=geqdsk_data["current"],
            fpol=geqdsk_data["fpol"],
            pres=geqdsk_data["pres"],
            ffprime=geqdsk_data["ffprime"],
            pprime=geqdsk_data["pprime"],
            qpsi=geqdsk_data["qpsi"],
            psirz=geqdsk_data["psirz"],
            rbdry=geqdsk_data["rbdry"],
            zbdry=geqdsk_data["zbdry"],
            cocos_input=cocos_input,
            rlim=rlim,
            zlim=zlim,
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

        Args:
            params: disruption-py physics method parameters for the shot.

        Returns:
            Dataset with ne [m^-3] and te [eV] plus errors on (idx, ts_channel),
            channel positions ts_channel_r and ts_channel_z [m] on (ts_channel,).

        Raises:
            ValueError: If the core TS channels are inconsistent (propagated from
                _get_region_channels). Edge TS failures are logged and skipped.
        """
        core = CmodThomsonMethods._get_region_channels(
            params, CmodThomsonMethods.core_nodes
        )
        # Core te is stored in keV, convert to eV to match IMAS and edge system
        core["te"] = core["te"] * 1000.0
        core["te_error"] = core["te_error"] * 1000.0

        edge = None
        try:
            edge = CmodThomsonMethods._get_region_channels(
                params, CmodThomsonMethods.edge_nodes
            )
            if not np.allclose(edge["time"], core["time"], atol=1e-4):
                raise ValueError("Edge TS timebase does not match core TS timebase")
        except Exception as e:
            params.logger.warning(
                "Edge Thomson scattering data not found, continuing with core only."
            )
            params.logger.warning(repr(e))
            params.logger.opt(exception=True).debug(e)
            edge = None

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
