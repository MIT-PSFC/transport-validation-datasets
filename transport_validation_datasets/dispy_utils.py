from functools import partialmethod

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import resolve_tokamak_from_environment
from disruption_py.settings import LogSettings, TimeSetting, TimeSettingParams
from disruption_py.workflow import get_database
from loguru import logger

from transport_validation_datasets.machine.generic import (
    cocos_from_signs,
    injected_power_on_grid,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
)

VERBOSE_LEVEL_NO = 15


def register_verbose_level():
    """Add loguru's custom VERBOSE level and bind logger.verbose().

    disruption_py does this inside LogSettings.setup_logging(), which
    passive_log_settings() deliberately skips. Its own modules still call
    logger.verbose(), so the level and the bound method have to exist anyway.
    Idempotent: safe to call once per process or many times.
    """
    try:
        logger.level("VERBOSE", color="<dim>")
    except ValueError:
        logger.level("VERBOSE", color="<dim>", no=VERBOSE_LEVEL_NO)
    if not hasattr(logger.__class__, "verbose"):
        logger.__class__.verbose = partialmethod(logger.__class__.log, "VERBOSE")


def passive_log_settings() -> LogSettings:
    """LogSettings that leave this process's loguru sinks alone.

    disruption_py's setup_logging() and reset_handlers() call logger.remove(),
    which wipes every sink on the global loguru logger, the workflow's run log file among them.
    Pre-setting the setup flag and a console level skips both reset paths in get_shots_data,
    so its messages flow through whatever sinks the caller already configured.

    Returns:
        LogSettings that leave the caller's loguru sinks in place.
    """
    register_verbose_level()
    return LogSettings(
        file_path=None, console_level="VERBOSE", _logging_has_been_setup=True
    )


def summary(
    summary_table: str,
    ipmax: float,
    pulse_length: float,
    min_shot: int,
    max_shot: int,
    shots: list[int] | bool = False,
) -> np.ndarray:
    """Find shots with high enough current and long enough pulse length.

    Performs a SELECT query on the `summary` table. Optionally selects shots from a
    given list. Snagged from
    https://github.com/MIT-PSFC/disruption-efit/blob/main/disruption_efit/sql.py

    Args:
        summary_table: Name of the summary table to query.
        ipmax: Threshold that maximum plasma current must exceed [A].
        pulse_length: Threshold that pulse length must exceed [s].
        min_shot: Disregard shots below this number. Ignored if not positive.
        max_shot: Disregard shots above this number. Ignored if not positive.
        shots: List of shots to be queried, or False to query all of them.

    Returns:
        Nx2 array of [shot_id, pulse_length] for the N shots which exceed the
        thresholds.
    """
    # database
    db = get_database(tokamak=resolve_tokamak_from_environment())

    # query
    query = [
        f"select distinct(shot), pulse_length from {summary_table} ",
        f"where ipmax > {ipmax} and pulse_length > {pulse_length}",
    ]
    if min_shot > 0:
        query += [f"and shot >= {min_shot}"]
    if max_shot > 0:
        query += [f"and shot <= {max_shot}"]
    if hasattr(shots, "__iter__"):
        query += [f"and shot in ({', '.join(str(s) for s in shots)})"]
    query += ["order by shot"]
    logger.trace("> {query}", query=" ".join(query))

    # results
    data = db.query(" ".join(query), use_pandas=True).values

    logger.trace("= {shape}", shape=data.shape)
    return data


def empty_result(result: xr.Dataset) -> bool:
    """Whether get_shots_data returned no usable data for a shot.

    When a retrieval fails (a missing MDSplus tree, say), get_shots_data logs the error
    and returns an empty dataset without the shot and time index variables,
    which set_index would choke on.

    Args:
        result: What get_shots_data returned.

    Returns:
        True when the result has no shot or time data.
    """
    return "shot" not in result or "time" not in result or result["time"].size == 0


def injected_power(
    params: PhysicsMethodParams, node: str, tree_name: str, time_scale: float = 1.0
) -> np.ndarray:
    """An injected heating power record on the timebase (injected_power_on_grid), in the record's units.

    0 when the shot has no record, that heating system did not run.
    A stray t = 0 sample closing the record (DIII-D echpwrc) is dropped.

    Args:
        params: disruption-py physics method parameters for the shot.
        node: MDSplus node of the power record.
        tree_name: Tree holding it.
        time_scale: Seconds per unit of the record's time axis.

    Returns:
        (n_t,) the power on params.times.
    """
    try:
        power, power_time_raw = params.mds_conn.get_data_with_dims(
            node, tree_name=tree_name
        )
    except mdsExceptions.MdsException:
        params.logger.debug("no {node} record, taking 0", node=node)
        return np.zeros(len(params.times))
    if power_time_raw.size > 1 and power_time_raw[-1] == 0:
        power_time_raw = power_time_raw[:-1]
        power = power[:-1]
    return injected_power_on_grid(power_time_raw * time_scale, power, params.times)


class UniformTimebaseSetting(TimeSetting):
    """The shot's uniform 1 kHz timebase from 0 to the end of its EFIT (make_uniform_1kHz_timebase).

    A device subclass reads the EFIT end time from its own tree (efit_end_time).
    """

    def efit_end_time(self, params: TimeSettingParams) -> float:
        """The last EFIT time of the shot [s].

        Args:
            params: Parameters needed to retrieve the timebase.

        Returns:
            The end time [s].
        """
        raise NotImplementedError

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        return make_uniform_1kHz_timebase(self.efit_end_time(params))


def assemble_geqdsk_dataset(
    shot_id: int,
    efit_time: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    rcentr: float,
    geqdsk_data: dict[str, np.ndarray],
    rlim: np.ndarray | None,
    zlim: np.ndarray | None,
    logger_override,
) -> xr.Dataset:
    """The GEQDSK block of one EFIT tree from its time-first node arrays, with its COCOS identified.

    Args:
        shot_id: Shot number.
        efit_time: (n_t,) reconstruction times [s].
        r_grid: (n_r,) grid major radii [m].
        z_grid: (n_z,) grid heights [m].
        rcentr: The radius bcentr is quoted at [m].
        geqdsk_data: The make_geqdsk_dataset signal arguments by name, time first (orient_signal).
        rlim: (n_lim,) limiter major radii [m], or None.
        zlim: (n_lim,) limiter heights [m], or None.
        logger_override: Logger for cocos_from_signs.

    Returns:
        The block on dim "idx" with "time" and "shot" coords and the "cocos" attribute.
    """
    cocos_input = cocos_from_signs(
        geqdsk_data["current"],
        geqdsk_data["bcentr"],
        geqdsk_data["simagx"],
        geqdsk_data["sibdry"],
        geqdsk_data["qpsi"],
        logger_override,
    )
    return make_geqdsk_dataset(
        shot_id=shot_id,
        times=efit_time,
        r_grid=r_grid,
        z_grid=z_grid,
        cocos_input=cocos_input,
        rcentr=rcentr,
        rlim=rlim,
        zlim=zlim,
        **geqdsk_data,
    )
