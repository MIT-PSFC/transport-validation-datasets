from functools import partialmethod

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak, resolve_tokamak_from_environment
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
    TimeSetting,
    TimeSettingParams,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_database, get_shots_data
from loguru import logger

from transport_validation_datasets.machine.generic import (
    cocos_from_signs,
    injected_power_on_grid,
    make_geqdsk_dataset,
    make_uniform_1kHz_timebase,
    snap_to_grid,
)

VERBOSE_LEVEL_NO = 15

# The SQL tables summary may read, so no other text reaches its query
SUMMARY_TABLES = ("summary",)


def _register_verbose_level():
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

    disruption_py's setup_logging and reset_handlers call logger.remove(),
    which would drop the workflow's log file sink (logs/<ds_name>_<time>.log).
    The preset setup flag skips both, so disruption_py logs through the caller's sinks.

    Returns:
        LogSettings that leave the caller's loguru sinks in place.
    """
    _register_verbose_level()
    return LogSettings(
        file_path=None, console_level="VERBOSE", _logging_has_been_setup=True
    )


def summary(
    summary_table: str,
    ipmax: float,
    pulse_length: float,
    min_shot: int,
    max_shot: int,
) -> np.ndarray:
    """Find shots with high enough current and long enough pulse length.

    Performs a SELECT query on a summary table. Snagged from
    https://github.com/MIT-PSFC/disruption-efit/blob/main/disruption_efit/sql.py

    Args:
        summary_table: Name of the summary table to query, one of SUMMARY_TABLES.
        ipmax: Threshold that maximum plasma current must exceed [A].
        pulse_length: Threshold that pulse length must exceed [s].
        min_shot: Disregard shots below this number. Ignored if not positive.
        max_shot: Disregard shots above this number. Ignored if not positive.

    Returns:
        Nx2 array of [shot_id, pulse_length] for the N shots which exceed the
        thresholds.

    Raises:
        ValueError: If summary_table is not one of SUMMARY_TABLES.
    """
    if summary_table not in SUMMARY_TABLES:
        raise ValueError(
            f"Unknown summary table '{summary_table}', one of {SUMMARY_TABLES}"
        )
    db = get_database(tokamak=resolve_tokamak_from_environment())

    # db.query takes no parameters, so every number is cast before it enters the text
    query = [
        f"select distinct(shot), pulse_length from {summary_table}",
        f"where ipmax > {float(ipmax)} and pulse_length > {float(pulse_length)}",
    ]
    if min_shot > 0:
        query += [f"and shot >= {int(min_shot)}"]
    if max_shot > 0:
        query += [f"and shot <= {int(max_shot)}"]
    query += ["order by shot"]
    logger.trace("> {query}", query=" ".join(query))

    # results
    data = db.query(" ".join(query), use_pandas=True).values

    logger.trace("= {shape}", shape=data.shape)
    return data


def retrieve_shot(
    tokamak: Tokamak,
    shot: int,
    retrieval_settings: RetrievalSettings,
    snap: bool = False,
) -> xr.Dataset | None:
    """One shot through disruption-py, in this process and through the caller's log sinks.

    Args:
        tokamak: The device.
        shot: Shot number.
        retrieval_settings: What to retrieve.
        snap: Snap the samples onto the uniform 1 kHz grid up to the last one (snap_to_grid),
            for a source on its own clock.

    Returns:
        The retrieval on dims (shot, time), or None when disruption-py returned nothing.
    """
    result = get_shots_data(
        tokamak=tokamak,
        shotlist_setting=[shot],
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if empty_result(result):
        return None
    if snap:
        last_time = float(result["time"].values.max())
        timebase = make_uniform_1kHz_timebase(last_time)
        result = snap_to_grid(result, timebase)
    return result.set_index(idx=["shot", "time"]).unstack("idx")


def empty_result(result: xr.Dataset) -> bool:
    """Check whether get_shots_data returned no usable data for a shot.

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
    """Place an injected heating power record on the timebase (injected_power_on_grid), in the record's units.

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
        power, record_time = params.mds_conn.get_data_with_dims(
            node, tree_name=tree_name
        )
    except mdsExceptions.MdsException:
        params.logger.debug("no {node} record, taking 0", node=node)
        return np.zeros(len(params.times))
    if record_time.size > 1 and record_time[-1] == 0:
        record_time = record_time[:-1]
        power = power[:-1]
    record_time_s = record_time * time_scale
    return injected_power_on_grid(record_time_s, power, params.times)


class UniformTimebaseSetting(TimeSetting):
    """The shot's uniform 1 kHz timebase from 0 to the end of its EFIT (make_uniform_1kHz_timebase).

    A device subclass reads the EFIT end time from its own tree (efit_end_time).
    """

    def efit_end_time(self, params: TimeSettingParams) -> float:
        """Read the last EFIT time of the shot.

        Args:
            params: Parameters needed to retrieve the timebase.

        Returns:
            The end time [s].
        """
        raise NotImplementedError

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """Build the shot's 1 kHz timebase out to the end of its EFIT.

        Args:
            params: Parameters needed to retrieve the timebase.

        Returns:
            Times from 0 to the last EFIT time in 1 ms steps [s].
        """
        end_time = self.efit_end_time(params)
        return make_uniform_1kHz_timebase(end_time)


def nan_contour_padding(
    r_contour: np.ndarray, z_contour: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the (0, 0) points EFIT pads a contour with into NaN.

    Args:
        r_contour: (..., n_points) contour major radii [m].
        z_contour: (..., n_points) contour heights [m].

    Returns:
        The two arrays as float with the padding NaN.
    """
    r_contour = np.asarray(r_contour, dtype=float)
    z_contour = np.asarray(z_contour, dtype=float)
    padding = (r_contour == 0.0) & (z_contour == 0.0)
    return np.where(padding, np.nan, r_contour), np.where(padding, np.nan, z_contour)


def efit_geqdsk_dataset(
    params: PhysicsMethodParams,
    efit_time: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    rcentr: float,
    geqdsk_data: dict[str, np.ndarray],
    rlim: np.ndarray | None,
    zlim: np.ndarray | None,
) -> xr.Dataset:
    """Build the GEQDSK block of an EFIT tree from its time-first node arrays, with its COCOS identified.

    EFIT pads the boundary and limiter contours with (0, 0) points to a fixed length,
    the stores pad with NaN (nan_contour_padding).

    Args:
        params: disruption-py physics method parameters for the shot.
        efit_time: (n_t,) reconstruction times [s].
        r_grid: (n_r,) grid major radii [m].
        z_grid: (n_z,) grid heights [m].
        rcentr: The radius bcentr is quoted at [m].
        geqdsk_data: The make_geqdsk_dataset signal arguments by name, time first (orient_signal). Modified in place.
        rlim: (n_lim,) limiter major radii [m], or None.
        zlim: (n_lim,) limiter heights [m], or None.

    Returns:
        The block on dim "idx" with "time" and "shot" coords and the "cocos" attribute.
    """
    geqdsk_data["rbdry"], geqdsk_data["zbdry"] = nan_contour_padding(
        geqdsk_data["rbdry"], geqdsk_data["zbdry"]
    )
    if rlim is not None:
        rlim, zlim = nan_contour_padding(rlim, zlim)
    cocos_input = cocos_from_signs(
        geqdsk_data["current"],
        geqdsk_data["bcentr"],
        geqdsk_data["simagx"],
        geqdsk_data["sibdry"],
        geqdsk_data["qpsi"],
        params.logger,
    )
    return make_geqdsk_dataset(
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
