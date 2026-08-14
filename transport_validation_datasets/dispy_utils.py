from functools import partialmethod

import numpy as np
from disruption_py.machine.tokamak import resolve_tokamak_from_environment
from disruption_py.settings import LogSettings
from disruption_py.workflow import get_database
from loguru import logger

VERBOSE_LEVEL_NO = 15


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

    disruption_py's setup_logging()/reset_handlers() call logger.remove(),
    which wipes every sink on the global loguru logger - including the
    dataset CLI's raw_data_{pid}.log file sink, silently emptying the run
    log. Pre-setting the setup flag and a console level skips both reset
    paths in disruption_py's get_shots_data, so its messages flow through
    whatever sinks the caller already configured.

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
