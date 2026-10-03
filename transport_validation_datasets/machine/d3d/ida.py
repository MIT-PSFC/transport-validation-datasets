"""The DIII-D IDA profile databases, and one IDA file's slices as unprocessed readings.

IDA (integrated data analysis) fits Te and ne with a GP on its own poloidal flux grid,
and the workflow carries those fits onto rho_tor_norm (the ida fit method).
No disruption-py or MDSplus here, so all of it is testable offline.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_validation_datasets.windows import read_shotlist

# The directory a database's shotlist name is resolved in
D3D_DIR = Path(__file__).parent

# IDA file variable (with an _err companion) -> standardized reading name
IDA_READINGS = {"T_e": "ida_t_e", "n_e": "ida_n_e"}


@dataclass(frozen=True)
class IdaDatabase:
    """An IDA profile database: a file pattern (may hold * wildcards) and, when set, the only shots it serves."""

    pattern: str
    shots: frozenset[int] | None = None

    def serves(self, shot: int) -> bool:
        """Whether the database serves a shot.

        Args:
            shot: Shot number.

        Returns:
            True when the database has no shotlist or the shot is on it.
        """
        return self.shots is None or shot in self.shots

    def path(self, shot: int) -> Path | None:
        """The shot's file in this database.

        Args:
            shot: Shot number.

        Returns:
            The first file matching the pattern, None when there is none or the database does not serve the shot.
        """
        if not self.serves(shot):
            return None
        candidate_name = self.pattern.format(shot=shot)
        candidate = Path(candidate_name)
        matches = sorted(candidate.parent.glob(candidate.name))
        if len(matches) > 1:
            logger.warning(
                f"Shot {shot}: {len(matches)} IDA files match {candidate}, using {matches[0]}"
            )
        return matches[0] if matches else None

    def available_shots(self) -> set[int]:
        """Every shot this database has a file for and serves.

        Returns:
            The shots.
        """
        pattern_path = Path(self.pattern)
        name_escaped = re.escape(pattern_path.name)
        name_regex = name_escaped.replace(re.escape("{shot}"), r"(\d+)").replace(
            re.escape("*"), ".*"
        )
        name_re = re.compile(name_regex)
        name_glob = pattern_path.name.format(shot="*")
        shots = set()
        for path in pattern_path.parent.glob(name_glob):
            match = name_re.fullmatch(path.name)
            if match:
                shots.add(int(match.group(1)))
        if self.shots is not None:
            shots &= self.shots
        return shots


def ida_databases(entries: list[dict[str, str]]) -> list[IdaDatabase]:
    """The IDA databases of the settings, in priority order.

    Args:
        entries: One {"pattern": ..., "shotlist": ...} table per database (D3DSettings.ida_databases).
            A shotlist is a file name in this package's d3d directory, absent for a database that serves every shot.

    Returns:
        The databases.
    """
    databases = []
    for entry in entries:
        shotlist_name = entry.get("shotlist")
        shots = None
        if shotlist_name is not None:
            shotlist, _ = read_shotlist(D3D_DIR / shotlist_name)
            shots = frozenset(shotlist)
        databases.append(IdaDatabase(pattern=entry["pattern"], shots=shots))
    return databases


def find_ida_path(shot: int, databases: list[IdaDatabase]) -> Path | None:
    """The shot's IDA file from the first database that has one.

    Args:
        shot: Shot number.
        databases: The databases in priority order.

    Returns:
        The file, None when no database has one for the shot.
    """
    for database in databases:
        path = database.path(shot)
        if path is not None:
            return path
    return None


def find_ida_shots(databases: list[IdaDatabase]) -> list[int]:
    """Every shot some database has a file for and serves.

    Args:
        databases: The databases.

    Returns:
        The shots, sorted.
    """
    shots: set[int] = set()
    for database in databases:
        shots |= database.available_shots()
    return sorted(shots)


def ida_dataset(path: Path, shot: int) -> xr.Dataset:
    """One IDA file's slices as unprocessed readings, ready for snap_to_grid.

    Each IDA slice is the GP fit of Te and ne on IDA's own psi_N points, with its 1-sigma error.
    The psi_N points are static per file and only differ between databases.
    Times are milliseconds in the file.

    Args:
        path: The IDA file.
        shot: Shot number.

    Returns:
        ida_t_e [eV], ida_n_e [m^-3] and their _error on ("idx", "ida_point") with "time" [s] and "shot" coords,
        and ida_psi_n on ("ida_point",).
    """
    with xr.open_dataset(path) as ida_file:
        ida = ida_file.load().sortby("time")
    ida_time = ida["time"].values / 1e3
    psi_n_points = np.asarray(ida["psi_n"].values, dtype=float)
    data_vars = {"ida_psi_n": (("ida_point",), psi_n_points)}
    for ida_name, reading_name in IDA_READINGS.items():
        readings = ida[ida_name].transpose("time", "psi_n").values
        errors = ida[f"{ida_name}_err"].transpose("time", "psi_n").values
        data_vars[reading_name] = (("idx", "ida_point"), readings.astype(float))
        data_vars[f"{reading_name}_error"] = (
            ("idx", "ida_point"),
            errors.astype(float),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("idx", ida_time),
            "shot": ("idx", np.repeat(shot, ida_time.size)),
            "ida_point": np.arange(psi_n_points.size),
        },
    )
