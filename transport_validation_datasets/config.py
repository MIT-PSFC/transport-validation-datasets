"""Run configuration from TOML files: the cluster and the device settings.

The CLI takes the files through --config, one or more comma separated. They
are layered: the tables are merged key by key and a later file overrides an
earlier one, so a shared file (configs/orcd.toml) carries the cluster and the
device settings, and a per-user file (configs/<user>.user.toml, gitignored)
carries the user-specific paths. The tables:

    [cluster]   SLURM dispatch of the fit stage, fields of gp_fitting.dispatcher.ClusterFitConfig.
                If this section is absent, the fits run locally.
    [<device>]  Settings of a device's workflow ([cmod], [mast]),
                to modify the fields of the workflow's settings_cls.
                If this section is absent, the device's workflow uses its defaults.
                One file may hold the tables of several devices,
                only the table of the device being built is read.

Keys are the dataclass field names, so those dataclasses are the reference
for what each table takes and what the defaults are.
A key the dataclass has no field for, a required key that is missing,
or an unexpected table raises an error, so a typo cannot silently fall back to a default.

    [cluster]
    ssh_host = "orcd-login"
    partitions = "sched_mit_psfc_r8@11:00:00"
    remote_workdir = "/path/on/cluster"
    venv_path = "/path/on/cluster/.venv"

    [cmod]
    efit_nickname = "EFIT21"
"""

import dataclasses
import tomllib
from collections.abc import Sequence
from pathlib import Path

from transport_validation_datasets.gp_fitting.dispatcher import ClusterFitConfig
from transport_validation_datasets.workflow import DeviceSettings

CLUSTER_TABLE = "cluster"


def config_paths(value: Path | str | Sequence[Path | str]) -> list[Path]:
    """Split a --config value into file paths.

    Args:
        value: One path, a comma separated string of paths, or a sequence of paths.

    Returns:
        The paths, in the order given.
    """
    if isinstance(value, (str, Path)):
        parts = str(value).split(",")
    else:
        parts = [str(p) for p in value]
    return [Path(p.strip()) for p in parts if p.strip()]


def read_config(paths: Sequence[Path | str]) -> dict:
    """Parse TOML config files into one config, later files overriding earlier ones.

    Tables are merged key by key: a key set in two files takes the value of
    the later file, keys set in only one file are kept.

    Args:
        paths: The files, in layering order.

    Returns:
        The merged tables.
    """
    merged: dict = {}
    for path in paths:
        with open(path, "rb") as f:
            config = tomllib.load(f)
        for table, section in config.items():
            if isinstance(section, dict) and isinstance(merged.get(table), dict):
                merged[table] = {**merged[table], **section}
            else:
                merged[table] = section
    return merged


def build_from_table(cls, config: dict, table: str):
    """Instantiate a dataclass from one table of the config.

    Args:
        cls: The dataclass whose fields the table holds.
        config: The parsed config.
        table: Name of the table.

    Returns:
        The instance, or None when the config has no such table.

    Raises:
        ValueError: If the table is not a table, has a key the dataclass has
            no field for, or lacks a field that has no default.
    """
    section = config.get(table)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"[{table}] must be a table, got {type(section).__name__}")
    fields = dataclasses.fields(cls)
    names = [f.name for f in fields]
    unknown = sorted(set(section) - set(names))
    if unknown:
        raise ValueError(f"[{table}] has unknown keys {unknown}. Known keys: {names}")
    required = [
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    missing = [name for name in required if name not in section]
    if missing:
        raise ValueError(f"[{table}] is missing required keys {missing}")
    return cls(**section)


def load_run_config(
    config: Path | str | Sequence[Path | str] | None,
    device: str,
    settings_cls: type[DeviceSettings],
    devices: Sequence[str],
) -> tuple[ClusterFitConfig | None, DeviceSettings | None]:
    """Read the cluster config and one device's settings from the config files.

    Args:
        config: The TOML files (see config_paths), or None for no config
            (local fits, default settings).
        device: Name of the device being built, the table that is read.
        settings_cls: The device workflow's settings dataclass.
        devices: Every device name the CLI knows, the tables allowed next to
            [cluster].

    Returns:
        The cluster config and the device settings, each None when its table
        is absent.

    Raises:
        ValueError: If the file has a table that is neither cluster nor a device.
    """
    if config is None:
        return None, None
    paths = config_paths(config)
    tables = read_config(paths)
    allowed = {CLUSTER_TABLE, *devices}
    unknown = sorted(set(tables) - allowed)
    if unknown:
        raise ValueError(
            f"{[str(p) for p in paths]} has unknown tables {unknown}. "
            f"Known tables: {sorted(allowed)}"
        )
    cluster_config = build_from_table(ClusterFitConfig, tables, CLUSTER_TABLE)
    settings = build_from_table(settings_cls, tables, device)
    return cluster_config, settings
