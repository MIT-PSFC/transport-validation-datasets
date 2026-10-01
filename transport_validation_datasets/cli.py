"""Command line entry point for building a device's dataset, one device per invocation.

    python -m transport_validation_datasets.cli <device> <data_assembly_dir> [flags]

See DatasetCLI for the stages and the cluster setup, or run the CLI with --help
"""

import os
from pathlib import Path

import fire

# Dataset creation plots every shot and normally runs headless
os.environ.setdefault("MPLBACKEND", "Agg")

STAGES = ("unprocessed", "fit", "stack", "publish", "export", "all")

# The subcommands below, and the device tables a config file may hold.
DEVICES = ("cmod", "mast")

DEFAULT_METHOD = "zk"


class DatasetCLI:
    """Build the dataset of one device: cmod, or mast.

        python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir

    One stage per invocation, or all of them with --stage all (the default):

        unprocessed: pull the source data, filter it, one netCDF per shot
        fit:         GP fit the Thomson profiles of every unprocessed shot
        stack:       combine both into the internal Zarr store
        publish:     derive the published store from the internal one, with
                     the device's published_strip_signals stripped out

    Every stage resumes: work already on disk is skipped, so a killed run is
    restarted by running the same command again. Stack and publish are the
    exceptions: stack always rebuilds the internal store from what the first
    two stages left on disk, and publish always rebuilds the published store
    from the internal one.

    Configuration. The cluster and the device settings live in TOML files
    passed through --config, comma separated and layered (a later file
    overrides an earlier one key by key, see config.py). The [cluster] table
    dispatches the fit stage to a SLURM cluster, the [cmod] or [mast] table
    sets that device's workflow settings, e.g. turning on C-Mod's fallback
    to the ANALYSIS tree for shots EFIT21 fails on (the default reads EFIT21 only):

        [cmod]
        efit_trees = ["EFIT21", "ANALYSIS"]

    configs/orcd.toml is the shared file; the per-user cluster paths go in a
    second, gitignored file:

        --config configs/orcd.toml,configs/$USER.user.toml

    Without --config the fits run locally and every setting keeps its default.

    GP fitting on a SLURM cluster. Set that up once per cluster: a Host entry
    in ~/.ssh/config with ControlMaster configured, then

        bash transport_validation_datasets/gp_fitting/bootstrap_remote.sh <host> <scratch-dir>

    and describe it in the [cluster] table, whose keys are the fields of
    gp_fitting.dispatcher.ClusterFitConfig:

        [cluster]
        ssh_host = "<host>"
        partitions = "sched_mit_psfc_r8@8:00:00,mit_preemptable@8:00:00@rocky8"
        remote_workdir = "<scratch-dir>"
        venv_path = "<scratch-dir>/.venv"

    An authenticated ssh session to the login node must already exist (its
    2FA prompt cannot be answered from here), so open one with a plain
    `ssh <host>` first. partitions is an ordered preference list of
    name@time_limit or name@time_limit@constraint entries, comma separated
    or a TOML list. Killed jobs are retried up to max_retries times, and both
    retries and jobs stuck PENDING past pending_timeout_s move to the next
    partition in the list (wrapping around).

    Time windows. --shotlist_file takes either one shot number per line,
    or a CSV with a header holding shot (or pulse_no), t_start and t_end [s],
    one row per window and a shot on as many rows as it has windows (other columns are ignored).
    If windows are provided only the Thomson samples inside them are fit,
    and the stores hold only the grid times inside them.
    Add --average_windows to pool every Thomson point of a window into one fit.
    The store then holds that profile over the whole window with fresh_profile marking the window center.
    Staged batches and fit results record the windows and mode they were built with,
    and a run whose shotlist disagrees with them stops with an error before fitting or stacking.
    """

    def cmod(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "cmod",
        shotlist_file: Path | str | None = None,
        max_num_shots: int | None = None,
        average_windows: bool = False,
        stage: str = "all",
        method: str = DEFAULT_METHOD,
        clean_fit_state: bool = False,
        skip_fit_plots: bool = False,
        mb_per_chunk: int = 50,
        config: Path | str | None = None,
    ):
        """Build the C-Mod dataset, sourced from MDSplus through disruption-py.

        Args:
            data_assembly_dir: Directory holding the intermediate files, plots,
                logs, and datasets.
            ds_name: Dataset name, used in paths and cluster job names.
            shotlist_file: File with one shot number per line, or a CSV with
                shot, t_start and t_end columns for time windows.
                None queries the C-Mod SQL database instead.
            max_num_shots: Stop once this many shots have unprocessed data
                files. None processes the whole shotlist.
            average_windows: Pool the Thomson points of each time window into
                one fit per window. Needs a shotlist with windows.
            stage: Which stage to run, one of STAGES.
            method: GP fitting method (see gp_fitting.registry).
            clean_fit_state: Before fitting, cancel this dataset's queued
                cluster jobs and delete every staged batch, so the fit starts
                from scratch. Destructive: fits already computed are lost.
                Unprocessed data files are kept.
            skip_fit_plots: Leave out the fit stage's per-shot PDFs, about a minute per shot.
                A later fit stage without it plots the shots that have no PDF yet.
            mb_per_chunk: Target size of each variable's chunks in the internal
                Zarr store, which is chunked along the shot dimension.
            config: TOML file(s), comma separated, with the [cluster] table
                and the [cmod] settings table (see config.py, CModSettings).
                None fits locally with the default settings.
        """
        from transport_validation_datasets.config import load_run_config
        from transport_validation_datasets.machine.cmod.cmod_dataset import (
            CModDataWorkflow,
        )

        cluster_config, settings = load_run_config(
            config, "cmod", CModDataWorkflow.settings_cls, DEVICES
        )
        workflow = CModDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=Path(data_assembly_dir),
            shotlist_file=shotlist_file,
            max_num_shots=max_num_shots,
            average_windows=average_windows,
            fit_method=method,
            cluster_config=cluster_config,
            settings=settings,
        )
        _execute(workflow, stage, clean_fit_state, skip_fit_plots, mb_per_chunk)

    def mast(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "mast",
        shotlist_file: Path | str | None = None,
        max_num_shots: int | None = None,
        average_windows: bool = False,
        stage: str = "all",
        method: str = DEFAULT_METHOD,
        clean_fit_state: bool = False,
        skip_fit_plots: bool = False,
        mb_per_chunk: int = 50,
        prepare_workers: int | None = None,
        config: Path | str | None = None,
    ):
        """Build the MAST dataset, sourced from the public level 2 Zarr store.

        Args:
            data_assembly_dir: Directory holding the intermediate files, plots,
                logs, and datasets.
            ds_name: Dataset name, used in paths and cluster job names.
            shotlist_file: File with one shot number per line, or a CSV with
                shot, t_start and t_end columns for time windows.
                None uses the shotlist shipped with the package.
            max_num_shots: Stop once this many shots have unprocessed data
                files. None processes the whole shotlist.
            average_windows: Pool the Thomson points of each time window into
                one fit per window. Needs a shotlist with windows.
            stage: Which stage to run, one of STAGES.
            method: GP fitting method (see gp_fitting.registry).
            clean_fit_state: Before fitting, cancel this dataset's queued
                cluster jobs and delete every staged batch, so the fit starts
                from scratch. Destructive: fits already computed are lost.
                Unprocessed data files are kept.
            skip_fit_plots: Leave out the fit stage's per-shot PDFs, about a minute per shot.
                A later fit stage without it plots the shots that have no PDF yet.
            mb_per_chunk: Target size of each variable's chunks in the internal
                Zarr store, which is chunked along the shot dimension.
            prepare_workers: Threads used to read source data. None keeps the
                MAST default, which the public S3 store tolerates.
            config: TOML file(s), comma separated, with the [cluster] table
                and the [mast] settings table (see config.py; MAST has no
                settings yet, so the table is empty or absent). None fits locally.
        """
        from transport_validation_datasets.config import load_run_config
        from transport_validation_datasets.machine.mast.mast_dataset import (
            MASTDataWorkflow,
        )

        cluster_config, settings = load_run_config(
            config, "mast", MASTDataWorkflow.settings_cls, DEVICES
        )
        workflow = MASTDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=Path(data_assembly_dir),
            shotlist_file=shotlist_file,
            max_num_shots=max_num_shots,
            average_windows=average_windows,
            fit_method=method,
            cluster_config=cluster_config,
            prepare_workers=prepare_workers,
            settings=settings,
        )
        _execute(workflow, stage, clean_fit_state, skip_fit_plots, mb_per_chunk)


def _execute(
    workflow,
    stage: str,
    clean_fit_state: bool,
    skip_fit_plots: bool,
    mb_per_chunk: int,
):
    """Run the requested stages of an already built workflow.

    Args:
        workflow: The device's DataWorkflow.
        stage: Which stage to run, one of STAGES.
        clean_fit_state: Wipe the staged fit batches before fitting.
        skip_fit_plots: Leave out the fit stage's per-shot PDFs.
        mb_per_chunk: Target size of each variable's chunks in the internal Zarr store.

    Raises:
        ValueError: If the stage is not one of STAGES.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Known stages: {list(STAGES)}")
    if stage in ("unprocessed", "all"):
        workflow.make_unprocessed_data_files()
    if stage in ("fit", "all"):
        if clean_fit_state:
            workflow.clean_fit_state()
        workflow.run_gp_fitting(skip_plots=skip_fit_plots)
    if stage in ("stack", "all"):
        workflow.stack_internal_dataset(mb_per_chunk=mb_per_chunk)
    if stage in ("publish", "all"):
        workflow.publish_dataset()
    # Deliberately not part of "all": needs the optional `imas` extra, and is
    # its own opt-in step (see DataWorkflow.export_to_imas).
    if stage == "export":
        workflow.export_to_imas()


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
