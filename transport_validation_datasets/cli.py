"""Command line entry point for building a device's dataset, one device per invocation.

    python -m transport_validation_datasets.cli <device> <data_assembly_dir> [flags]

See DatasetCLI for the stages and the cluster setup, or run the CLI with --help
"""

import os
from pathlib import Path

import fire

# Dataset creation plots every shot and normally runs headless
os.environ.setdefault("MPLBACKEND", "Agg")

STAGES = ("unprocessed", "fit", "assemble", "all")

DEFAULT_METHOD = "zk"


def _build_cluster_config(
    cluster_ssh_host: str | None,
    cluster_partitions: str | None,
    cluster_remote_workdir: str | None,
    cluster_venv: str | None,
    cluster_max_jobs: int,
    cluster_shots_per_batch: int,
    cluster_cpus_per_job: int,
    cluster_mem: str | None,
    cluster_max_retries: int,
    cluster_pending_timeout_s: float,
):
    """Build a ClusterFitConfig from the cluster options, if any were given.

    Args:
        cluster_ssh_host: Host alias from ~/.ssh/config for the cluster login node.
        cluster_partitions: Ordered partition preference list,
            name@time_limit[@constraint], comma separated.
        cluster_remote_workdir: Scratch directory on the cluster. Each dataset
            keeps its batch files, worker package, and job logs in its own
            <workdir>/<ds_name>/ subdirectory.
        cluster_venv: Pre-built venv on the cluster (see bootstrap_remote.sh).
        cluster_max_jobs: Cap on simultaneously queued or running fitting jobs.
        cluster_shots_per_batch: Shots packed into one job.
        cluster_cpus_per_job: cpus-per-task of each fitting job.
        cluster_mem: Memory per node, e.g. "64G". None keeps the partition default.
        cluster_max_retries: Resubmissions allowed per batch after a job failure.
        cluster_pending_timeout_s: Seconds a job may sit PENDING before it moves
            to the next partition.

    Returns:
        The ClusterFitConfig, or None when no cluster host was given (fitting
        then runs single-threaded in this process).

    Raises:
        ValueError: If a host was given without the other options the cluster needs.
    """
    if cluster_ssh_host is None:
        return None
    missing = [
        name
        for name, value in (
            ("--cluster_partitions", cluster_partitions),
            ("--cluster_remote_workdir", cluster_remote_workdir),
            ("--cluster_venv", cluster_venv),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"{', '.join(missing)} are required with --cluster_ssh_host")

    from transport_validation_datasets.gp_fitting.dispatcher import ClusterFitConfig

    return ClusterFitConfig(
        ssh_host=cluster_ssh_host,
        partitions=cluster_partitions,
        remote_workdir=cluster_remote_workdir,
        venv_path=cluster_venv,
        max_concurrent_jobs=cluster_max_jobs,
        shots_per_batch=cluster_shots_per_batch,
        cpus_per_job=cluster_cpus_per_job,
        memory_per_node=cluster_mem,
        max_retries=cluster_max_retries,
        pending_timeout_s=cluster_pending_timeout_s,
    )


class DatasetCLI:
    """Build the dataset of one device: cmod, or mast.

        python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir

    One stage per invocation, or all of them with --stage all (the default):

        unprocessed: pull the source data, filter it, one netCDF per shot
        fit:         GP fit the Thomson profiles of every unprocessed shot
        assemble:    combine both into the final Zarr store

    Every stage resumes: work already on disk is skipped, so a killed run is
    restarted by running the same command again. Assembly is the exception, it
    always rebuilds the store from what the first two stages left on disk.

    GP fitting can be dispatched to a SLURM cluster. Set that up once per
    cluster: a Host entry in ~/.ssh/config with ControlMaster configured, then

        bash transport_validation_datasets/gp_fitting/bootstrap_remote.sh <host> <scratch-dir>

    and pass --cluster_ssh_host <host>, --cluster_partitions <spec>,
    --cluster_remote_workdir <scratch-dir>, and
    --cluster_venv <scratch-dir>/.venv. An authenticated ssh session to the
    login node must already exist (its 2FA prompt cannot be answered from here),
    so open one with a plain `ssh <host>` first.

    --cluster_partitions is an ordered preference list of comma-separated
    name@time_limit or name@time_limit@constraint entries, e.g.

        --cluster_partitions "sched_mit_psfc_r8@8:00:00,mit_preemptable@8:00:00@rocky8"

    Killed jobs are retried up to --cluster_max_retries times, and both retries
    and jobs stuck PENDING past --cluster_pending_timeout_s move to the next
    partition in the list (wrapping around).
    """

    def cmod(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "cmod",
        shotlist_file: Path | str | None = None,
        max_num_shots: int | None = None,
        stage: str = "all",
        method: str = DEFAULT_METHOD,
        clean_fit_state: bool = False,
        mb_per_chunk: int = 50,
        cluster_ssh_host: str | None = None,
        cluster_partitions: str | None = None,
        cluster_remote_workdir: str | None = None,
        cluster_venv: str | None = None,
        cluster_max_jobs: int = 8,
        cluster_shots_per_batch: int = 10,
        cluster_cpus_per_job: int = 32,
        cluster_mem: str | None = None,
        cluster_max_retries: int = 2,
        cluster_pending_timeout_s: float = 1800.0,
    ):
        """Build the C-Mod dataset, sourced from MDSplus through disruption-py.

        Args:
            data_assembly_dir: Directory holding the intermediate files, plots,
                logs, and final dataset.
            ds_name: Dataset name, used in paths and cluster job names.
            shotlist_file: File with one shot number per line. None queries the
                C-Mod SQL database instead.
            max_num_shots: Stop once this many shots have unprocessed data
                files. None processes the whole shotlist.
            stage: Which stage to run, one of STAGES.
            method: GP fitting method (see gp_fitting.registry).
            clean_fit_state: Before fitting, cancel this dataset's queued
                cluster jobs and delete every staged batch, so the fit starts
                from scratch. Destructive: fits already computed are lost.
                Unprocessed data files are kept.
            mb_per_chunk: Target size of a chunk of the final Zarr store,
                which is chunked along the shot dimension.
            cluster_ssh_host: See _build_cluster_config.
            cluster_partitions: See _build_cluster_config.
            cluster_remote_workdir: See _build_cluster_config.
            cluster_venv: See _build_cluster_config.
            cluster_max_jobs: See _build_cluster_config.
            cluster_shots_per_batch: See _build_cluster_config.
            cluster_cpus_per_job: See _build_cluster_config.
            cluster_mem: See _build_cluster_config.
            cluster_max_retries: See _build_cluster_config.
            cluster_pending_timeout_s: See _build_cluster_config.
        """
        from transport_validation_datasets.machine.cmod.cmod_dataset import (
            CModDataWorkflow,
        )

        workflow = CModDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=Path(data_assembly_dir),
            shotlist_file=shotlist_file,
            max_num_shots=max_num_shots,
            fit_method=method,
            cluster_config=_build_cluster_config(
                cluster_ssh_host=cluster_ssh_host,
                cluster_partitions=cluster_partitions,
                cluster_remote_workdir=cluster_remote_workdir,
                cluster_venv=cluster_venv,
                cluster_max_jobs=cluster_max_jobs,
                cluster_shots_per_batch=cluster_shots_per_batch,
                cluster_cpus_per_job=cluster_cpus_per_job,
                cluster_mem=cluster_mem,
                cluster_max_retries=cluster_max_retries,
                cluster_pending_timeout_s=cluster_pending_timeout_s,
            ),
        )
        _execute(workflow, stage, clean_fit_state, mb_per_chunk)

    def mast(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "mast",
        shotlist_file: Path | str | None = None,
        max_num_shots: int | None = None,
        stage: str = "all",
        method: str = DEFAULT_METHOD,
        clean_fit_state: bool = False,
        mb_per_chunk: int = 50,
        prepare_workers: int | None = None,
        cluster_ssh_host: str | None = None,
        cluster_partitions: str | None = None,
        cluster_remote_workdir: str | None = None,
        cluster_venv: str | None = None,
        cluster_max_jobs: int = 8,
        cluster_shots_per_batch: int = 10,
        cluster_cpus_per_job: int = 32,
        cluster_mem: str | None = None,
        cluster_max_retries: int = 2,
        cluster_pending_timeout_s: float = 1800.0,
    ):
        """Build the MAST dataset, sourced from the public level 2 Zarr store.

        Args:
            data_assembly_dir: Directory holding the intermediate files, plots,
                logs, and final dataset.
            ds_name: Dataset name, used in paths and cluster job names.
            shotlist_file: File with one shot number per line. None uses the
                shotlist shipped with the package.
            max_num_shots: Stop once this many shots have unprocessed data
                files. None processes the whole shotlist.
            stage: Which stage to run, one of STAGES.
            method: GP fitting method (see gp_fitting.registry).
            clean_fit_state: Before fitting, cancel this dataset's queued
                cluster jobs and delete every staged batch, so the fit starts
                from scratch. Destructive: fits already computed are lost.
                Unprocessed data files are kept.
            mb_per_chunk: Target size of a chunk of the final Zarr store,
                which is chunked along the shot dimension.
            prepare_workers: Threads used to read source data. None keeps the
                MAST default, which the public S3 store tolerates.
            cluster_ssh_host: See _build_cluster_config.
            cluster_partitions: See _build_cluster_config.
            cluster_remote_workdir: See _build_cluster_config.
            cluster_venv: See _build_cluster_config.
            cluster_max_jobs: See _build_cluster_config.
            cluster_shots_per_batch: See _build_cluster_config.
            cluster_cpus_per_job: See _build_cluster_config.
            cluster_mem: See _build_cluster_config.
            cluster_max_retries: See _build_cluster_config.
            cluster_pending_timeout_s: See _build_cluster_config.
        """
        from transport_validation_datasets.machine.mast.mast_dataset import (
            MASTDataWorkflow,
        )

        workflow = MASTDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=Path(data_assembly_dir),
            shotlist_file=shotlist_file,
            max_num_shots=max_num_shots,
            fit_method=method,
            cluster_config=_build_cluster_config(
                cluster_ssh_host=cluster_ssh_host,
                cluster_partitions=cluster_partitions,
                cluster_remote_workdir=cluster_remote_workdir,
                cluster_venv=cluster_venv,
                cluster_max_jobs=cluster_max_jobs,
                cluster_shots_per_batch=cluster_shots_per_batch,
                cluster_cpus_per_job=cluster_cpus_per_job,
                cluster_mem=cluster_mem,
                cluster_max_retries=cluster_max_retries,
                cluster_pending_timeout_s=cluster_pending_timeout_s,
            ),
            prepare_workers=prepare_workers,
        )
        _execute(workflow, stage, clean_fit_state, mb_per_chunk)


def _execute(workflow, stage: str, clean_fit_state: bool, mb_per_chunk: int):
    """Run the requested stages of an already built workflow.

    Args:
        workflow: The device's DataWorkflow.
        stage: Which stage to run, one of STAGES.
        clean_fit_state: Wipe the staged fit batches before fitting.
        mb_per_chunk: Target chunk size of the final Zarr store.

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
        workflow.run_gp_fitting()
    if stage in ("assemble", "all"):
        workflow.assemble_final_dataset(mb_per_chunk=mb_per_chunk)


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
