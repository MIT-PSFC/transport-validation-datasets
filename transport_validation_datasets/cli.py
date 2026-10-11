"""Command line entry point for building a device's dataset, one device per invocation.

    python -m transport_validation_datasets.cli <device> <data_assembly_dir> [flags]

One stage per invocation, or all of them with --stage all (the default):

    unprocessed: pull the source data, filter it, one netCDF per shot
    fit:         GP fit the Thomson profiles of every unprocessed shot
                 (DIII-D carries its IDA fits onto the fit grid instead)
    stack:       combine both into the internal Zarr store
    publish:     derive the published store from the internal one, with
                 the device's published_strip_signals stripped out

TCV and DIII-D data has no release permission, so those two stop at the internal store:
--stage all leaves publishing out, and --stage publish or export refuse them.
Every stage resumes from the work already on disk, except stack and publish, which rebuild their store.
The --config files, the cluster setup and the time windows of --shotlist_file are described in the README.
"""

import importlib
import os
from pathlib import Path

import fire
from loguru import logger

# Dataset creation plots every shot and normally runs headless
os.environ.setdefault("MPLBACKEND", "Agg")

STAGES = ("unprocessed", "fit", "stack", "publish", "export", "all")

# Device name -> (module, workflow class), imported only for the device being built.
# The names are also the device tables a config file may hold.
DEVICE_WORKFLOWS = {
    "cmod": (
        "transport_validation_datasets.machine.cmod.cmod_dataset",
        "CModDataWorkflow",
    ),
    "mast": (
        "transport_validation_datasets.machine.mast.mast_dataset",
        "MASTDataWorkflow",
    ),
    "tcv": ("transport_validation_datasets.machine.tcv.tcv_dataset", "TCVDataWorkflow"),
    "d3d": ("transport_validation_datasets.machine.d3d.d3d_dataset", "D3DDataWorkflow"),
}
DEVICES = tuple(DEVICE_WORKFLOWS)


def main(
    device: str,
    data_assembly_dir: Path | str,
    ds_name: str | None = None,
    shotlist_file: Path | str | None = None,
    max_num_shots: int | None = None,
    average_windows: bool = False,
    stage: str = "all",
    method: str | None = None,
    clean_fit_state: bool = False,
    skip_fit_plots: bool = False,
    max_fit_pages: int | None = None,
    mb_per_chunk: int = 50,
    prepare_workers: int | None = None,
    config: Path | str | None = None,
):
    """Build the dataset of one device: cmod, mast, tcv, or d3d.

    Args:
        device: The device, one of DEVICES.
        data_assembly_dir: Directory holding the intermediate files, plots, logs, and datasets.
        ds_name: Dataset name, used in paths and cluster job names. None takes the device name.
        shotlist_file: File with one shot number per line, or a CSV with
            shot, t_start and t_end columns for time windows.
            None takes the device's own shotlist (get_shotlist_from_source).
        max_num_shots: Stop once this many shots have unprocessed data files. None processes the whole shotlist.
        average_windows: Pool the Thomson points of each time window into one fit per window.
            Needs a shotlist with windows, DIII-D refuses it.
        stage: Which stage to run, one of STAGES.
        method: GP fitting method (see gp_fitting.registry), None takes the device's first fit_methods.
        clean_fit_state: Before fitting, cancel this dataset's queued cluster jobs and delete every staged batch,
            so the fit starts from scratch. Destructive: fits already computed are lost.
            Unprocessed data files are kept.
        skip_fit_plots: Leave out the fit stage's per-shot PDFs, about a minute per shot.
            A later fit stage without it plots the shots that have no PDF yet.
        max_fit_pages: Most pages of a shot's fit PDF. None plots every slice.
        mb_per_chunk: Target size of each variable's chunks in the Zarr stores, chunked along the shot dimension.
        prepare_workers: Threads used to read source data. None keeps the device's default.
        config: TOML file(s), comma separated, with the [cluster] table and the device's settings table
            (see config.py). None fits locally with the default settings.

    Raises:
        ValueError: If the device is not one of DEVICES.
    """
    if device not in DEVICE_WORKFLOWS:
        raise ValueError(f"Unknown device '{device}'. Known devices: {list(DEVICES)}")
    module_name, class_name = DEVICE_WORKFLOWS[device]
    device_module = importlib.import_module(module_name)
    workflow_cls = getattr(device_module, class_name)
    _run_device(
        workflow_cls,
        device,
        config=config,
        stage=stage,
        clean_fit_state=clean_fit_state,
        skip_fit_plots=skip_fit_plots,
        max_fit_pages=max_fit_pages,
        mb_per_chunk=mb_per_chunk,
        ds_name=ds_name or device,
        data_assembly_dir=Path(data_assembly_dir),
        shotlist_file=shotlist_file,
        max_num_shots=max_num_shots,
        average_windows=average_windows,
        fit_method=method,
        prepare_workers=prepare_workers,
    )


def _run_device(
    workflow_cls,
    device: str,
    *,
    config: Path | str | None,
    stage: str,
    clean_fit_state: bool,
    skip_fit_plots: bool,
    max_fit_pages: int | None,
    mb_per_chunk: int,
    **workflow_kwargs,
):
    """Build one device's workflow from its run configuration, then run the requested stages.

    The stage is checked first, building the workflow can query the device's shotlist.
    --stage all leaves publishing out for a device that is not publishable,
    an explicit --stage publish lets the workflow refuse it.

    Args:
        workflow_cls: The device's DataWorkflow subclass.
        device: The device, which names its config table.
        config: TOML file(s), comma separated, see config.py.
            None takes the default settings and fits locally.
        stage: Which stage to run, one of STAGES.
        clean_fit_state: Wipe the staged fit batches before fitting.
        skip_fit_plots: Leave out the fit stage's per-shot PDFs.
        max_fit_pages: Most pages of a shot's fit PDF, None for every slice.
        mb_per_chunk: Target size of each variable's chunks in the internal Zarr store.
        **workflow_kwargs: The workflow's own arguments.

    Raises:
        ValueError: If the stage is not one of STAGES.
    """
    from transport_validation_datasets.config import load_run_config

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Known stages: {list(STAGES)}")
    cluster_config, settings = load_run_config(
        config, device, workflow_cls.settings_cls, DEVICES
    )
    workflow = workflow_cls(
        cluster_config=cluster_config, settings=settings, **workflow_kwargs
    )
    if stage in ("unprocessed", "all"):
        workflow.make_unprocessed_data_files()
    if stage in ("fit", "all"):
        if clean_fit_state:
            workflow.clean_fit_state()
        workflow.run_gp_fitting(max_pages=max_fit_pages, skip_plots=skip_fit_plots)
    if stage in ("stack", "all"):
        workflow.stack_internal_dataset(mb_per_chunk=mb_per_chunk)
    if stage == "all" and not workflow.publishable:
        logger.info(
            f"{type(workflow).__name__} data stays internal, leaving publishing out"
        )
    elif stage in ("publish", "all"):
        workflow.publish_dataset(mb_per_chunk=mb_per_chunk)
    # Not part of "all": it needs the optional imas extra (see DataWorkflow.export_to_imas)
    if stage == "export":
        workflow.export_to_imas()


if __name__ == "__main__":
    fire.Fire(main)
