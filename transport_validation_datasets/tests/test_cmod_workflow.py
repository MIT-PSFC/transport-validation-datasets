import shutil
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.gp_fitting.dispatcher import ClusterFitConfig
from transport_validation_datasets.machine.cmod.cmod_dataset import CModDataWorkflow

pytestmark = pytest.mark.skipif(
    not Path("/usr/local/mfe/ml_data_dump").exists(),
    reason="C-Mod data source not available on this system",
)

TEST_DIR = PACKAGE_ROOT / "tests" / "test_outputs" / "test_cmod_workflow"


def cmod_workflow(
    test_dir: Path, shotlist: list[int] | None = None, clean=True, **kwargs
) -> CModDataWorkflow:
    # Delete the test directory if it exists to ensure a clean test environment
    if clean and test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    if shotlist:
        shotlist_file = test_dir / "shotlist.txt"
        with open(shotlist_file, "w") as f:
            for shot in shotlist:
                f.write(f"{shot}\n")
    else:
        shotlist_file = None

    workflow = CModDataWorkflow(
        ds_name="cmod_test",
        shotlist_file=shotlist_file,
        data_assembly_dir=test_dir,
        **kwargs,
    )
    return workflow


class TestMakeUnprocessedDataFiles:
    test_dir = TEST_DIR / "test_make_unprocessed_data_files"

    def test_basic(self):
        # Basic check that unprocessed data files are created for each shot in the shotlist
        test_dir = self.test_dir / "test_basic"
        max_num_shots = 2
        workflow = cmod_workflow(test_dir, max_num_shots=max_num_shots)

        workflow.make_unprocessed_data_files()

        # Check that as many unprocessed data files as were asked for got created
        shots = workflow.unprocessed_shots()
        assert len(shots) == max_num_shots, (
            f"Expected {max_num_shots} unprocessed data files, got {len(shots)}: {shots}"
        )

    def test_graceful_skip(self):
        # Check that shots with known problems are handled gracefully
        test_dir = self.test_dir / "test_graceful_skip"

        shotlist_missing = [
            1160503006,  # Missing TS data
            1160503011,  # Missing EFIT data
            1160503015,  # No EFIT21 tree at all
        ]
        shotlist_present = [
            1160503007,  # Should be present
        ]
        shotlist = shotlist_missing + shotlist_present
        workflow = cmod_workflow(test_dir, shotlist=shotlist)

        workflow.make_unprocessed_data_files()

        for shot in shotlist_present:
            file_path = workflow.unprocessed_data_dir / f"{shot}.nc"
            assert file_path.exists(), (
                f"Unprocessed data file for shot {shot} does not exist"
            )

        for shot in shotlist_missing:
            file_path = workflow.unprocessed_data_dir / f"{shot}.nc"
            assert not file_path.exists(), (
                f"Unprocessed data file for shot {shot} should not exist"
            )


def _trim_to_three_ts_slices(nc_path: Path):
    # Keep only the first, middle, and last TS slices of an unprocessed data
    # file (the serial GP fit takes ~80s per slice, a full shot has ~90)
    ds = xr.load_dataset(nc_path)
    ts_vars = [
        "ts_channel_t_e",
        "ts_channel_t_e_error",
        "ts_channel_n_e",
        "ts_channel_n_e_error",
    ]
    # TS slices are the times where any channel has a finite te or ne
    ts_any = (
        (ds["ts_channel_t_e"].notnull() | ds["ts_channel_n_e"].notnull())
        .any(dim="ts_channel")
        .squeeze("shot", drop=True)
        .transpose("time")
        .values
    )
    ts_idx = np.flatnonzero(ts_any)
    keep = ts_idx[[0, len(ts_idx) // 2, -1]]
    drop = np.setdiff1d(ts_idx, keep)
    for name in ts_vars:
        ds[name][{"time": drop}] = np.nan
    ds.to_netcdf(nc_path)


@cache
def ssh_host_reachable(alias: str) -> bool:
    # The dispatcher needs an already-authenticated ControlMaster session
    # (the login node's 2FA prompt cannot be answered from a test), so
    # check that one really answers rather than just that the alias exists.
    # Cached: this runs at collection time, once per alias, not per skipif.
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, "true"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class TestGPFit:
    test_dir = TEST_DIR / "test_gp_fit"

    @pytest.mark.slow  # serial GP fit, ~80s per TS slice
    @pytest.mark.parametrize("method", ["zk"])
    def test_serial(self, method: str):
        # Basic check that GP fitting can be performed on unprocessed data files
        test_dir = self.test_dir / "test_serial" / method
        max_num_shots = 1
        workflow = cmod_workflow(
            test_dir, max_num_shots=max_num_shots, fit_method=method
        )

        workflow.make_unprocessed_data_files()

        # Trim the files to three TS slices each to reduce test time
        for shot in workflow.unprocessed_shots():
            _trim_to_three_ts_slices(workflow.unprocessed_data_dir / f"{shot}.nc")

        workflow.run_gp_fitting(max_pages=20)

        # Check that the per-shot fit result files were written and cover the shots
        for shot in workflow.unprocessed_shots():
            shot_path = workflow.fit_shots_dir / f"{shot}.nc"
            assert shot_path.exists(), (
                f"GP fit results file for shot {shot} does not exist"
            )

    @pytest.mark.skipif(
        not ssh_host_reachable("orcd-login"),
        reason="no authenticated ssh session to orcd-login",
    )
    @pytest.mark.parametrize("method", ["zk"])
    def test_dispatched(self, method: str):
        # check that GP fitting can be done via the dispatcher on a cluster (requires ssh config set up)
        test_dir = self.test_dir / "test_dispatched" / method
        max_num_shots = 1
        cluster_config = ClusterFitConfig(
            ssh_host="orcd-login",
            partitions="sched_mit_psfc_r8@8:00:00",
            job_name_prefix="test_gpfit",
            remote_workdir="/home/zkeith/orcd/scratch/tests/transport_validation_datasets/test_dispatched",
            venv_path="/home/zkeith/orcd/scratch/tests/transport_validation_datasets/.venv",
            shots_per_batch=1,
        )

        # Delete all files (minus the venv) in the remote workdir to ensure a clean test environment
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                cluster_config.ssh_host,
                f"mkdir -p {cluster_config.remote_workdir} && "
                f"cd {cluster_config.remote_workdir} && "
                "find . -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +",
            ],
            check=True,
        )

        workflow = cmod_workflow(
            test_dir,
            max_num_shots=max_num_shots,
            fit_method=method,
            cluster_config=cluster_config,
        )

        workflow.make_unprocessed_data_files()

        # Trim the files to three TS slices each to reduce test time
        for shot in workflow.unprocessed_shots():
            _trim_to_three_ts_slices(workflow.unprocessed_data_dir / f"{shot}.nc")

        workflow.run_gp_fitting(max_pages=20)

        # Check that the per-shot fit result files were written and cover the shots
        for shot in workflow.unprocessed_shots():
            shot_path = workflow.fit_shots_dir / f"{shot}.nc"
            assert shot_path.exists(), (
                f"GP fit results file for shot {shot} does not exist"
            )

    @pytest.mark.slow
    @pytest.mark.skipif(
        not ssh_host_reachable("orcd-login"),
        reason="no authenticated ssh session to orcd-login",
    )
    @pytest.mark.parametrize("method", ["zk"])
    def test_dispatched_20(self, method: str):
        # check that GP fitting can be done via the dispatcher on a cluster for many full shots
        test_dir = self.test_dir / "test_dispatched_20" / method
        max_num_shots = 20
        cluster_config = ClusterFitConfig(
            ssh_host="orcd-login",
            partitions="sched_mit_psfc_r8@8:00:00",
            job_name_prefix="test_gpfit",
            remote_workdir="/home/zkeith/orcd/scratch/tests/transport_validation_datasets/test_dispatched_20",
            venv_path="/home/zkeith/orcd/scratch/tests/transport_validation_datasets/.venv",
            shots_per_batch=4,
        )

        # Delete all files (minus the venv) in the remote workdir to ensure a clean test environment
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                cluster_config.ssh_host,
                f"mkdir -p {cluster_config.remote_workdir} && "
                f"cd {cluster_config.remote_workdir} && "
                "find . -mindepth 1 -maxdepth 1 ! -name '.venv' -exec rm -rf {} +",
            ],
            check=True,
        )

        workflow = cmod_workflow(
            test_dir,
            max_num_shots=max_num_shots,
            fit_method=method,
            cluster_config=cluster_config,
        )

        workflow.make_unprocessed_data_files()
        workflow.run_gp_fitting()

        # Check that every shot with an unprocessed data file made it into the
        # fit results (shots skipped at the unprocessed stage do not count)
        for shot in workflow.unprocessed_shots():
            shot_path = workflow.fit_shots_dir / f"{shot}.nc"
            assert shot_path.exists(), (
                f"GP fit results file for shot {shot} does not exist"
            )


class TestFinalAssembly:
    test_dir = TEST_DIR / "test_final_assembly"

    def test_basic(self):
        # Basic check that the final assembly can be performed on GP fit results
        test_dir = self.test_dir / "test_basic"
        max_num_shots = 6
        workflow = cmod_workflow(
            test_dir, clean=False, max_num_shots=max_num_shots, fit_method="zk"
        )
        if workflow.final_ds_dir.exists():
            shutil.rmtree(workflow.final_ds_dir)

        workflow.make_unprocessed_data_files()
        for shot in workflow.unprocessed_shots():
            _trim_to_three_ts_slices(workflow.unprocessed_data_dir / f"{shot}.nc")

        workflow.run_gp_fitting(max_pages=20)
        workflow.assemble_final_dataset()

        # Check that the final assembled dataset was created and covers the shots
        final_ds_path = workflow.final_ds_dir / "cmod_test.zarr"
        assert final_ds_path.exists(), "Final assembled dataset does not exist"
        final_ds = xr.open_dataset(final_ds_path)
        for shot in workflow.unprocessed_shots():
            assert shot in final_ds["shot"].values, f"Final dataset missing shot {shot}"
