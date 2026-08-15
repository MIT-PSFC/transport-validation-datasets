import shutil
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
    test_dir: Path, shotlist: list[int] | None = None, **kwargs
) -> CModDataWorkflow:
    # Delete the test directory if it exists to ensure a clean test environment
    if test_dir.exists():
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

        # Check that unprocessed data files were created for each shot
        for shot in workflow.shotlist[:max_num_shots]:
            file_path = workflow.unprocessed_data_dir / f"{shot}.nc"
            assert file_path.exists(), (
                f"Unprocessed data file for shot {shot} does not exist"
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


class TestGPFit:
    test_dir = TEST_DIR / "test_gp_fit"

    @pytest.mark.slow  # serial GP fit, ~80s per TS slice
    @pytest.mark.parametrize("method", ["zk"])
    def test_serial(self, method: str):
        # Basic check that GP fitting can be performed on unprocessed data files
        test_dir = self.test_dir / "test_serial" / method
        max_num_shots = 1
        workflow = cmod_workflow(test_dir, max_num_shots=max_num_shots)

        workflow.make_unprocessed_data_files()

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

        # Trim the files to three TS slices each to reduce test time
        for shot in workflow.shotlist[:max_num_shots]:
            _trim_to_three_ts_slices(workflow.unprocessed_data_dir / f"{shot}.nc")

        workflow.run_gp_fitting(method=method)

        # Check that the collected fit results file was created and covers the shots
        results_path = workflow.fit_results_dir / method / "fit_results.nc"
        assert results_path.exists(), "Collected GP fit results file does not exist"
        results = xr.open_dataset(results_path)
        for shot in workflow.shotlist[:max_num_shots]:
            assert shot in results["shot"].values, f"GP fit results missing shot {shot}"

    def _ssh_host_configured(alias: str) -> bool:
        # True if ~/.ssh/config has a Host entry naming this alias
        cfg = Path.home() / ".ssh" / "config"
        if not cfg.exists():
            return False
        for line in cfg.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].lower() == "host" and alias in parts[1:]:
                return True
        return False

    @pytest.mark.skipif(
        not _ssh_host_configured("orcd-login"),
        reason="no orcd-login entry in ~/.ssh/config",
    )
    @pytest.mark.parametrize("method", ["zk"])
    def test_dispatched(self, method: str):
        # check that GP fitting can be done via the dispatcher on a cluster (requires ssh config set up)
        test_dir = self.test_dir / "test_dispatched" / method
        max_num_shots = 1
        cluster_config = ClusterFitConfig(
            ssh_host="orcd-login",
            partitions="sched_mit_psfc_r8",
            job_name_prefix="test_gpfit",
        )
        workflow = cmod_workflow(
            test_dir, max_num_shots=max_num_shots, cluster_config=cluster_config
        )

        workflow.make_unprocessed_data_files()

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

        # Trim the files to three TS slices each to reduce test time
        for shot in workflow.shotlist[:max_num_shots]:
            _trim_to_three_ts_slices(workflow.unprocessed_data_dir / f"{shot}.nc")

        workflow.run_gp_fitting(method=method)

        # Check that the collected fit results file was created and covers the shots
        results_path = workflow.fit_results_dir / method / "fit_results.nc"
        assert results_path.exists(), "Collected GP fit results file does not exist"
        results = xr.open_dataset(results_path)
        for shot in workflow.shotlist[:max_num_shots]:
            assert shot in results["shot"].values, f"GP fit results missing shot {shot}"
