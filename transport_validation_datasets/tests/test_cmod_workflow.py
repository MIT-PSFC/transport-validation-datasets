import shutil
from pathlib import Path

import pytest

from transport_validation_datasets import PACKAGE_ROOT
from transport_validation_datasets.machine.cmod.cmod_dataset import CModDataWorkflow

pytestmark = pytest.mark.skipif(
    not Path("/usr/local/mfe/ml_data_dump").exists(),
    reason="C-Mod data source not available on this system",
)

TEST_DIR = PACKAGE_ROOT / "tests" / "test_outputs" / "test_cmod_workflow"


def cmod_workflow(
    test_dir: Path, shotlist: list[int] | None = None, max_num_shots: int | None = None
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
        max_num_shots=max_num_shots,
        data_assembly_dir=test_dir,
    )
    return workflow


class TestMakeUnprocessedDataFiles:
    TEST_DIR = TEST_DIR / "test_make_unprocessed_data_files"

    def test_basic(self):
        # Basic check that unprocessed data files are created for each shot in the shotlist
        test_dir = TEST_DIR / "test_basic"
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
        test_dir = TEST_DIR / "test_graceful_skip"

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
