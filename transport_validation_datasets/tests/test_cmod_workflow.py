from pathlib import Path

import pytest

from transport_validation_datasets.machine.cmod.cmod_dataset import CModDataWorkflow

pytestmark = pytest.mark.skipif(
    not Path("/usr/local/mfe/ml_data_dump").exists(),
    reason="C-Mod data source not available on this system",
)


@pytest.fixture(scope="module")
def cmod_workflow(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cmod_processing")
    workflow = CModDataWorkflow(
        ds_name="cmod_test", max_num_shots=2, data_assembly_dir=tmp
    )
    return workflow


class TestMakeUnprocessedDataFiles:
    def test_make_unprocessed_data_files(self, cmod_workflow: CModDataWorkflow):
        cmod_workflow.make_unprocessed_data_files()
        # Check that unprocessed data files were created for each shot
        for shot in cmod_workflow.shotlist:
            file_path = cmod_workflow.unprocessed_data_dir / f"shot_{shot}.nc"
            assert file_path.exists(), (
                f"Unprocessed data file for shot {shot} does not exist"
            )
