#!/bin/bash
# Script used to make 20-shot sample dataset for DIII-D, up to the internal store (no release permission)
# NOTE: Runs on omega, with the DIII-D data servers and the /fusion IDA databases.
# The ida fit method carries IDA's fits onto the fit grid locally, no cluster needed.

MAX_NUM_SHOTS=20

uv run python -m transport_validation_datasets.cli d3d \
    /cscratch/$USER/tvd_builds \
    --ds_name d3d_sample_$MAX_NUM_SHOTS \
    --max_num_shots $MAX_NUM_SHOTS \
    --stage all
