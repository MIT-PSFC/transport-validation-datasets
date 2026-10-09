"""Unit tests for the C-Mod retrieval helpers that need no MDSplus.

A namespace with a shot number and a logger stands in for disruption-py's physics method parameters.
"""

from types import SimpleNamespace

import numpy as np
from loguru import logger

from transport_validation_datasets.machine.cmod.dispy_methods import (
    CmodThomsonMethods,
)


class TestAlignRegion:
    def test_edge_samples_land_on_their_partner_core_pulse(self):
        # Pulses every 16.7 ms, the edge reading each 20 us after the core.
        # The core skips the 50 ms pulse and the edge the 66.7 ms one,
        # so the edge sample at 50 ms has no partner, and 66.7 ms is the nearest core time to it.
        core_time = np.array([0.0, 0.0167, 0.0333, 0.0667, 0.0833])
        edge_time = np.array([0.0, 0.0167, 0.0333, 0.05, 0.0833]) + 20e-6
        edge_te = np.tile(np.arange(edge_time.size, dtype=float)[:, None], (1, 2))
        region = {
            "z": np.array([0.1, 0.2]),
            "time": edge_time,
            "te": edge_te,
            "te_error": edge_te,
            "ne": edge_te,
            "ne_error": edge_te,
        }
        params = SimpleNamespace(shot_id=1160909025, logger=logger)

        aligned = CmodThomsonMethods._align_region(params, region, core_time)

        np.testing.assert_array_equal(aligned["time"], core_time)
        # The 66.7 ms pulse stays empty, never handed the unpartnered 50 ms sample
        np.testing.assert_array_equal(aligned["te"][:, 0], [0.0, 1.0, 2.0, np.nan, 4.0])
