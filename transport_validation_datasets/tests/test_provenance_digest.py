import numpy as np
import xarray as xr

from transport_validation_datasets.provenance import dataset_digest


def _inputs():
    t_e = np.arange(6.0).reshape(2, 3)
    return xr.Dataset(
        {"t_e": (("time", "rho"), t_e)},
        coords={"time": [0.1, 0.2], "channel_name": ("rho", ["a", "b", "c"])},
        attrs={"fit_stamp": "provenance of one run"},
    )


class TestDatasetDigest:
    def test_not_changed_by_attrs(self):
        ds_inputs = _inputs()
        ds_rewritten = ds_inputs.assign_attrs(fit_stamp="provenance of another run")
        digest_rewritten = dataset_digest(ds_rewritten)
        digest_inputs = dataset_digest(ds_inputs)
        assert digest_rewritten == digest_inputs

    def test_changes_with_one_value(self):
        ds_changed = _inputs()
        ds_changed["t_e"][1, 2] = 5.5
        digest_changed = dataset_digest(ds_changed)
        ds_inputs = _inputs()
        digest_inputs = dataset_digest(ds_inputs)
        assert digest_changed != digest_inputs

    def test_strings_hashed_by_value(self, tmp_path):
        # Read back from netCDF the strings are objects, whose bytes would be pointers
        nc_path = tmp_path / "inputs.nc"
        ds_inputs = _inputs()
        ds_inputs.to_netcdf(nc_path)
        with (
            xr.open_dataset(nc_path) as ds_first,
            xr.open_dataset(nc_path) as ds_second,
        ):
            digest_first = dataset_digest(ds_first)
            digest_second = dataset_digest(ds_second)
        assert digest_first == digest_second
