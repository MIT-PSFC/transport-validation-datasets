"""Writes one equilibrium reconstruction time to a real `.geqdsk` file.

Simple by design: the GEQDSK block staged in a shot's unprocessed file
(`machine.generic.make_geqdsk_dataset`'s schema, FreeQDSK canonical names,
see `machine.generic.DATASET_EQUILIBRIUM_SIGNALS`) already matches
`freeqdsk.geqdsk`'s own field names almost one-to-one.
So writing is a straight field copy plus NaN-stripping for the boundary and limiter contours,
whose point counts can vary slice to slice and are padded to a common width with NaN.
The fields are copied untouched, in the convention they were extracted in, which is NOT one fixed COCOS.
The unprocessed file records it as its cocos attribute (see `machine.generic.cocos_from_signs`).
COCOS conversion happens downstream, when `scenario_export.py` reads the file back.

`scenario_export.py`'s equilibrium builder needs a real file on disk because
the `eqdsk` package it uses for COCOS conversion reads one, so this writer
runs once per equilibrium time before that package is handed the path.
"""

from pathlib import Path
from typing import Mapping

import numpy as np
from freeqdsk import geqdsk

# GEQDSK block field -> freeqdsk.geqdsk field.
# Only `current` differs from make_geqdsk_dataset's own naming (freeqdsk calls it `cpasma`),
# and `psirz` maps to freeqdsk's `psi`.
_SCALAR_FIELDS = (
    "rdim",
    "zdim",
    "rcentr",
    "rleft",
    "zmid",
    "rmagx",
    "zmagx",
    "simagx",
    "sibdry",
    "bcentr",
)
_PROFILE_FIELDS = ("fpol", "pres", "ffprime", "pprime", "qpsi")


def write_geqdsk(
    path: Path | str,
    eq_slice: Mapping,
    shot: int = 0,
    time_ms: int = 0,
    label: str | None = None,
) -> Path:
    """Write one equilibrium reconstruction time to a `.geqdsk` file.

    Args:
        path: Destination `.geqdsk` path, parent directories are created.
        eq_slice: One time slice of the GEQDSK block
            (an `xr.Dataset.isel(idx=i)`, or any mapping with the same field names),
            `machine.generic.make_geqdsk_dataset`'s schema.
        shot: Shot number recorded in the file header.
        time_ms: Reconstruction time in milliseconds, recorded in the file header.
        label: Header label, `freeqdsk` defaults to `"FREEGS"` if omitted.

    Returns:
        The written path.
    """

    def _get(name: str) -> np.ndarray:
        return np.asarray(eq_slice[name], dtype=float)

    def _finite_contour(r_name: str, z_name: str) -> tuple[np.ndarray, np.ndarray]:
        r, z = _get(r_name), _get(z_name)
        finite = np.isfinite(r) & np.isfinite(z)
        return r[finite], z[finite]

    data = {name: float(eq_slice[name]) for name in _SCALAR_FIELDS}
    data["cpasma"] = float(eq_slice["current"])
    for name in _PROFILE_FIELDS:
        data[name] = _get(name)
    data["psi"] = _get("psirz")

    rbdry, zbdry = _finite_contour("rbdry", "zbdry")
    if rbdry.size > 0:
        data["rbdry"], data["zbdry"] = rbdry, zbdry
    if "rlim" in eq_slice and "zlim" in eq_slice:
        rlim, zlim = _finite_contour("rlim", "zlim")
        if rlim.size > 0:
            data["rlim"], data["zlim"] = rlim, zlim

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        geqdsk.write(data, fh, label=label, shot=int(shot), time=int(time_ms))
    return path
