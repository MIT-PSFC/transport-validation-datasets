"""Method-agnostic batch containers and npz io for GP profile fitting.

One staged batch file serves every fitting method:
it carries the cleaned Thomson channel data in fit units: Te [keV], ne [1e20 m^-3]
(with the device-specific error floors already baked in at staging),
the target rho_tor_norm grid, and the per-variable fit bound knobs and anchors.
Workers read a batch, fit it, and write a result file whose rows stay aligned with the input rows
A slice that was skipped or culled is an all-NaN row, never a dropped one.

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

FIT_VARIABLES = ("te", "ne")

# How the staged rows were built. Workers do not care.
# Carried in the batch file so a resumed run never mixes them (see workflow.stage_fit_batches).
FIT_MODE_SAMPLE = "sample"  # one row per Thomson sample
FIT_MODE_WINDOW_SAMPLE = "window_sample"  # one row per Thomson sample in a time window
FIT_MODE_WINDOW_AVERAGE = "window_average"  # one row per time window, samples pooled

# Per-slice fit statuses, stored as int8 arrays in the result files.
STATUS_OK = 0  # clean fit
STATUS_REPAIRED = 1  # refit after a pin release or channel drop
STATUS_CULLED = 2  # still nonphysical after the repairs, arrays are NaN
STATUS_SKIPPED = 3  # fewer than min_points valid channels
STATUS_FAILED = 4  # the GP fit raised or returned nothing

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_REPAIRED: "repaired",
    STATUS_CULLED: "culled",
    STATUS_SKIPPED: "skipped",
    STATUS_FAILED: "failed",
}


@dataclass(frozen=True)
class FitBounds:
    """Per-variable fit bound knobs carried in the batch file.

    Their semantics are defined by the fitting method, other methods may ignore fields they do not use.
    One instance per fitted variable, so each variable of each device can run its own range.
    """

    l1_min: float = 0.4
    l1_max: float = 0.7
    var_min: float = 1.0e-2
    var_max: float = 20.0


def default_fit_bounds() -> dict[str, FitBounds]:
    """Build default bounds for every fitted variable.

    Returns:
        Default FitBounds keyed by variable name.
    """
    return {var: FitBounds() for var in FIT_VARIABLES}


@dataclass(frozen=True)
class FitAnchors:
    """Virtual observations one variable is fit with, carried in the batch file.

    Rows are in fit units (Te [keV], ne [1e20 m^-3]), set per device (see workflow.DeviceSettings).
    Every method adds them to every slice.

    Attributes:
        value: (n, 3) rows of (rho_tor_norm, value, error).
        grad: (n, 3) rows of (rho_tor_norm, d/drho_tor_norm, error), per unit rho_tor_norm.
    """

    value: np.ndarray
    grad: np.ndarray

    def scaled(self, scale: float) -> "FitAnchors":
        """Divide the values and errors by scale, keeping the positions.

        Args:
            scale: The slice normalization the channel data was divided by.

        Returns:
            The anchors in the normalized units.
        """
        factor = np.array([1.0, 1.0 / scale, 1.0 / scale])
        return FitAnchors(value=self.value * factor, grad=self.grad * factor)


@dataclass
class ShotFitInput:
    """Cleaned Thomson channel data for one shot, ready for GP fitting.

    All channel arrays are (n_t, n_ch).
    x is the radial coordinate of each channel (rho_tor_norm),
    shared between te and ne since both come from the same channels.
    Invalid points are NaN.

    windows is the shot's (n_w, 2) time window bounds [s] and window_index
    the (n_t,) window each row belongs to (see transport_validation_datasets.windows).
    A shot staged without windows has an empty windows array and -1 in every window_index.
    In a pooled row a channel appears once per Thomson sample, so n_ch is then samples x channels.
    """

    x: np.ndarray
    te_y: np.ndarray
    te_err: np.ndarray
    ne_y: np.ndarray
    ne_err: np.ndarray
    time: np.ndarray
    windows: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    window_index: np.ndarray | None = None

    def __post_init__(self):
        if self.window_index is None:
            self.window_index = np.full(np.asarray(self.time).shape, -1, dtype=np.int64)

    def has_fittable_points(self) -> bool:
        """Check that te and ne each have at least one finite (x, y, err) point.

        A shot failing this can only come back all NaN from the fit, so callers
        should record it as failed and skip it before staging.

        Returns:
            True if both variables have at least one fittable point.
        """
        x_ok = np.isfinite(self.x)
        te_ok = x_ok & np.isfinite(self.te_y) & np.isfinite(self.te_err)
        ne_ok = x_ok & np.isfinite(self.ne_y) & np.isfinite(self.ne_err)
        return bool(te_ok.any() and ne_ok.any())


@dataclass
class FitBatch:
    """Everything a worker needs: the in-memory image of one staged batch file.

    Attributes:
        shot_inputs: Per-shot channel data, keyed by shot number.
        x_star: (n_x,) target rho_tor_norm grid the profiles are fit on.
        min_points: Minimum valid channels per slice to attempt a fit.
        scale_per_slice: Normalize each slice by its max before fitting.
        bounds: Per-variable fit bound knobs, keyed by FIT_VARIABLES.
        anchors: Per-variable virtual observations, keyed by FIT_VARIABLES.
        pedestal_rho_tor_norm: Pedestal location for the methods whose model has one
            (zk's length-scale transition, akho's mtanh centre).
        sol_extension: How the staged positions continue outside the LCFS (machine.generic.SOL_EXTENSIONS).
            Workers never read it, it only keeps a resumed run from mixing extensions.
        fit_mode: One of the FIT_MODE_* values, how the rows were built.
    """

    shot_inputs: dict[int, ShotFitInput]
    x_star: np.ndarray
    min_points: int
    scale_per_slice: bool
    bounds: dict[str, FitBounds]
    anchors: dict[str, FitAnchors]
    pedestal_rho_tor_norm: float
    sol_extension: str
    fit_mode: str = FIT_MODE_SAMPLE


@dataclass
class ShotFitOutput:
    """GP-fitted profiles for one shot.

    The fit/std/grad/grad_std arrays are (n_t, n_x), row-aligned with the
    input slices. A slice that was not fit is an all-NaN row.
    The gradients are the GP posterior derivative d/drho_tor_norm (mean and latent std)
    in the profile's units per unit rho_tor_norm.
    The status arrays are (n_t,) int8 STATUS_* codes.
    The hyps arrays are optional method diagnostics ((n_t, n_hyp), NaN where a slice
    was not fit at optimized hyperparameters) workers that have none leave them as None.
    """

    te_fit: np.ndarray
    te_std: np.ndarray
    te_grad: np.ndarray
    te_grad_std: np.ndarray
    ne_fit: np.ndarray
    ne_std: np.ndarray
    ne_grad: np.ndarray
    ne_grad_std: np.ndarray
    te_status: np.ndarray
    ne_status: np.ndarray
    time: np.ndarray
    te_hyps: np.ndarray | None = None
    ne_hyps: np.ndarray | None = None

    @classmethod
    def empty(cls, n_t: int, n_x: int, time: np.ndarray) -> "ShotFitOutput":
        """Build an all-NaN output with every slice marked STATUS_SKIPPED.

        Args:
            n_t: Number of time slices.
            n_x: Number of rho_tor_norm grid points.
            time: (n_t,) slice times [s], echoed from the input.

        Returns:
            Output container ready to be filled slice by slice.
        """
        return cls(
            te_fit=np.full((n_t, n_x), np.nan),
            te_std=np.full((n_t, n_x), np.nan),
            te_grad=np.full((n_t, n_x), np.nan),
            te_grad_std=np.full((n_t, n_x), np.nan),
            ne_fit=np.full((n_t, n_x), np.nan),
            ne_std=np.full((n_t, n_x), np.nan),
            ne_grad=np.full((n_t, n_x), np.nan),
            ne_grad_std=np.full((n_t, n_x), np.nan),
            te_status=np.full(n_t, STATUS_SKIPPED, dtype=np.int8),
            ne_status=np.full(n_t, STATUS_SKIPPED, dtype=np.int8),
            time=np.asarray(time, dtype=np.float32),
        )


def pack_fit_batch(path: Path | str, batch: FitBatch):
    """Write a batch of shot fit inputs to a single npz file (atomically).

    Args:
        path: Destination npz path.
        batch: Batch to write.
    """
    arrays = {
        "shots": np.array(sorted(batch.shot_inputs), dtype=np.int64),
        "x_star": np.asarray(batch.x_star, dtype=np.float64),
        "min_points": np.int64(batch.min_points),
        "scale_per_slice": np.bool_(batch.scale_per_slice),
        "pedestal_rho_tor_norm": np.float64(batch.pedestal_rho_tor_norm),
        "fit_mode": np.str_(batch.fit_mode),
        "sol_extension": np.str_(batch.sol_extension),
    }
    for var in FIT_VARIABLES:
        for f in fields(FitBounds):
            arrays[f"bounds:{var}:{f.name}"] = np.float64(
                getattr(batch.bounds[var], f.name)
            )
        arrays[f"anchors:{var}:value"] = np.asarray(
            batch.anchors[var].value, dtype=np.float64
        )
        arrays[f"anchors:{var}:grad"] = np.asarray(
            batch.anchors[var].grad, dtype=np.float64
        )
    for shot, si in batch.shot_inputs.items():
        arrays[f"{shot}:time"] = np.asarray(si.time, dtype=np.float32)
        arrays[f"{shot}:x"] = np.asarray(si.x, dtype=np.float32)
        arrays[f"{shot}:te_y"] = np.asarray(si.te_y, dtype=np.float32)
        arrays[f"{shot}:te_err"] = np.asarray(si.te_err, dtype=np.float32)
        arrays[f"{shot}:ne_y"] = np.asarray(si.ne_y, dtype=np.float32)
        arrays[f"{shot}:ne_err"] = np.asarray(si.ne_err, dtype=np.float32)
        arrays[f"{shot}:windows"] = np.asarray(si.windows, dtype=np.float64)
        arrays[f"{shot}:window_index"] = np.asarray(si.window_index, dtype=np.int64)
    _atomic_savez(path, arrays)


def unpack_fit_batch(path: Path | str) -> FitBatch:
    """Read a batch input npz.

    Args:
        path: Batch input npz path.

    Returns:
        The staged batch. A bounds key absent from the file reads back that
        FitBounds field's default.
    """
    with np.load(path) as data:
        bounds = {
            var: FitBounds(
                **{
                    f.name: float(data[key])
                    for f in fields(FitBounds)
                    if (key := f"bounds:{var}:{f.name}") in data.files
                }
            )
            for var in FIT_VARIABLES
        }
        anchors = _unpack_anchors(data)
        shot_inputs = {
            shot: ShotFitInput(
                x=data[f"{shot}:x"],
                te_y=data[f"{shot}:te_y"],
                te_err=data[f"{shot}:te_err"],
                ne_y=data[f"{shot}:ne_y"],
                ne_err=data[f"{shot}:ne_err"],
                time=data[f"{shot}:time"],
                windows=data[f"{shot}:windows"],
                window_index=data[f"{shot}:window_index"],
            )
            for shot in data["shots"].tolist()
        }
        return FitBatch(
            shot_inputs=shot_inputs,
            x_star=data["x_star"],
            min_points=int(data["min_points"]),
            scale_per_slice=bool(data["scale_per_slice"]),
            bounds=bounds,
            anchors=anchors,
            pedestal_rho_tor_norm=float(data["pedestal_rho_tor_norm"]),
            sol_extension=str(data["sol_extension"].item()),
            fit_mode=str(data["fit_mode"].item()),
        )


def _unpack_anchors(data) -> dict[str, FitAnchors]:
    """Read the per-variable anchors out of an open batch npz.

    Args:
        data: The open npz file.

    Returns:
        FitAnchors keyed by variable name.
    """
    return {
        var: FitAnchors(
            value=data[f"anchors:{var}:value"], grad=data[f"anchors:{var}:grad"]
        )
        for var in FIT_VARIABLES
    }


def read_batch_anchors(path: Path | str) -> dict[str, FitAnchors]:
    """Read only the anchors from a batch input npz (cheap).

    Args:
        path: Batch input npz path.

    Returns:
        FitAnchors keyed by variable name.
    """
    with np.load(path) as data:
        return _unpack_anchors(data)


def read_batch_shots(path: Path | str) -> list[int]:
    """Read only the shot list from a batch input npz (cheap).

    Args:
        path: Batch input npz path.

    Returns:
        Shot numbers in the batch.
    """
    with np.load(path) as data:
        return data["shots"].tolist()


def read_batch_windows(path: Path | str) -> dict[int, np.ndarray]:
    """Read only the time windows each shot of a batch input npz was staged with.

    Cheap: the channel arrays are never touched.

    Args:
        path: Batch input npz path.

    Returns:
        Per shot its (n_w, 2) window bounds [s], empty for a shot staged
        without windows.
    """
    with np.load(path) as data:
        return {shot: data[f"{shot}:windows"] for shot in data["shots"].tolist()}


def read_batch_setting(path: Path | str, name: str) -> str:
    """Read only one scalar setting from a batch input npz (cheap).

    Args:
        path: Batch input npz path.
        name: The FitBatch field, "fit_mode", "sol_extension" or "pedestal_rho_tor_norm".

    Returns:
        The setting's value, as a string.
    """
    with np.load(path) as data:
        return str(data[name].item())


def pack_fit_results(
    path: Path | str, outputs: dict[int, ShotFitOutput], x_star: np.ndarray
):
    """Write fitted profiles to a single npz file (atomically).

    Args:
        path: Destination npz path.
        outputs: Per-shot fit outputs, keyed by shot number.
        x_star: (n_x,) rho_tor_norm grid the profiles were fit on.
    """
    arrays = {
        "shots": np.array(sorted(outputs), dtype=np.int64),
        "x_star": np.asarray(x_star, dtype=np.float64),
    }
    for shot, so in outputs.items():
        arrays[f"{shot}:time"] = np.asarray(so.time, dtype=np.float32)
        for var in FIT_VARIABLES:
            for name in ("fit", "std", "grad", "grad_std"):
                arrays[f"{shot}:{var}_{name}"] = np.asarray(
                    getattr(so, f"{var}_{name}"), dtype=np.float32
                )
            arrays[f"{shot}:{var}_status"] = np.asarray(
                getattr(so, f"{var}_status"), dtype=np.int8
            )
            hyps = getattr(so, f"{var}_hyps")
            if hyps is not None:
                arrays[f"{shot}:{var}_hyps"] = np.asarray(hyps, dtype=np.float32)
    _atomic_savez(path, arrays)


def unpack_fit_results(path: Path | str) -> dict[int, ShotFitOutput]:
    """Read a batch result npz into per-shot fit outputs.

    Args:
        path: Batch result npz path.

    Returns:
        Per-shot fit outputs, keyed by shot number.
        Absent hyps keys read back as None.
    """
    with np.load(path) as data:
        outputs = {}
        for shot in data["shots"].tolist():
            kwargs = {"time": data[f"{shot}:time"]}
            for var in FIT_VARIABLES:
                for name in ("fit", "std", "grad", "grad_std", "status"):
                    kwargs[f"{var}_{name}"] = data[f"{shot}:{var}_{name}"]
                hyps_key = f"{shot}:{var}_hyps"
                kwargs[f"{var}_hyps"] = (
                    data[hyps_key] if hyps_key in data.files else None
                )
            outputs[shot] = ShotFitOutput(**kwargs)
        return outputs


def _atomic_savez(path: Path | str, arrays: dict):
    """Write npz to a temp file then rename, so readers never see partial files.

    Args:
        path: Destination npz path.
        arrays: Arrays to save, keyed by npz key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)
