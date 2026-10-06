"""Shared scaffolding for fit workers: the CLI runner and the slice fan-out.

Every worker module (worker_zk.py, worker_akho.py, ...) provides two
module-level entry points:

    fit_batch(batch: FitBatch, num_workers: int = 1) -> dict[int, ShotFitOutput]
    main(argv: list[str] | None = None)

main is the cluster CLI that run_worker_cli implements once for all workers.
(`python -m ...worker_x input.npz output.npz --num-workers N`)
map_slices implements the per-(shot, time slice) fan-out for workers whose
method fits slices independently. A worker with a different structure can
ignore it and implement fit_batch directly.

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
(diagnostics use print instead of loguru)
"""

import argparse
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_CULLED,
    STATUS_REPAIRED,
    FitAnchors,
    FitBatch,
    FitBounds,
    ShotFitOutput,
    pack_fit_results,
    unpack_fit_batch,
)


@dataclass(frozen=True)
class SliceTask:
    """One (shot, time slice) fit task, self-contained for multiprocessing.

    All channel arrays are (n_ch,) rows of the shot's input. Frozen and free of
    closures and module-global state so it pickles cleanly into a Pool and
    survives any multiprocessing start method.
    """

    shot: int
    i_time: int
    x: np.ndarray
    te_y: np.ndarray
    te_err: np.ndarray
    ne_y: np.ndarray
    ne_err: np.ndarray
    x_star: np.ndarray
    min_points: int
    scale_per_slice: bool
    te_bounds: FitBounds
    ne_bounds: FitBounds
    te_anchors: FitAnchors
    ne_anchors: FitAnchors
    pedestal_rho_tor_norm: float


@dataclass
class VariableFit:
    """Fit of one variable of one slice; arrays are None when it was not fit.

    Attributes:
        fit: (n_x,) fitted profile.
        std: (n_x,) predictive std.
        grad: (n_x,) posterior derivative d/drho.
        grad_std: (n_x,) latent derivative std.
        hyps: Method-specific hyperparameter diagnostics, or None.
        status: STATUS_* code for this slice.
    """

    fit: np.ndarray | None
    std: np.ndarray | None
    grad: np.ndarray | None
    grad_std: np.ndarray | None
    hyps: np.ndarray | None
    status: int


@dataclass
class SliceResult:
    """Te and ne fits of one (shot, time slice)."""

    shot: int
    i_time: int
    te: VariableFit
    ne: VariableFit


def map_slices(
    fit_slice: Callable[[SliceTask], SliceResult],
    batch: FitBatch,
    num_workers: int = 1,
    max_slices_per_shot: int | None = None,
) -> dict[int, ShotFitOutput]:
    """Run a per-slice fit function over every (shot, time slice) in the batch.

    Slices run serially (num_workers <= 1) or across a multiprocessing Pool.
    fit_slice must be a top-level module function so it pickles into the Pool.

    Args:
        fit_slice: Fits one SliceTask and returns its SliceResult.
        batch: The staged batch to fit.
        num_workers: Slice-level worker processes; <= 1 fits serially in this
            process.
        max_slices_per_shot: If set, only fit the first N time slices of each
            shot (debug aid); the remaining rows stay STATUS_SKIPPED.

    Returns:
        Per-shot outputs, keyed by shot number, row-aligned with the inputs.
    """
    x_star = np.asarray(batch.x_star, dtype=float)
    tasks = []
    for shot, si in batch.shot_inputs.items():
        n_t = si.te_y.shape[0]
        if max_slices_per_shot is not None:
            n_t = min(n_t, max_slices_per_shot)
        tasks.extend(
            SliceTask(
                shot=shot,
                i_time=i_time,
                x=si.x[i_time, :],
                te_y=si.te_y[i_time, :],
                te_err=si.te_err[i_time, :],
                ne_y=si.ne_y[i_time, :],
                ne_err=si.ne_err[i_time, :],
                x_star=x_star,
                min_points=batch.min_points,
                scale_per_slice=batch.scale_per_slice,
                te_bounds=batch.bounds["te"],
                ne_bounds=batch.bounds["ne"],
                te_anchors=batch.anchors["te"],
                ne_anchors=batch.anchors["ne"],
                pedestal_rho_tor_norm=batch.pedestal_rho_tor_norm,
            )
            for i_time in range(n_t)
        )

    outputs = {
        shot: ShotFitOutput.empty(si.te_y.shape[0], x_star.size, si.time)
        for shot, si in batch.shot_inputs.items()
    }

    def _store(result: SliceResult):
        so = outputs[result.shot]
        for var, vf in (("te", result.te), ("ne", result.ne)):
            getattr(so, f"{var}_status")[result.i_time] = vf.status
            if vf.fit is not None:
                getattr(so, f"{var}_fit")[result.i_time, :] = vf.fit
                getattr(so, f"{var}_std")[result.i_time, :] = vf.std
                getattr(so, f"{var}_grad")[result.i_time, :] = vf.grad
                getattr(so, f"{var}_grad_std")[result.i_time, :] = vf.grad_std
            if vf.hyps is not None:
                if getattr(so, f"{var}_hyps") is None:
                    setattr(
                        so,
                        f"{var}_hyps",
                        np.full((so.te_fit.shape[0], vf.hyps.size), np.nan),
                    )
                getattr(so, f"{var}_hyps")[result.i_time, :] = vf.hyps

    n_total = len(tasks)
    n_done = 0
    t_start = time.monotonic()
    if num_workers <= 1:
        for task in tasks:
            _store(fit_slice(task))
            n_done += 1
            if n_done % 10 == 0:
                print(f"[fit_worker] {n_done}/{n_total} slices done", flush=True)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(fit_slice, tasks, chunksize=1):
                _store(result)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[fit_worker] {n_done}/{n_total} slices done", flush=True)

    elapsed = time.monotonic() - t_start
    print(
        f"[fit_worker] finished {n_total} slices (Te+ne) for "
        f"{len(batch.shot_inputs)} shots in {elapsed:.0f}s "
        f"({elapsed / max(n_total, 1):.2f}s per slice)",
        flush=True,
    )
    return outputs


def _print_repair_tallies(outputs: dict[int, ShotFitOutput]):
    """Print per-shot repaired/culled slice counts from the status arrays.

    Repaired slices were refit after a pin release or a channel drop; culled
    slices stayed nonphysical and were dropped. Logged so systematic problems
    (a bad edge channel wrecking a whole shot) are visible in the job log.

    Args:
        outputs: Per-shot fit outputs.
    """
    for shot in sorted(outputs):
        so = outputs[shot]
        counts = {
            var: (
                int((getattr(so, f"{var}_status") == STATUS_REPAIRED).sum()),
                int((getattr(so, f"{var}_status") == STATUS_CULLED).sum()),
            )
            for var in ("te", "ne")
        }
        if any(n for pair in counts.values() for n in pair):
            print(
                f"[fit_worker] shot {shot} nonphysical fits: "
                f"Te repaired {counts['te'][0]} culled {counts['te'][1]}, "
                f"ne repaired {counts['ne'][0]} culled {counts['ne'][1]}",
                flush=True,
            )


def run_worker_cli(
    fit_batch_fn: Callable[..., dict[int, ShotFitOutput]],
    prog: str,
    argv: list[str] | None = None,
):
    """Run a worker's CLI: read a batch npz, fit it, write the result npz.

    Args:
        fit_batch_fn: The worker's fit_batch(batch, num_workers=N) function.
        prog: Program name shown in the argparse help.
        argv: Command-line arguments; None uses sys.argv.
    """
    parser = argparse.ArgumentParser(prog=prog, description="GP profile fit worker")
    parser.add_argument("input", help="Path to batch input npz")
    parser.add_argument("output", help="Path to write batch output npz")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Slice-level worker processes. Defaults to SLURM_CPUS_PER_TASK "
        "or cpu count.",
    )
    args = parser.parse_args(argv)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    batch = unpack_fit_batch(args.input)
    print(
        f"[fit_worker] fitting {len(batch.shot_inputs)} shots with "
        f"{num_workers} workers (min_points={batch.min_points}, "
        f"scale_per_slice={batch.scale_per_slice}, bounds={batch.bounds})",
        flush=True,
    )
    outputs = fit_batch_fn(batch, num_workers=num_workers)
    pack_fit_results(args.output, outputs, batch.x_star)
    _print_repair_tallies(outputs)
    print(f"[fit_worker] wrote results to {args.output}", flush=True)
