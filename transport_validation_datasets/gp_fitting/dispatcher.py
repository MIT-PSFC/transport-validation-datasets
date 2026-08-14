"""Dispatch GP profile fitting batches to a SLURM cluster.

The dispatcher uploads batch input files plus the standalone fit_worker.py,
(one npz per batch of shots to avoid many-small-file transfers on clusters like Engaging)
submits one CPU job per batch, polls until completion, and pulls the result files back.

Two backends:
- "ssh": submit to a remote cluster through an srunx SSH profile (set up once
  with `srunx ssh profile add <name> --ssh-host <host>`). Used for C-Mod,
  where the cluster has no access to the source data. Both the file transfers
  and the SLURM commands run over subprocess ssh, using the options from
  ~/.ssh/config - see _ShellJobControl for why srunx's own SLURM client is
  not used.
- "local": running on the cluster itself (e.g. MAST fitting on Engaging);
  files are copied on the shared filesystem and sbatch runs locally.

Jobs get deterministic names gpfit-{device}-{batch_id}-a{attempt}
(batch_id is a hash of the shot list),
so a restarted workflow finds in-flight jobs instead of resubmitting them.
Killed jobs (TIMEOUT, PREEMPTED, OOM, ...) are retried up to max_retries times,
and both retries and jobs stuck PENDING past pending_timeout_s move to
the next partition in the preference list (wrapping around).
"""

from dataclasses import dataclass


@dataclass
class PartitionSpec:
    """One partition to try, with its own time limit and optional constraint.

    Attributes:
        name: Name of the SLURM partition.
        time_limit: Wall time limit for jobs on this partition. Must not exceed the
            partition's MaxTime; query it with
            `scontrol show partition <name> | grep MaxTime`
            or `sinfo -p <name> -O partitionname,time`.
        constraint: Optional SLURM node constraint for this partition.
    """

    name: str
    time_limit: str
    constraint: str | None = None


def parse_partition_specs(value) -> list["PartitionSpec"]:
    """Parse a partition spec string into PartitionSpecs.

    Args:
        value: Comma-separated entries of name@time_limit or
            name@time_limit@constraint, e.g.
            "sched_psfc_mit_r8@8:00:00,mit_preemptable@8:00:00@rocky8".
            Also accepts a tuple/list of entry strings (Python Fire may
            pre-split comma-separated arguments).

    Returns:
        One PartitionSpec per entry.

    Raises:
        ValueError: If an entry does not match the format, or the spec is empty.
    """
    if isinstance(value, str):
        entries = value.split(",")
    else:
        entries = list(value)
    specs = []
    for entry in entries:
        fields = entry.strip().split("@")
        if len(fields) == 2:
            specs.append(PartitionSpec(name=fields[0], time_limit=fields[1]))
        elif len(fields) == 3:
            specs.append(
                PartitionSpec(
                    name=fields[0], time_limit=fields[1], constraint=fields[2]
                )
            )
        else:
            raise ValueError(
                f"Bad partition spec '{entry}': expected name@time_limit or name@time_limit@constraint"
            )
    if not specs:
        raise ValueError("Empty partition spec")
    return specs


@dataclass
class ClusterFitConfig:
    """Launch options for cluster-based GP fitting.

    Attributes:
        profile: srunx SSH profile name, or "local" when already running on the
            target cluster (shared filesystem, local sbatch).
        partitions: Ordered partition preference list, each with its own time limit
            and optional constraint (see parse_partition_specs for the string
            format). mkgp is CPU-only, so these should be CPU partitions. A batch is
            submitted to the first partition; it falls back to the next (wrapping
            around) when its job is killed or sits PENDING longer than
            pending_timeout_s.
        remote_workdir: Scratch directory on the cluster where batch files, the
            worker script, and job logs are placed. Cluster-specific.
        venv_path: Path to a pre-built venv on the cluster (see bootstrap_remote.sh).
        max_concurrent_jobs: Cap on simultaneously queued/running fitting jobs, to be
            a good cluster citizen.
        shots_per_batch: Shots packed into one npz / one job. At ~40 core-minutes per
            shot, 10 shots on 32 CPUs is ~15 minutes wall time. Small batches keep
            jobs running concurrently and bound the work lost to a killed job.
        cpus_per_job: cpus-per-task for each fitting job; the worker runs this many
            slice-fit processes.
        max_retries: Resubmissions allowed per batch after a terminal job failure
            (TIMEOUT, PREEMPTED, OOM, ...). Each retry moves to the next partition in
            the list, wrapping around.
        pending_timeout_s: Cancel a PENDING job and resubmit it on the next partition
            after this long in the queue. Stops once every partition has been tried.
    """

    profile: str
    partitions: list[PartitionSpec] | str | tuple
    remote_workdir: str
    venv_path: str
    max_concurrent_jobs: int = 8
    shots_per_batch: int = 10
    cpus_per_job: int = 32
    memory_per_node: str | None = None
    poll_interval_s: float = 60.0
    job_name_prefix: str = "gpfit"
    max_retries: int = 2
    pending_timeout_s: float = 1800.0

    def __post_init__(self):
        if not isinstance(self.partitions, list) or not all(
            isinstance(p, PartitionSpec) for p in self.partitions
        ):
            self.partitions = parse_partition_specs(self.partitions)
