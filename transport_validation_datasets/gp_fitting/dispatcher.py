"""Dispatch GP profile fitting batches to a SLURM cluster.

The dispatcher uploads staged batch input files (one npz per batch of shots,
to avoid many-small-file transfers on clusters like Engaging) plus the
gp_fitting worker package, submits one CPU job per batch, polls until
completion, and pulls the result files back.
Staging and result collection are the workflow's job (see DataWorkflow.run_gp_fitting),
the dispatcher only ships, executes, and pulls.

The single backend runs everything over a host alias from ~/.ssh/config
(ClusterFitConfig.ssh_host).
Used for C-Mod, where the cluster has no access to the source data.
Both the file transfers (subprocess rsync) and the SLURM commands run over
the OpenSSH client, using the options from ~/.ssh/config - see
_ShellJobControl for why no Python SSH library is involved. An
authenticated ControlMaster session to the login node must already exist
(2FA): open one with a plain `ssh <host>` before dispatching. If fitting ever
runs on the cluster itself again (shared filesystem, local sbatch - the old
MAST-on-Engaging case), reintroduce a local backend implementing the same
push_file/pull_file/ensure_dir/remove_glob/submit_script/queue surface with
shutil and subprocess sbatch.

Jobs get deterministic names gpfit-{ds_name}-{method}-{batch_id}-a{attempt}
(batch_id is a hash of the shot list), so a restarted workflow finds in-flight
jobs instead of resubmitting them, and two methods' runs never adopt each
other's jobs. Killed jobs (TIMEOUT, PREEMPTED, OOM, ...) are retried up to
max_retries times, and both retries and jobs stuck PENDING past
pending_timeout_s move to the next partition in the preference list (wrapping
around).
"""

import hashlib
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from transport_validation_datasets.gp_fitting import registry
from transport_validation_datasets.gp_fitting.batch_io import read_batch_shots

# rsync transfers go through the cluster login node, which sometimes drops
# connections (exit 255), so transfers are retried before giving up
_TRANSFER_ATTEMPTS = 3
_TRANSFER_RETRY_DELAY_S = 10.0
# rsync exit codes that mean the source file does not exist (not a
# connection problem), so retrying the transfer is pointless
_RSYNC_SOURCE_MISSING_CODES = {23, 24}
# polls to wait for a COMPLETED job's output to become pullable before
# declaring the batch failed
_MAX_OUTPUT_PULL_POLLS = 3
# polls to tolerate a job in an unrecognized/unknown state (e.g. it vanished
# from the queue) before treating the attempt as failed
_MAX_UNKNOWN_POLLS = 5
# how long clean() waits for cancelled jobs to actually leave the queue before
# it deletes their batch files (see _wait_for_jobs_to_drain)
CLEAN_DRAIN_TIMEOUT_S = 120.0
CLEAN_DRAIN_POLL_S = 5.0

# SLURM states that mean the job will never produce output
_TERMINAL_FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


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
        ssh_host: Host alias from ~/.ssh/config for the cluster login node,
            with ControlMaster/ControlPath configured. An authenticated
            session must already exist before dispatching (open one with a
            plain `ssh <host>`); the login node's 2FA prompt cannot be
            answered from here.
        partitions: Ordered partition preference list, each with its own time limit
            and optional constraint (see parse_partition_specs for the string
            format). mkgp is CPU-only, so these should be CPU partitions. A batch is
            submitted to the first partition; it falls back to the next (wrapping
            around) when its job is killed or sits PENDING longer than
            pending_timeout_s.
        remote_workdir: Scratch directory on the cluster where batch files, the
            worker package, and job logs are placed. Cluster-specific.
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

    ssh_host: str
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


def make_batch_id(ds_name: str, shots: list[int]) -> str:
    """Compute a deterministic short id for a batch, stable across restarts.

    Args:
        ds_name: Dataset name the batch belongs to.
        shots: Shot numbers in the batch.

    Returns:
        Ten-character hex id.
    """
    digest = hashlib.sha1(
        f"{ds_name}:{','.join(str(s) for s in sorted(shots))}".encode()
    )
    return digest.hexdigest()[:10]


def plan_batches(
    ds_name: str,
    pending_shots: list[int],
    batches_dir: Path,
    shots_per_batch: int,
) -> dict[str, list[int]]:
    """Assign pending shots to batches, reusing batch files from earlier runs.

    Existing batch input npz files in batches_dir keep their membership,
    and therefore their batch id and job name,
    so a restarted workflow lines up with jobs already in the cluster queue.
    Shots not covered by an existing batch are chunked into new batches of shots_per_batch.

    Args:
        ds_name: Dataset name, part of new batch ids.
        pending_shots: Shots that need fitting.
        batches_dir: Local directory holding batch npz files.
        shots_per_batch: Shots per newly planned batch.

    Returns:
        Mapping of batch id to the shots from pending_shots in that batch.
    """
    pending = set(pending_shots)
    batches: dict[str, list[int]] = {}

    for batch_path in sorted(batches_dir.glob("batch_*.npz")):
        if "_out_" in batch_path.name:
            continue
        try:
            batch_shots = read_batch_shots(batch_path)
        except Exception as e:
            logger.warning(f"Could not read existing batch file {batch_path}: {e}")
            continue
        batch_id = batch_path.stem.removeprefix("batch_")
        claimed = [s for s in batch_shots if s in pending]
        if claimed:
            batches[batch_id] = claimed
            pending -= set(claimed)

    remaining = sorted(pending)
    for i in range(0, len(remaining), shots_per_batch):
        chunk = remaining[i : i + shots_per_batch]
        batches[make_batch_id(ds_name, chunk)] = chunk

    return batches


def _ssh_config_user(dest: str) -> str:
    """Resolve the username ssh would use for dest, from ~/.ssh/config.

    The User line usually lives in ~/.ssh/config rather than in the host
    string, and without a username the squeue calls would scan every user's
    jobs. `ssh -G` resolves the config the same way the real connection
    does, without opening one.

    Args:
        dest: ssh destination (host or user@host).

    Returns:
        The resolved username, or "" if none was found.
    """
    if "@" in dest:
        return dest.split("@", 1)[0]
    result = subprocess.run(
        ["ssh", "-G", dest], capture_output=True, text=True, check=False
    )
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "user":
            return value.strip()
    return ""


class _ShellJobControl:
    """Queue inspection, cancellation and submission over subprocess ssh.

    Everything goes through the OpenSSH client, never a Python SSH library:
    paramiko and friends open a fresh transport per connection, which cannot
    get through a login node that requires publickey AND keyboard-interactive
    (2FA) - pubkey passes, the 2FA prompt goes unanswered, and the
    connection dies. The OpenSSH client instead reuses the authenticated
    ControlMaster socket from ~/.ssh/config, so one interactive `ssh <host>`
    beforehand carries the whole dispatch without further 2FA rounds.
    BatchMode=yes makes a dead master fail fast instead of hanging on the
    2FA prompt.

    Job state resolution: squeue for active jobs, sacct for jobs that have
    left the queue, scontrol as the last resort when slurmdbd is unreachable.
    Jobs found in none of the three are omitted, which the caller reads as
    UNKNOWN.
    """

    def _ssh(self, cmd: str, *, stdin: str | None = None):
        """Run a command on the cluster over the OpenSSH client.

        Args:
            cmd: Shell command to run remotely.
            stdin: Optional text piped to the command's stdin.

        Returns:
            The completed subprocess result.
        """
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self._host, cmd],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    def submit_script(self, script: str, job_name: str) -> int:
        """Write a job script on the cluster and sbatch it.

        The script is piped in and written remotely rather than passed as an
        argument: it is multi-line shell, and a copy on the cluster is what
        you need to debug a job that misbehaved.

        Args:
            script: Full sbatch script text.
            job_name: Job name; also names the remote script file.

        Returns:
            The submitted SLURM job id.

        Raises:
            RuntimeError: If sbatch fails.
        """
        remote_script = f"{self._workdir}/{job_name}.sh"
        quoted = shlex.quote(remote_script)
        result = self._ssh(
            f"cat > {quoted} && sbatch --parsable {quoted}", stdin=script
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed for {job_name}: {result.stderr.strip()}")
        return int(result.stdout.strip().splitlines()[-1].split(";")[0])

    def queued_jobs(self) -> list[tuple[str, int]]:
        """List (name, job id) for every queued/running job of this user.

        A list, not a name-keyed dict: two jobs can carry the same name (two
        runs each submitting attempt 1 of the same batch), and clean() has to
        cancel both.

        Returns:
            (job name, job id) pairs.

        Raises:
            RuntimeError: If squeue fails.
        """
        user_arg = f"-u {shlex.quote(self._username)} " if self._username else ""
        result = self._ssh(f"squeue {user_arg}-h -o {shlex.quote('%i|%j')}")
        if result.returncode != 0:
            raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
        jobs = []
        for line in result.stdout.splitlines():
            jid, _, name = line.strip().partition("|")
            if jid.isdigit() and name:
                jobs.append((name, int(jid)))
        return jobs

    def queued_job_names(self) -> dict[str, int]:
        """Map this user's queued/running job names to job ids.

        Returns:
            Job name to job id mapping.
        """
        return dict(self.queued_jobs())

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        """Resolve the SLURM state of each job id.

        Args:
            job_ids: Job ids to look up.

        Returns:
            Job id to state mapping; ids resolvable by none of squeue, sacct,
            or scontrol are omitted (callers read that as UNKNOWN).
        """
        if not job_ids:
            return {}
        id_arg = ",".join(str(i) for i in job_ids)
        states: dict[int, str] = {}

        result = self._ssh(
            f"squeue --jobs {id_arg} -h -o {shlex.quote('%i|%T')} 2>/dev/null || true"
        )
        for line in result.stdout.splitlines():
            jid, _, st = line.strip().partition("|")
            if jid.isdigit() and st:
                states[int(jid)] = st

        missing = [j for j in job_ids if j not in states]
        if missing:
            # sacct reports per-step rows (12345.batch, 12345.extern) and
            # decorates some states ("CANCELLED by 1234"); keep the job row
            # and the bare state so it matches _TERMINAL_FAILURE_STATES.
            sacct_ids = ",".join(str(i) for i in missing)
            result = self._ssh(
                f"sacct -j {sacct_ids} -n -P -o JobID,State 2>/dev/null || true"
            )
            for line in result.stdout.splitlines():
                jid, _, st = line.strip().partition("|")
                if jid.isdigit() and st:
                    states[int(jid)] = st.split()[0]

        missing = [j for j in job_ids if j not in states]
        for jid in missing:
            result = self._ssh(
                f"scontrol show job {jid} 2>/dev/null | tr ' ' '\\n' | grep '^JobState=' || true"
            )
            st = result.stdout.strip().partition("=")[2]
            if st:
                states[jid] = st
        return states

    def cancel(self, job_id: int):
        """Cancel one job.

        Args:
            job_id: SLURM job id to cancel.

        Raises:
            RuntimeError: If scancel fails.
        """
        result = self._ssh(f"scancel {int(job_id)}")
        if result.returncode != 0:
            raise RuntimeError(f"scancel {job_id} failed: {result.stderr.strip()}")


class _SSHBackend(_ShellJobControl):
    """File transfer and job control on a remote cluster over ssh.

    File transfers run over subprocess rsync with the OpenSSH client as the
    transport, so they ride the same ControlMaster session as the SLURM
    commands. No --mkpath: remote rsync may be too old for it (e.g. Engaging
    has 3.1.3), so callers ensure_dir before pushing.
    """

    def __init__(self, config: ClusterFitConfig):
        """Remember the target host and resolve the ssh username.

        Args:
            config: Cluster launch options.
        """
        self._host = config.ssh_host
        self._workdir = config.remote_workdir
        self._username = _ssh_config_user(config.ssh_host)
        if not self._username:
            logger.warning(
                f"No username resolved for ssh host '{config.ssh_host}' (no User "
                "line in ~/.ssh/config?); job adoption will scan all users' queued jobs"
            )

    def _rsync(self, src: str, dst: str):
        """Run one rsync transfer over the OpenSSH client.

        Args:
            src: Source, local path or host:path.
            dst: Destination, local path or host:path.

        Returns:
            The completed subprocess result.
        """
        return subprocess.run(
            ["rsync", "-az", "-e", "ssh -o BatchMode=yes", src, dst],
            capture_output=True,
            text=True,
            check=False,
        )

    def push_file(self, local: Path, remote_dir: str):
        """Push one local file into a remote directory, with retries.

        Args:
            local: Local file to push.
            remote_dir: Remote destination directory.

        Raises:
            RuntimeError: If every attempt fails.
        """
        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            result = self._rsync(str(local), f"{self._host}:{remote_dir}/")
            if result.returncode == 0:
                return
            if attempt < _TRANSFER_ATTEMPTS:
                logger.warning(
                    f"rsync push of {local} failed (attempt {attempt}/{_TRANSFER_ATTEMPTS}), "
                    f"retrying in {_TRANSFER_RETRY_DELAY_S:.0f}s: {result.stderr.strip()}"
                )
                time.sleep(_TRANSFER_RETRY_DELAY_S)
        raise RuntimeError(f"rsync push of {local} failed: {result.stderr}")

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        """Pull a single remote file into local_dir.

        Args:
            remote_path: Remote file path.
            local_dir: Local destination directory.

        Returns:
            True if the file was pulled, False if it is unavailable.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            result = self._rsync(f"{self._host}:{remote_path}", f"{local_dir}/")
            if result.returncode == 0:
                return (local_dir / Path(remote_path).name).exists()
            if result.returncode in _RSYNC_SOURCE_MISSING_CODES:
                return False
            if attempt < _TRANSFER_ATTEMPTS:
                logger.warning(
                    f"rsync pull of {remote_path} failed (attempt {attempt}/{_TRANSFER_ATTEMPTS}), "
                    f"retrying in {_TRANSFER_RETRY_DELAY_S:.0f}s: {result.stderr.strip()}"
                )
                time.sleep(_TRANSFER_RETRY_DELAY_S)
        return False

    def ensure_dir(self, path: str):
        """Create a remote directory (and parents) if missing.

        Args:
            path: Remote directory path.

        Raises:
            RuntimeError: If the mkdir fails.
        """
        result = self._ssh(f"mkdir -p {shlex.quote(path)}")
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create remote dir {path}: {result.stderr.strip()}"
            )

    def remove_glob(self, remote_dir: str, pattern: str):
        """Delete remote files matching pattern (the remote shell expands it).

        Args:
            remote_dir: Remote directory holding the files.
            pattern: Shell glob of files to delete.

        Raises:
            RuntimeError: If the remote rm fails.
        """
        result = self._ssh(f"rm -f -- {shlex.quote(remote_dir)}/{pattern}")
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to remove remote files {remote_dir}/{pattern}: {result.stderr.strip()}"
            )


@dataclass
class BatchState:
    """Dispatch bookkeeping for one staged batch."""

    batch_id: str
    shots: list[int]
    input_path: Path
    output_path: Path
    job_base_name: str
    job_id: int | None = None
    attempt: int = 0  # submissions so far; job names carry -a{attempt}
    failures: int = 0  # terminal failures so far, vs config.max_retries
    partition_idx: int = 0
    pending_since: float | None = None
    pending_hops: int = 0
    fail_reason: str | None = None
    done: bool = False
    failed: bool = False
    output_pull_polls: int = 0
    unknown_polls: int = 0

    @property
    def job_name(self) -> str:
        """The job name of the current attempt.

        Returns:
            The base name with the attempt suffix.
        """
        return f"{self.job_base_name}-a{self.attempt}"

    def reset_for_resubmit(self):
        """Clear the per-attempt state so the batch can be submitted again."""
        self.job_id = None
        self.pending_since = None
        self.output_pull_polls = 0
        self.unknown_polls = 0


class ClusterFitDispatcher:
    """Run staged GP fitting batches on a SLURM cluster."""

    def __init__(
        self, config: ClusterFitConfig, ds_name: str, batches_dir: Path, method: str
    ):
        """Set up the dispatcher for one dataset and fitting method.

        Args:
            config: Cluster launch options.
            ds_name: Dataset name, used in job names and batch ids.
            batches_dir: Local directory holding the staged batch npz files.
            method: Fitting method name; part of job names and output
                filenames so two methods' runs never adopt each other's jobs
                or outputs.
        """
        self.config = config
        self.ds_name = ds_name
        self.method = method
        self.worker_module = registry.worker_module(method)
        self.batches_dir = Path(batches_dir)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.backend = _SSHBackend(config)

    def job_name(self, batch_id: str) -> str:
        """Build the base job name of a batch.

        Args:
            batch_id: Batch id.

        Returns:
            The job name without the attempt suffix.
        """
        return f"{self.config.job_name_prefix}-{self.ds_name}-{self.method}-{batch_id}"

    def input_path(self, batch_id: str) -> Path:
        """Get the local input npz path of a batch.

        Args:
            batch_id: Batch id.

        Returns:
            Path of the batch input npz.
        """
        return self.batches_dir / f"batch_{batch_id}.npz"

    def output_path(self, batch_id: str) -> Path:
        """Get the local result npz path of a batch for this method.

        Args:
            batch_id: Batch id.

        Returns:
            Path of the batch result npz.
        """
        return self.batches_dir / f"batch_{batch_id}_out_{self.method}.npz"

    def run(self, batches: dict[str, list[int]]) -> dict[str, bool]:
        """Fit the staged batches on the cluster.

        Blocks until every batch has either produced results or failed.
        Idempotent: completed outputs (local or already on the cluster) and
        queued jobs from a previous run are reused rather than redone. Batch
        failures are transient from the workflow's point of view - they are
        reported here and retried on the next invocation.

        Args:
            batches: Mapping of batch id to staged shots, from
                DataWorkflow.stage_fit_batches.

        Returns:
            Mapping of batch id to whether its results were produced.
        """
        states = []
        for batch_id, shots in sorted(batches.items()):
            state = BatchState(
                batch_id=batch_id,
                shots=shots,
                input_path=self.input_path(batch_id),
                output_path=self.output_path(batch_id),
                job_base_name=self.job_name(batch_id),
            )
            if not state.input_path.exists():
                logger.warning(f"Batch {batch_id}: staged input missing, skipping")
                state.failed = True
                state.fail_reason = "staged input npz missing"
            elif state.output_path.exists():
                state.done = True
                logger.info(
                    f"Batch {batch_id}: output already present locally, skipping job"
                )
            states.append(state)

        self._run_jobs(states)
        summary = self._run_summary(states)
        if any(not s.done for s in states):
            logger.error(summary)
        else:
            logger.info(summary)
        return {s.batch_id: s.done for s in states}

    def clean(self):
        """Cancel this dataset's queued jobs and remove its batch files.

        Covers every method's jobs and files for the dataset, local and
        remote. Call before run() for a from-scratch fit: otherwise
        plan_batches/_run_jobs would adopt the cancelled jobs or reuse
        leftover batch outputs on the cluster.
        """
        prefix = f"{self.config.job_name_prefix}-{self.ds_name}-"
        for name, job_id in self.backend.queued_jobs():
            if not name.startswith(prefix):
                continue
            logger.info(f"Clean: cancelling job {name} (id {job_id})")
            try:
                self.backend.cancel(job_id)
            except Exception as e:
                logger.warning(f"Clean: failed to cancel job {name} (id {job_id}): {e}")

        self._wait_for_jobs_to_drain(prefix)

        # Remove by remote glob, not by mirroring the local batch listing:
        # remote files with no local counterpart (e.g. outputs from a run
        # whose staging was already cleaned) would otherwise survive and be
        # adopted as pre-existing results by the next run. The trailing * also
        # takes the .npz.tmp a killed worker leaves behind mid-write. Raises
        # on failure so a clean that did not actually clean stops the run.
        self.backend.remove_glob(self.config.remote_workdir, "batch_*.npz*")
        if self.batches_dir.exists():
            shutil.rmtree(self.batches_dir)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def _wait_for_jobs_to_drain(self, prefix: str):
        """Block until no job named with prefix is left in the queue.

        scancel returns as soon as it is issued, but SLURM only SIGTERMs (then
        SIGKILLs) the job some time later. Deleting the batch files before the
        job is really gone lets it write its output npz AFTER the delete, and
        the next run pulls that file back and adopts the OLD fit - silently,
        which is the exact failure clean exists to prevent.

        Args:
            prefix: Job name prefix to wait out.

        Raises:
            RuntimeError: If jobs are still queued after the drain timeout; a
                stuck job (e.g. wedged in COMPLETING) needs a human, and
                proceeding would quietly reuse stale fits.
        """
        deadline = time.monotonic() + CLEAN_DRAIN_TIMEOUT_S
        while True:
            remaining = [
                (name, job_id)
                for name, job_id in self.backend.queued_jobs()
                if name.startswith(prefix)
            ]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                listed = ", ".join(
                    f"{name} (id {job_id})" for name, job_id in remaining
                )
                raise RuntimeError(
                    f"Clean: {len(remaining)} {prefix}* jobs still queued "
                    f"{CLEAN_DRAIN_TIMEOUT_S:.0f}s after cancelling: {listed}. "
                    "Not removing batch files - a job that outlives the delete "
                    "would leave a stale output for the next run to adopt. Wait "
                    "for the queue to clear (or scancel them by hand) and rerun."
                )
            logger.info(
                f"Clean: waiting for {len(remaining)} cancelled jobs to leave the queue"
            )
            time.sleep(CLEAN_DRAIN_POLL_S)

    def _push_worker_package(self):
        """Upload the gp_fitting worker package to the cluster.

        The package subtree lands under {remote_workdir}/pkg, and the job
        script exports PYTHONPATH there, so `python -m` resolves the same
        module path as a local run. Only worker-reachable modules ship;
        dispatcher.py stays home (it needs loguru, which the minimal cluster
        venv does not carry).
        """
        import transport_validation_datasets

        pkg_root = Path(transport_validation_datasets.__file__).parent
        remote_pkg = f"{self.config.remote_workdir}/pkg/transport_validation_datasets"
        pushes = [(pkg_root / "__init__.py", remote_pkg)]
        pushes += [
            (p, f"{remote_pkg}/gp_fitting")
            for p in sorted((pkg_root / "gp_fitting").glob("*.py"))
            if p.name != "dispatcher.py"
        ]
        pushes += [
            (p, f"{remote_pkg}/gp_fitting/zk")
            for p in sorted((pkg_root / "gp_fitting" / "zk").glob("*.py"))
        ]
        for remote_dir in sorted({d for _, d in pushes}):
            self.backend.ensure_dir(remote_dir)
        for local, remote_dir in pushes:
            self.backend.push_file(local, remote_dir)

    def _run_jobs(self, states: list[BatchState]):
        """Drive every batch to done or failed: submit, poll, retry.

        Args:
            states: Batch states to run.
        """
        todo = [b for b in states if not b.done and not b.failed]
        if not todo:
            return

        # A previous run's job may have produced output that never made it
        # back (e.g. the pull failed transiently), so check the cluster
        # before submitting anything.
        for state in todo:
            remote_out = f"{self.config.remote_workdir}/{state.output_path.name}"
            if self.backend.pull_file(remote_out, self.batches_dir):
                state.done = True
                logger.info(
                    f"Batch {state.batch_id}: pulled existing results from cluster, skipping job"
                )
        todo = [b for b in todo if not b.done]
        if not todo:
            return

        logger.info(f"Uploading worker package to {self.config.remote_workdir}/pkg")
        self._push_worker_package()
        # sbatch does not create --output directories
        self.backend.ensure_dir(f"{self.config.remote_workdir}/logs")

        # Adopt jobs already in the queue from a previous run
        queued = self.backend.queued_job_names()
        for state in todo:
            self._adopt_queued_job(state, queued)

        while True:
            self._poll_finished(todo)
            self._submit_ready(todo)
            remaining = [b for b in todo if not b.done and not b.failed]
            if not remaining:
                break
            n_active = len([b for b in remaining if b.job_id is not None])
            logger.info(
                f"Waiting on {len(remaining)} batches ({n_active} jobs active), "
                f"polling again in {self.config.poll_interval_s:.0f}s"
            )
            time.sleep(self.config.poll_interval_s)

    def _submit_ready(self, todo: list[BatchState]):
        """Submit unsubmitted batches while the concurrency budget allows.

        Args:
            todo: Batch states still in play.
        """
        active = [
            b for b in todo if b.job_id is not None and not b.done and not b.failed
        ]
        budget = self.config.max_concurrent_jobs - len(active)
        for state in todo:
            if budget <= 0:
                break
            if state.done or state.failed or state.job_id is not None:
                continue
            state.job_id = self._submit_batch(state)
            budget -= 1

    def _adopt_queued_job(self, state: BatchState, queued: dict[str, int]):
        """Adopt a queued job from a previous run instead of resubmitting.

        Job names carry an attempt suffix (-a{n}). The highest attempt wins
        and lower-attempt stragglers are cancelled.

        Args:
            state: Batch to adopt a job for.
            queued: Queued job names to job ids.
        """
        candidates: list[tuple[int, int]] = []  # (attempt, job_id)
        for name, job_id in queued.items():
            if name.startswith(f"{state.job_base_name}-a"):
                suffix = name.removeprefix(f"{state.job_base_name}-a")
                if suffix.isdigit():
                    candidates.append((int(suffix), job_id))
        if not candidates:
            return
        candidates.sort()
        state.attempt, state.job_id = candidates[-1]
        state.pending_since = time.monotonic()
        logger.info(
            f"Batch {state.batch_id}: found existing job {state.job_id} "
            f"(attempt {state.attempt}) in queue, not resubmitting"
        )
        for _, stale_id in candidates[:-1]:
            logger.info(
                f"Batch {state.batch_id}: cancelling stale lower-attempt job {stale_id}"
            )
            try:
                self.backend.cancel(stale_id)
            except Exception as e:
                logger.warning(
                    f"Batch {state.batch_id}: failed to cancel stale job {stale_id}: {e}"
                )

    def _render_script(self, state: BatchState, part: PartitionSpec) -> str:
        """Render the sbatch script of one batch attempt.

        Args:
            state: Batch to run.
            part: Partition to submit to.

        Returns:
            The sbatch script text.
        """
        workdir = self.config.remote_workdir
        lines = [
            "#!/bin/bash",
            "",
            f"#SBATCH --job-name={state.job_name}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_job}",
        ]
        if self.config.memory_per_node:
            lines.append(f"#SBATCH --mem={self.config.memory_per_node}")
        lines.append(f"#SBATCH --time={part.time_limit}")
        lines.append(f"#SBATCH --partition={part.name}")
        if part.constraint:
            lines.append(f"#SBATCH --constraint={part.constraint}")
        lines += [
            f"#SBATCH --output={workdir}/logs/%x_%j.log",
            f"#SBATCH --error={workdir}/logs/%x_%j.log",
            f"#SBATCH --chdir={workdir}",
            "#SBATCH --wait-all-nodes=1",
            "",
            "set -euxo pipefail",
            "",
            f"source '{self.config.venv_path}/bin/activate'",
            f"export PYTHONPATH={workdir}/pkg",
            "",
            f"srun python -m {self.worker_module} {workdir}/{state.input_path.name} "
            f"{workdir}/{state.output_path.name} --num-workers {self.config.cpus_per_job}",
            "",
        ]
        return "\n".join(lines)

    def _submit_batch(self, state: BatchState) -> int:
        """Push a batch's input and submit its job.

        Args:
            state: Batch to submit.

        Returns:
            The submitted SLURM job id.
        """
        workdir = self.config.remote_workdir
        self.backend.push_file(state.input_path, workdir)

        part = self.config.partitions[state.partition_idx]
        state.attempt += 1
        script = self._render_script(state, part)
        job_id = self.backend.submit_script(script, state.job_name)
        state.pending_since = time.monotonic()
        logger.info(
            f"Batch {state.batch_id}: submitted job {state.job_name} (id {job_id}, "
            f"{len(state.shots)} shots, partition {part.name}, attempt {state.attempt})"
        )
        return job_id

    def _poll_finished(self, todo: list[BatchState]):
        """Poll active jobs, pulling outputs and handling failures.

        Args:
            todo: Batch states still in play.
        """
        active = [
            b for b in todo if b.job_id is not None and not b.done and not b.failed
        ]
        if not active:
            return
        states = self.backend.job_states([b.job_id for b in active])
        for state in active:
            slurm_state = states.get(state.job_id, "UNKNOWN")
            if slurm_state == "PENDING":
                self._check_pending_timeout(state)
                continue
            if slurm_state in ("RUNNING", "COMPLETING", "CONFIGURING"):
                state.pending_since = None
                continue
            # Terminal or unknown: the output file is the source of truth
            remote_out = f"{self.config.remote_workdir}/{state.output_path.name}"
            if self.backend.pull_file(remote_out, self.batches_dir):
                state.done = True
                logger.info(
                    f"Batch {state.batch_id}: job {state.job_id} finished, results pulled back"
                )
            elif slurm_state in _TERMINAL_FAILURE_STATES:
                self._handle_failure(
                    state,
                    f"job {state.job_id} ended in state {slurm_state} without producing {remote_out}",
                )
            elif slurm_state == "COMPLETED":
                # The job claims success, so the output may exist but be
                # unreachable (login node dropping connections) or still in
                # flight. Keep trying for a few polls before giving up.
                state.output_pull_polls += 1
                if state.output_pull_polls >= _MAX_OUTPUT_PULL_POLLS:
                    self._handle_failure(
                        state,
                        f"job {state.job_id} COMPLETED but {remote_out} could "
                        f"not be pulled after {state.output_pull_polls} polls",
                    )
                else:
                    logger.warning(
                        f"Batch {state.batch_id}: job {state.job_id} COMPLETED but "
                        f"output not retrieved yet (poll "
                        f"{state.output_pull_polls}/{_MAX_OUTPUT_PULL_POLLS}), will retry"
                    )
            else:
                # UNKNOWN (e.g. the job vanished from the queue): tolerate a
                # few polls, then treat the attempt as failed
                state.unknown_polls += 1
                if state.unknown_polls >= _MAX_UNKNOWN_POLLS:
                    self._handle_failure(
                        state,
                        f"job {state.job_id} in state {slurm_state} for "
                        f"{state.unknown_polls} polls with no output",
                    )
                else:
                    logger.warning(
                        f"Batch {state.batch_id}: job {state.job_id} state {slurm_state}, no output yet"
                    )

    def _check_pending_timeout(self, state: BatchState):
        """Cancel a job stuck PENDING too long and hop to the next partition.

        Stops hopping after one full cycle through the partition list: if
        every partition is congested, cancelling only resets the batch's
        queue position.

        Args:
            state: Batch whose job is PENDING.
        """
        if state.pending_since is None:
            # Job returned to PENDING (e.g. preemption requeue): restart the clock
            state.pending_since = time.monotonic()
            return
        if len(self.config.partitions) < 2 or state.pending_hops >= len(
            self.config.partitions
        ):
            return
        elapsed = time.monotonic() - state.pending_since
        if elapsed <= self.config.pending_timeout_s:
            return
        old_part = self.config.partitions[state.partition_idx].name
        state.partition_idx = (state.partition_idx + 1) % len(self.config.partitions)
        state.pending_hops += 1
        new_part = self.config.partitions[state.partition_idx].name
        logger.warning(
            f"Batch {state.batch_id}: job {state.job_id} PENDING for {elapsed:.0f}s "
            f"on {old_part}, cancelling and falling back to {new_part}"
        )
        try:
            self.backend.cancel(state.job_id)
        except Exception as e:
            logger.warning(
                f"Batch {state.batch_id}: failed to cancel job {state.job_id}: {e}"
            )
        state.reset_for_resubmit()
        if state.pending_hops >= len(self.config.partitions):
            logger.warning(
                f"Batch {state.batch_id}: tried every partition for pending "
                "fallback, will wait in queue from now on"
            )

    def _handle_failure(self, state: BatchState, reason: str):
        """Retry a failed batch on the next partition, or give up past max_retries.

        Args:
            state: Batch whose attempt failed.
            reason: What went wrong, for the logs and the run summary.
        """
        state.failures += 1
        part = self.config.partitions[state.partition_idx]
        if state.failures > self.config.max_retries:
            state.failed = True
            state.fail_reason = (
                f"{reason} (partition {part.name}, attempt {state.attempt}, "
                f"{state.failures - 1}/{self.config.max_retries} retries used)"
            )
            logger.error(
                f"Batch {state.batch_id}: {state.fail_reason}; giving up, see logs "
                f"in {self.config.remote_workdir}/logs"
            )
            return
        state.partition_idx = (state.partition_idx + 1) % len(self.config.partitions)
        next_part = self.config.partitions[state.partition_idx].name
        logger.warning(
            f"Batch {state.batch_id}: {reason}; retry {state.failures}/{self.config.max_retries} "
            f"on partition {next_part}"
        )
        state.reset_for_resubmit()

    @staticmethod
    def _run_summary(states: list[BatchState]) -> str:
        """Build the reconciliation report: every batch is accounted for.

        Args:
            states: Batch states after the run.

        Returns:
            Multi-line summary text.
        """
        failed = [s for s in states if not s.done]
        n_shots = sum(len(s.shots) for s in states)
        n_failed_shots = sum(len(s.shots) for s in failed)
        lines = [
            f"GP fit dispatch: {len(states)} batches ({n_shots} shots), "
            f"{len(states) - len(failed)} done, {len(failed)} FAILED "
            f"({n_failed_shots} shots; failed batches retry on the next run)"
        ]
        for state in failed:
            reason = state.fail_reason or "output missing, unreadable, or incomplete"
            lines.append(
                f"  batch {state.batch_id} (attempts {state.attempt}): {reason}; "
                f"shots: {', '.join(str(s) for s in state.shots)}"
            )
        return "\n".join(lines)
