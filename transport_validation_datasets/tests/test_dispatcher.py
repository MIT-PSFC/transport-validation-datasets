"""The cluster dispatcher's job bookkeeping, against a fake SLURM backend.

No ssh and no SLURM: the backend is replaced by an in-memory stand-in
that records what the dispatcher asked of it.
"""

import subprocess

import pytest

import transport_validation_datasets.gp_fitting.dispatcher as dispatcher_module
from transport_validation_datasets.gp_fitting.dispatcher import (
    BatchState,
    ClusterFitConfig,
    ClusterFitDispatcher,
    ClusterUnreachableError,
)


class FakeBackend:
    """Stand-in for _SSHBackend, scripted from the test."""

    def __init__(self, config, workdir):
        self.queued: dict[str, list[int]] = {}
        self.states: dict[int, str] = {}
        self.unreachable = False
        self.cancelled: list[int] = []
        self.submitted: list[str] = []
        self.next_job_id = 100

    def queued_jobs(self):
        return self.queued

    def job_states(self, job_ids):
        if self.unreachable:
            raise ClusterUnreachableError("ssh: connect to host failed")
        return {jid: self.states[jid] for jid in job_ids if jid in self.states}

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def submit_script(self, script, job_name):
        self.submitted.append(job_name)
        self.next_job_id += 1
        return self.next_job_id

    def push_file(self, local, remote_dir):
        pass

    def pull_file(self, remote_path, local_dir):
        return False

    def ensure_dir(self, path):
        pass


def _config() -> ClusterFitConfig:
    return ClusterFitConfig(
        ssh_host="cluster",
        partitions="part_a@1:00:00,part_b@2:00:00",
        remote_workdir="/scratch/gpfit",
        venv_path="/scratch/gpfit/.venv",
    )


@pytest.fixture
def dispatcher(tmp_path, monkeypatch) -> ClusterFitDispatcher:
    monkeypatch.setattr(dispatcher_module, "_SSHBackend", FakeBackend)
    return ClusterFitDispatcher(
        _config(), "ds", tmp_path / "batches", "zk", tmp_path / "logs"
    )


def _state(dispatcher: ClusterFitDispatcher) -> BatchState:
    batch_id = "abc"
    input_path = dispatcher.input_path(batch_id)
    input_path.write_bytes(b"")
    return BatchState(
        batch_id=batch_id,
        shots=[1, 2],
        input_path=input_path,
        output_path=dispatcher.output_path(batch_id),
        job_base_name=dispatcher.job_name(batch_id),
    )


def test_adopts_one_of_duplicate_names_and_cancels_others(dispatcher):
    state = _state(dispatcher)
    base = state.job_base_name
    dispatcher.backend.queued = {
        f"{base}-a1": [12, 11],
        f"{base}-a0": [10],
        "gpfit-other-zk-xyz-a1": [50],
    }

    dispatcher._adopt_queued_job(state, dispatcher.backend.queued_jobs())

    assert (state.attempt, state.job_id) == (1, 11)
    assert sorted(dispatcher.backend.cancelled) == [10, 12]


def test_unknown_job_is_cancelled_before_resubmission(dispatcher, monkeypatch):
    monkeypatch.setattr(dispatcher_module, "_MAX_UNKNOWN_POLLS", 2)
    state = _state(dispatcher)
    state.job_id, state.attempt = 7, 1

    dispatcher._poll_finished([state])
    assert state.unknown_polls == 1
    assert dispatcher.backend.cancelled == []

    dispatcher._poll_finished([state])
    assert dispatcher.backend.cancelled == [7]
    assert state.job_id is None
    assert not state.failed

    dispatcher._submit_ready([state])
    assert state.job_id == 101
    assert dispatcher.backend.submitted == [f"{state.job_base_name}-a2"]


def test_ssh_outage_does_not_count_as_an_unknown_poll(dispatcher):
    state = _state(dispatcher)
    state.job_id = 7
    dispatcher.backend.unreachable = True

    for _ in range(dispatcher_module._MAX_UNKNOWN_POLLS + 1):
        dispatcher._poll_finished([state])

    assert state.unknown_polls == 0
    assert state.job_id == 7
    assert dispatcher.backend.cancelled == []


class TestSSHBackend:
    @pytest.fixture
    def backend(self, monkeypatch):
        monkeypatch.setattr(dispatcher_module, "_ssh_config_user", lambda dest: "me")
        return dispatcher_module._SSHBackend(_config(), "/scratch/gpfit/ds")

    def test_queued_jobs_keeps_every_id_of_a_name(self, backend, monkeypatch):
        def ssh(cmd, *, stdin=None):
            return subprocess.CompletedProcess(cmd, 0, "11|n\n12|n\n13|m\n", "")

        monkeypatch.setattr(backend, "_ssh", ssh)

        assert backend.queued_jobs() == {"n": [11, 12], "m": [13]}

    def test_job_states_reads_squeue_past_a_gone_job(self, backend, monkeypatch):
        # squeue exits 1 when one of the ids has left the queue,
        # sacct then resolves that one, and the third is found nowhere
        def ssh(cmd, *, stdin=None):
            if cmd.startswith("squeue"):
                return subprocess.CompletedProcess(
                    cmd, 1, "7|RUNNING\n", "Invalid job id"
                )
            if cmd.startswith("sacct"):
                return subprocess.CompletedProcess(cmd, 0, "8|CANCELLED by 1\n", "")
            return subprocess.CompletedProcess(cmd, 1, "", "")

        monkeypatch.setattr(backend, "_ssh", ssh)

        assert backend.job_states([7, 8, 9]) == {7: "RUNNING", 8: "CANCELLED"}
