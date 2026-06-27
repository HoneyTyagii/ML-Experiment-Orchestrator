"""Temporary smoke check for the Kubeflow launcher (fake kfp client)."""

from types import SimpleNamespace

from orchestrator.core import (
    Experiment,
    ExperimentStatus,
    available_launchers,
    get_launcher,
)
# importing the module registers the launcher
from orchestrator.integrations.kubeflow import (
    KubeflowError,
    KubeflowLauncher,
    _map_state,
)


class FakeKfp:
    def __init__(self):
        self.scripts = {}
        self.metrics = {}
        self.terminated = []
        self._n = 0
        self.next_script = None
        self.next_metrics = None
        self.last_args = None

    def create_run_from_pipeline_package(self, pipeline, arguments=None, run_name=None,
                                         experiment_name=None, **kw):
        self._n += 1
        rid = f"kfp-{self._n}"
        self.scripts[rid] = list(self.next_script or ["Running", "Succeeded"])
        self.metrics[rid] = dict(self.next_metrics or {})
        self.last_args = arguments
        self.next_script = None
        self.next_metrics = None
        return SimpleNamespace(run_id=rid)

    def get_run(self, run_id):
        seq = self.scripts[run_id]
        state = seq[0] if len(seq) == 1 else seq.pop(0)
        return SimpleNamespace(state=state)

    def terminate_run(self, run_id):
        self.terminated.append(run_id)
        self.scripts[run_id] = ["Canceled"]


def metrics_reader(client, run_id):
    return client.metrics.get(run_id, {})


# ---- state mapping --------------------------------------------------------
print("map succeeded:", _map_state("Succeeded") is ExperimentStatus.COMPLETED)
print("map failed:", _map_state("Failed") is ExperimentStatus.FAILED)
print("map canceled:", _map_state("Canceled") is ExperimentStatus.CANCELLED)
print("map running:", _map_state("Running") is ExperimentStatus.RUNNING)
print("map unknown -> running:", _map_state("weird") is ExperimentStatus.RUNNING)

# ---- registered in the launcher registry ----------------------------------
print("registered as kubeflow:", "kubeflow" in available_launchers())

# ---- lazy import error (no kfp installed) ---------------------------------
try:
    KubeflowLauncher(pipeline="p.yaml").client
    print("lazy import: NO error (unexpected)")
except KubeflowError as exc:
    print("lazy import raises KubeflowError:", "not installed" in str(exc))

exp = Experiment(objective_id="o", name="exp-a", hyperparameters={"lr": 0.01, "bs": 64})

# ---- launch / poll transition / result ------------------------------------
fake = FakeKfp()
fake.next_metrics = {"val_accuracy": 0.91, "loss": 0.2}
launcher = get_launcher("kubeflow", pipeline="pipe.yaml", client=fake,
                        experiment_name="exp", metrics_fn=metrics_reader)
job = launcher.launch(exp)
print("launch -> RUNNING + handle:", job.status is ExperimentStatus.RUNNING and job.handle == "kfp-1")
print("arguments mapped from hyperparameters:", fake.last_args == {"lr": 0.01, "bs": 64})
print("first poll RUNNING:", launcher.poll(job) is ExperimentStatus.RUNNING)
print("second poll COMPLETED:", launcher.poll(job) is ExperimentStatus.COMPLETED and job.finished_at is not None)
res = launcher.result(job)
print("result metrics:", {(m.name, m.value) for m in res.metrics} == {("val_accuracy", 0.91), ("loss", 0.2)})
print("result status COMPLETED:", res.status is ExperimentStatus.COMPLETED)

# ---- run() convenience (launch + wait) ------------------------------------
fake.next_metrics = {"val_accuracy": 0.8}
res2 = launcher.run(Experiment(objective_id="o", name="exp-b"), poll_interval=0)
print("run() reaches COMPLETED:", res2.status is ExperimentStatus.COMPLETED)

# ---- failure mapping ------------------------------------------------------
fake.next_script = ["Failed"]
fjob = launcher.launch(Experiment(objective_id="o", name="exp-fail"))
print("failure poll:", launcher.poll(fjob) is ExperimentStatus.FAILED)
fres = launcher.result(fjob)
print("failure result error mentions state:", "Failed" in (fres.error or ""))

# ---- cancel terminates the run --------------------------------------------
cjob = launcher.launch(Experiment(objective_id="o", name="exp-cancel"))
launcher.poll(cjob)  # Running
launcher.cancel(cjob)
print("cancel terminated run:", cjob.handle in fake.terminated)
print("cancel -> CANCELLED:", cjob.status is ExperimentStatus.CANCELLED)

# ---- no pipeline configured -> error --------------------------------------
try:
    KubeflowLauncher(client=FakeKfp()).launch(exp)
    print("no pipeline: NO error (unexpected)")
except KubeflowError as exc:
    print("no pipeline rejected:", "no pipeline" in str(exc))
