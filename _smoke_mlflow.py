"""Temporary smoke check for the MLflow integration (with a fake client)."""

import time
from types import SimpleNamespace

from orchestrator.core import (
    ExperimentStatus,
    GridSearch,
    LocalLauncher,
    MetricMonitor,
    intake_objective,
    run_local,
)
from orchestrator.integrations import (
    MlflowError,
    MlflowListener,
    MlflowTracker,
    track_result,
)


class FakeClient:
    """Minimal stand-in for mlflow.tracking.MlflowClient that records calls."""

    def __init__(self):
        self.experiments = {}  # name -> id
        self.runs = {}  # run_id -> dict(params, metrics, tags, status, experiment_id)
        self._n = 0

    def get_experiment_by_name(self, name):
        if name in self.experiments:
            return SimpleNamespace(experiment_id=self.experiments[name])
        return None

    def create_experiment(self, name):
        eid = f"exp-{len(self.experiments)}"
        self.experiments[name] = eid
        return eid

    def create_run(self, experiment_id, tags=None, **kw):
        self._n += 1
        run_id = f"run-{self._n}"
        self.runs[run_id] = {
            "experiment_id": experiment_id,
            "params": {},
            "metrics": [],
            "tags": dict(tags or {}),
            "status": None,
        }
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_param(self, run_id, key, value):
        self.runs[run_id]["params"][key] = value

    def log_metric(self, run_id, key, value, timestamp=None, step=None):
        self.runs[run_id]["metrics"].append((key, value, step, timestamp))

    def set_tag(self, run_id, key, value):
        self.runs[run_id]["tags"][key] = value

    def set_terminated(self, run_id, status=None, **kw):
        self.runs[run_id]["status"] = status


# ---- not-installed path: lazy import error -------------------------------
t = MlflowTracker("noclient")  # no client injected, mlflow not installed
try:
    _ = t.client
    print("lazy import: NO error (unexpected — is mlflow installed?)")
except MlflowError as exc:
    print("lazy import raises MlflowError:", "not installed" in str(exc))


obj = intake_objective("examples/objectives/tune_resnet.yaml")
obj = obj.model_copy(update={"max_experiments": 3, "max_concurrency": 1, "target_metric_value": None})


def train(experiment):
    return {"val_accuracy": 0.7, "loss": 0.3}


res = run_local(obj, train, strategy="random")

# ---- log a completed result ----------------------------------------------
fake = FakeClient()
tracker = MlflowTracker.for_objective(obj, client=fake)
tracker.log_result(res)

print("experiment created:", obj.name in fake.experiments)
print("one run per experiment:", len(fake.runs) == len(res.experiments))
sample = next(iter(fake.runs.values()))
print("params logged:", "learning_rate" in sample["params"])
print("metrics logged (acc+loss):", {m[0] for m in sample["metrics"]} == {"val_accuracy", "loss"})
print("status terminated FINISHED:", sample["status"] == "FINISHED")
print("final_status tag set:", sample["tags"].get("final_status") == "completed")
best_run = fake.runs[tracker.run_id_for(res.best_experiment.id)]
print("best run tagged:", best_run["tags"].get("best") == "true")

# ---- status mapping for failures -----------------------------------------
def boom(e):
    raise RuntimeError("x")

resf = run_local(obj.model_copy(update={"max_experiments": 1}), boom, strategy="random")
fake2 = FakeClient()
MlflowTracker("fail-exp", client=fake2).log_result(resf)
frun = next(iter(fake2.runs.values()))
print("failed run -> FAILED status:", frun["status"] == "FAILED")

# ---- live listener wired into the monitor --------------------------------
def streaming(experiment, ctx):
    for step in range(4):
        if ctx.should_stop():
            return None
        ctx.report("val_accuracy", 0.6 + step * 0.05, step=step)
        time.sleep(0.01)
    return {"val_accuracy": 0.8}


fake3 = FakeClient()
tracker3 = MlflowTracker("live-exp", client=fake3)
with LocalLauncher(streaming, max_workers=2) as launcher:
    exps = GridSearch().propose(obj, count=2)
    mon = MetricMonitor(launcher, interval=0.01)
    listener = MlflowListener(tracker3, exps)
    mon.add_listener(listener)
    jobs = [launcher.launch(e) for e in exps]
    mon.track_all(jobs)
    list(mon.stream(timeout=5))

print("live: runs created for both:", len(fake3.runs) == 2)
some = next(iter(fake3.runs.values()))
print("live: streamed metrics logged (>=4):", sum(1 for m in some["metrics"] if m[0] == "val_accuracy") >= 4)
print("live: runs terminated:", all(r["status"] is not None for r in fake3.runs.values()))

# ---- track_result convenience --------------------------------------------
fake4 = FakeClient()
tr = track_result(res, objective=obj, client=fake4)
print("track_result logged all:", len(fake4.runs) == len(res.experiments))
