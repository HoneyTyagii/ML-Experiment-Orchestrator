"""Temporary smoke check for the Weights & Biases integration (fake init)."""

import time

from orchestrator.core import (
    GridSearch,
    LocalLauncher,
    MetricMonitor,
    intake_objective,
    run_local,
)
from orchestrator.integrations.wandb import (
    WandbError,
    WandbListener,
    WandbTracker,
    track_result,
)


class FakeRun:
    def __init__(self, **kw):
        self.kwargs = kw
        self.history = []   # list of (data, step)
        self.summary = {}
        self.finished = None  # exit_code

    def log(self, data, step=None):
        self.history.append((dict(data), step))

    def finish(self, exit_code=None):
        self.finished = exit_code


class FakeInit:
    def __init__(self):
        self.runs = []

    def __call__(self, **kwargs):
        run = FakeRun(**kwargs)
        self.runs.append(run)
        return run


# ---- not-installed path ---------------------------------------------------
t = WandbTracker("noinit")
try:
    _ = t.init_fn
    print("lazy import: NO error (unexpected — is wandb installed?)")
except WandbError as exc:
    print("lazy import raises WandbError:", "not installed" in str(exc))


obj = intake_objective("examples/objectives/tune_resnet.yaml")
obj = obj.model_copy(update={"max_experiments": 3, "max_concurrency": 1, "target_metric_value": None})


def train(experiment):
    return {"val_accuracy": 0.7, "loss": 0.3}


res = run_local(obj, train, strategy="random")

# ---- log a completed result ----------------------------------------------
fake = FakeInit()
tracker = WandbTracker.for_objective(obj, entity="team", init=fake)
tracker.log_result(res)

print("one run per experiment:", len(fake.runs) == len(res.experiments))
r0 = fake.runs[0]
print("project + entity passed:", r0.kwargs["project"] == obj.name and r0.kwargs["entity"] == "team")
print("config has hyperparameters:", "learning_rate" in r0.kwargs["config"])
print("group defaults to objective id:", r0.kwargs["group"] == obj.id)
print("reinit set:", r0.kwargs["reinit"] is True)
print("metrics logged (acc+loss keys):", {k for d, _ in r0.history for k in d} == {"val_accuracy", "loss"})
print("final_status in summary:", r0.summary.get("final_status") == "completed")
print("runs finished exit 0:", all(r.finished == 0 for r in fake.runs))
best_run = tracker.run_for(res.best_experiment.id)
print("best run flagged:", best_run.summary.get("best") is True)

# ---- failure -> exit_code 1 ----------------------------------------------
def boom(e):
    raise RuntimeError("x")

resf = run_local(obj.model_copy(update={"max_experiments": 1}), boom, strategy="random")
fake2 = FakeInit()
WandbTracker("fail", init=fake2).log_result(resf)
print("failed run exit_code 1:", fake2.runs[0].finished == 1)
print("failed run status FAILED:", fake2.runs[0].summary.get("final_status") == "failed")

# ---- step monotonicity: out-of-order final step handled ------------------
fake3 = FakeInit()
tr = WandbTracker("steps", init=fake3)
from orchestrator.core import Experiment, MetricValue
e = Experiment(objective_id="o", name="e")
e.record_metric("acc", 0.5, step=0)
e.record_metric("acc", 0.6, step=1)
e.record_metric("acc", 0.7, step=2)
e.record_metric("acc", 0.99, step=0)  # late, out-of-order (final summary-style)
tr.log_experiment(e)
hist = fake3.runs[0].history
print("in-order logs keep step:", hist[0][1] == 0 and hist[2][1] == 2)
print("out-of-order logged without step:", hist[3][1] is None)

# ---- live listener --------------------------------------------------------
def streaming(experiment, ctx):
    for step in range(4):
        if ctx.should_stop():
            return None
        ctx.report("val_accuracy", 0.6 + 0.05 * step, step=step)
        time.sleep(0.01)
    return {"val_accuracy": 0.8}


fake4 = FakeInit()
tracker4 = WandbTracker("live", init=fake4)
with LocalLauncher(streaming, max_workers=2) as launcher:
    exps = GridSearch().propose(obj, count=2)
    mon = MetricMonitor(launcher, interval=0.01)
    mon.add_listener(WandbListener(tracker4, exps))
    jobs = [launcher.launch(x) for x in exps]
    mon.track_all(jobs)
    list(mon.stream(timeout=5))

print("live: 2 runs:", len(fake4.runs) == 2)
print("live: streamed >=4 each:", all(len(r.history) >= 4 for r in fake4.runs))
print("live: all finished:", all(r.finished is not None for r in fake4.runs))

# ---- track_result convenience --------------------------------------------
fake5 = FakeInit()
track_result(res, objective=obj, init=fake5)
print("track_result logged all:", len(fake5.runs) == len(res.experiments))
