"""Report generation with metric summaries (pipeline stage 7: *generate reports*).

Turns the structured outcome of a tuning run into a human-readable summary. The
report is assembled once into a :class:`Report` value object and can then be
rendered to Markdown, plain text, or JSON -- so the same data drives a console
recap, a committed ``REPORT.md``, or a machine-readable artifact.

A report captures: the objective and its target, a run-status breakdown, summary
statistics for the primary metric, the leaderboard (top-N), the winning
configuration, and a tally of early-stop reasons.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.core.models import Goal, Objective
from orchestrator.core.ranking import Leaderboard, RankingSummary, rank_result

if TYPE_CHECKING:
    from orchestrator.core.loop import TuningResult


def _fmt(value: float | None, places: int = 4) -> str:
    """Compactly format a float for display."""
    if value is None:
        return "-"
    return f"{value:.{places}g}"


def _params(hyperparameters: dict[str, Any]) -> str:
    """Render hyperparameters as a compact ``k=v`` string."""
    if not hyperparameters:
        return "(defaults)"
    return ", ".join(f"{k}={_fmt(v) if isinstance(v, float) else v}" for k, v in hyperparameters.items())


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


@dataclass
class Report:
    """Structured summary of a tuning run, renderable to several formats."""

    title: str
    objective_name: str
    objective_id: str
    primary_metric: str
    goal: Goal
    dataset: str | None
    target_metric_value: float | None
    target_reached: bool
    rounds: int
    total_experiments: int
    status_counts: dict[str, int]
    summary: RankingSummary
    leaderboard: Leaderboard
    stop_reasons: dict[str, int]
    top_n: int = 10
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # -- serializable view -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        best = self.leaderboard.best
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "objective": {
                "id": self.objective_id,
                "name": self.objective_name,
                "primary_metric": self.primary_metric,
                "goal": self.goal.value,
                "dataset": self.dataset,
                "target_metric_value": self.target_metric_value,
                "target_reached": self.target_reached,
            },
            "run": {
                "total_experiments": self.total_experiments,
                "rounds": self.rounds,
                "status_counts": self.status_counts,
                "early_stops": self.stop_reasons,
            },
            "metric_summary": {
                "metric": self.summary.metric,
                "goal": self.summary.goal.value,
                "count": self.summary.count,
                "best": self.summary.best,
                "worst": self.summary.worst,
                "mean": self.summary.mean,
                "median": self.summary.median,
            },
            "best": None
            if best is None
            else {
                "experiment_id": best.experiment.id,
                "name": best.experiment.name,
                "score": best.score,
                "hyperparameters": best.experiment.hyperparameters,
            },
            "leaderboard": self.leaderboard.rows()[: self.top_n],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # -- text renderers ----------------------------------------------------

    def to_markdown(self) -> str:
        """Render the report as Markdown."""
        goal = self.goal.value
        target = "none" if self.target_metric_value is None else _fmt(self.target_metric_value)
        lines: list[str] = [
            f"# {self.title}",
            "",
            f"- **Objective:** {self.objective_name} (`{self.objective_id}`)",
            f"- **Primary metric:** `{self.primary_metric}` ({goal})",
            f"- **Dataset:** {self.dataset or '-'}",
            f"- **Target:** {target}  ·  **reached:** {'yes' if self.target_reached else 'no'}",
            f"- **Rounds:** {self.rounds}  ·  **Experiments:** {self.total_experiments}",
            f"- **Generated:** {self.generated_at.isoformat(timespec='seconds')}",
            "",
            "## Run summary",
            "",
            _md_table(
                ["Status", "Count"],
                [[status, str(count)] for status, count in sorted(self.status_counts.items())],
            ),
            "",
            f"## Metric summary — `{self.primary_metric}` ({goal})",
            "",
        ]
        if self.summary.count:
            lines += [
                f"- **Best:** {_fmt(self.summary.best)}",
                f"- **Worst:** {_fmt(self.summary.worst)}",
                f"- **Mean:** {_fmt(self.summary.mean)}",
                f"- **Median:** {_fmt(self.summary.median)}",
                "",
            ]
        else:
            lines += ["_No experiment produced the primary metric._", ""]

        lines += [f"## Leaderboard (top {self.top_n})", ""]
        entries = self.leaderboard.top(self.top_n)
        if entries:
            rows = [
                [
                    str(e.rank),
                    e.experiment.name or e.experiment.id[:8],
                    _fmt(e.score),
                    e.experiment.status.value,
                    _params(e.experiment.hyperparameters),
                ]
                for e in entries
            ]
            lines.append(_md_table(["Rank", "Experiment", "Score", "Status", "Params"], rows))
        else:
            lines.append("_No ranked experiments._")
        lines.append("")

        best = self.leaderboard.best
        if best is not None:
            lines += [
                "## Best configuration",
                "",
                f"- **Experiment:** {best.experiment.name or best.experiment.id}",
                f"- **Score:** {_fmt(best.score)} ({self.primary_metric}, {goal})",
                f"- **Hyperparameters:** {_params(best.experiment.hyperparameters)}",
                "",
            ]

        if self.stop_reasons:
            lines += ["## Early stops", ""]
            lines.append(
                _md_table(
                    ["Reason", "Count"],
                    [[reason, str(count)] for reason, count in sorted(self.stop_reasons.items())],
                )
            )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_text(self) -> str:
        """Render the report as plain text (Markdown stripped of decoration)."""
        text = self.to_markdown()
        out = []
        for raw in text.splitlines():
            line = raw.replace("**", "").replace("`", "")
            if line.startswith("# "):
                line = line[2:].upper()
            elif line.startswith("## "):
                line = line[3:].upper()
            out.append(line)
        return "\n".join(out)


def build_report(
    result: TuningResult,
    *,
    objective: Objective | None = None,
    leaderboard: Leaderboard | None = None,
    top_n: int = 10,
    title: str | None = None,
) -> Report:
    """Assemble a :class:`Report` from a tuning result.

    Parameters
    ----------
    objective:
        Optional source of richer header details (name, dataset, target). When
        omitted, values are taken from ``result`` where available.
    leaderboard:
        Precomputed leaderboard; if omitted, one is built via :func:`rank_result`.
    top_n:
        Maximum number of leaderboard rows to include.
    """
    board = leaderboard if leaderboard is not None else rank_result(result)

    status_counts = Counter(e.status.value for e in result.experiments)
    stop_reasons = Counter(d.reason.value for d in result.stop_decisions)

    name = objective.name if objective is not None else result.objective_id
    dataset = objective.dataset if objective is not None else None
    target = (
        objective.target_metric_value
        if objective is not None
        else getattr(result, "target_metric_value", None)
    )

    return Report(
        title=title or f"Experiment Report: {name}",
        objective_name=name,
        objective_id=result.objective_id,
        primary_metric=result.primary_metric,
        goal=result.goal,
        dataset=dataset,
        target_metric_value=target,
        target_reached=result.target_reached,
        rounds=result.rounds,
        total_experiments=len(result.experiments),
        status_counts=dict(status_counts),
        summary=board.summary(),
        leaderboard=board,
        stop_reasons=dict(stop_reasons),
        top_n=top_n,
    )


_RENDERERS = {
    "markdown": "to_markdown",
    "md": "to_markdown",
    "text": "to_text",
    "txt": "to_text",
    "json": "to_json",
}


def generate_report(
    result: TuningResult,
    *,
    fmt: str = "markdown",
    objective: Objective | None = None,
    top_n: int = 10,
) -> str:
    """Build and render a report in one call. ``fmt``: markdown | text | json."""
    try:
        method = _RENDERERS[fmt.lower()]
    except KeyError:
        raise ValueError(f"unknown report format {fmt!r}; choose from {sorted(_RENDERERS)}") from None
    report = build_report(result, objective=objective, top_n=top_n)
    return getattr(report, method)()


def write_report(
    result: TuningResult,
    path: str | Path,
    *,
    fmt: str | None = None,
    objective: Objective | None = None,
    top_n: int = 10,
) -> Path:
    """Render a report and write it to ``path``; returns the path.

    The format is inferred from the file extension when ``fmt`` is not given.
    """
    path = Path(path)
    if fmt is None:
        fmt = path.suffix.lstrip(".") or "markdown"
    content = generate_report(result, fmt=fmt, objective=objective, top_n=top_n)
    path.write_text(content, encoding="utf-8")
    return path
