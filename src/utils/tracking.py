
from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ROOT
from .io import write_jsonl

try:  # optional dependency
    import mlflow  # type: ignore

    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False


def _flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ",".join(map(str, obj))
    else:
        out[prefix] = obj
    return out


class RunTracker:
    """Context manager wrapping one experiment run."""

    def __init__(self, run_name: str, config: Optional[Any] = None,
                 experiment: str = "algosathi", backend: str = "auto"):
        self.run_name = run_name
        self.config = config
        self.experiment = experiment
        self.backend = "mlflow" if (backend in ("auto", "mlflow") and _HAS_MLFLOW) else "jsonl"
        self.run_id = f"{run_name}-{_dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.record: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_name": run_name,
            "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        self._active = None

    # -------------------------------------------------------------- lifecycle
    def __enter__(self) -> "RunTracker":
        if self.backend == "mlflow":
            uri = (self.config.get("tracking.tracking_uri") if self.config else None) \
                or "file:./experiments/mlruns"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.experiment)
            self._active = mlflow.start_run(run_name=self.run_name)
        if self.config is not None:
            self.log_params(_flatten("", self.config.as_dict(), {}))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.record["ended_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.record["status"] = "FAILED" if exc_type else "FINISHED"
        out = ROOT / "experiments" / "runs" / f"{self.experiment}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(self.record, ensure_ascii=False, default=str) + "\n")
        if self.backend == "mlflow":
            mlflow.end_run()

    # ------------------------------------------------------------------- logs
    def log_params(self, params: Dict[str, Any]) -> None:
        clean = {k: v for k, v in params.items() if v is not None}
        self.record["params"].update(clean)
        if self.backend == "mlflow":
            for key, value in clean.items():
                try:
                    mlflow.log_param(key[:250], str(value)[:250])
                except Exception:
                    pass

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and v == v}
        if step is None:
            self.record["metrics"].update(numeric)
        else:
            self.record.setdefault("history", []).append({"step": step, **numeric})
        if self.backend == "mlflow":
            for key, value in numeric.items():
                try:
                    mlflow.log_metric(key.replace("@", "_at_"), value, step=step)
                except Exception:
                    pass

    def log_artifact(self, path: str | Path) -> None:
        self.record["artifacts"].append(str(path))
        if self.backend == "mlflow":
            try:
                mlflow.log_artifact(str(path))
            except Exception:
                pass


def load_runs(experiment: str = "algosathi") -> list:
    """Read back all JSONL-tracked runs (used by the Streamlit results page)."""
    path = ROOT / "experiments" / "runs" / f"{experiment}.jsonl"
    if not path.exists():
        return []
    from .io import read_jsonl

    return read_jsonl(path)
