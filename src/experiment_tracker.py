"""
Experiment Tracker (Fase 11) — Registro trazable y reproducible de experimentos.

Persiste cada experimento como un archivo JSON individual (`EXP-XXXX.json`) y
mantiene un índice secuencial append-only (`registry.jsonl`) conforme al Plan Maestro §13.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from contracts import ExperimentRecord


_EXP_ID_PATTERN = re.compile(r"^EXP-(\d+)$")


def _sanitize_for_json(obj: Any) -> Any:
    """Convierte recursivamente estructuras científicas a tipos JSON estrictos (sin NaN/Inf)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _sanitize_for_json(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    return str(obj)


def get_git_commit_short(cwd: Optional[Union[str, Path]] = None) -> str:
    """Obtiene el hash corto del commit actual con `git rev-parse --short HEAD` (fallback 'unknown')."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
            if commit:
                return commit
    except Exception:
        pass
    return "unknown"


class ExperimentTracker:
    """
    Gestor persistente de registros `ExperimentRecord` en `experiments_dir`.

    - Genera IDs secuenciales `EXP-0001`, `EXP-0002`, ... inspeccionando los archivos
      existentes en `experiments_dir`.
    - Guarda cada registro en `experiments_dir / f"{experiment_id}.json"`.
    - Anexa una línea JSON en `experiments_dir / "registry.jsonl"`.
    """

    def __init__(self, experiments_dir: Union[str, Path] = "data/experiments") -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.experiments_dir / "registry.jsonl"

    def _next_experiment_id(self) -> str:
        """Calcula el siguiente identificador secuencial `EXP-XXXX`."""
        max_num = 0

        for path in self.experiments_dir.glob("EXP-*.json"):
            match = _EXP_ID_PATTERN.match(path.stem)
            if match:
                max_num = max(max_num, int(match.group(1)))

        if self.registry_path.exists():
            with self.registry_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        rec = json.loads(stripped)
                        exp_id = str(rec.get("experiment_id", ""))
                        match = _EXP_ID_PATTERN.match(exp_id)
                        if match:
                            max_num = max(max_num, int(match.group(1)))
                    except json.JSONDecodeError:
                        continue

        return f"EXP-{max_num + 1:04d}"

    def log_experiment(
        self,
        hypothesis: str,
        dataset_version: str,
        feature_version: str,
        label_version: str,
        model_name: str,
        parameters: Dict[str, Any],
        train_period: str,
        val_period: str,
        test_period: str = "SEALED_NOT_TOUCHED",
        cost_assumptions: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
        result: str = "COMPLETED",
        conclusion: str = "",
        experiment_id: Optional[str] = None,
        git_commit: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> ExperimentRecord:
        """
        Registra y persiste un `ExperimentRecord` completo en JSON y `registry.jsonl`.
        """
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        exp_id = experiment_id if experiment_id is not None else self._next_experiment_id()
        commit = git_commit if git_commit is not None else get_git_commit_short()
        timestamp = (
            created_at
            if created_at is not None
            else datetime.now(timezone.utc).isoformat()
        )

        record = ExperimentRecord(
            experiment_id=str(exp_id),
            hypothesis=str(hypothesis),
            dataset_version=str(dataset_version),
            feature_version=str(feature_version),
            label_version=str(label_version),
            model_name=str(model_name),
            parameters=_sanitize_for_json(parameters or {}),
            train_period=str(train_period),
            val_period=str(val_period),
            test_period=str(test_period),
            cost_assumptions=_sanitize_for_json(cost_assumptions or {}),
            metrics=_sanitize_for_json(metrics or {}),
            git_commit=str(commit),
            random_seed=int(random_seed),
            result=str(result),
            conclusion=str(conclusion),
            created_at=str(timestamp),
        )

        payload = _sanitize_for_json(record.to_dict())

        record_path = self.experiments_dir / f"{record.experiment_id}.json"
        with record_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)

        with self.registry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

        return record

    def load_all(self) -> List[Dict[str, Any]]:
        """Carga todos los registros de experimentos en orden cronológico."""
        if self.registry_path.exists():
            records: List[Dict[str, Any]] = []
            with self.registry_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        records.append(json.loads(stripped))
            return records

        records = []
        for path in sorted(self.experiments_dir.glob("EXP-*.json")):
            with path.open("r", encoding="utf-8") as fh:
                records.append(json.load(fh))
        return records

    def get(self, experiment_id: str) -> Dict[str, Any]:
        """Recupera un experimento específico por su `experiment_id`."""
        record_path = self.experiments_dir / f"{experiment_id}.json"
        if record_path.exists():
            with record_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        if self.registry_path.exists():
            for rec in self.load_all():
                if rec.get("experiment_id") == experiment_id:
                    return rec

        raise FileNotFoundError(
            f"Experimento '{experiment_id}' no encontrado en {self.experiments_dir}."
        )
