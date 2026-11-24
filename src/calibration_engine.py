"""
Calibration Engine (Fase 6) — Calibración de probabilidades, tablas de fiabilidad,
diagnóstico por régimen y análisis de thresholds desacoplado de la economía.

Diseño conforme al Plan Maestro §8:
- Se eliminan el techo duro arbitrario de `0.75` y el recorte heurístico `[-4]`
  en la cola superior de la calibración isotónica; el dominio se limita únicamente
  a `[1e-6, 1 - 1e-6]` para estabilidad numérica de log-loss y log-odds.
- Separación explícita en tres capas independientes:
    1. Calibración de probabilidad (`ProbabilityCalibrator`, `compute_calibration_metrics`)
    2. Análisis de threshold de clasificación (`evaluate_thresholds`)
    3. Economía / EV / sizing (delegado al módulo económico, fuera de este módulo)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss


VALID_CALIBRATION_METHODS: Tuple[str, ...] = ("none", "platt", "isotonic")
PROB_EPS: float = 1e-6


def _clip_probs(probs: Sequence[float], eps: float = PROB_EPS) -> np.ndarray:
    arr = np.asarray(probs, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("El vector de probabilidades no puede estar vacío.")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("El vector de probabilidades contiene NaN o infinitos.")
    return np.clip(arr, eps, 1.0 - eps)


def _logit(probs: np.ndarray) -> np.ndarray:
    p = _clip_probs(probs, eps=PROB_EPS)
    return np.log(p / (1.0 - p))


class ProbabilityCalibrator:
    """
    Calibrador post-hoc de probabilidades crudas ajustado exclusivamente sobre
    un conjunto temporalmente separado `(raw_calib_probs, y_calib)`.

    Métodos soportados:
    - `"none"`     : Identidad recortada solo a `[1e-6, 1 - 1e-6]` para estabilidad de log-loss.
    - `"platt"`    : Platt Scaling (`LogisticRegression` sobre log-odds `logit(clip(p))`).
    - `"isotonic"` : `IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")`
                     SIN techo duro arbitrario de `0.75` ni recorte heurístico `[-4]`
                     (conforme al Plan Maestro §8).
    """

    def __init__(self, method: str = "isotonic", random_state: int = 42) -> None:
        normalized = method.lower().strip()
        if normalized not in VALID_CALIBRATION_METHODS:
            raise ValueError(
                f"Método de calibración inválido: '{method}'. "
                f"Valores permitidos: {VALID_CALIBRATION_METHODS}."
            )
        self.method: str = normalized
        self.random_state: int = int(random_state)
        self._model: Any = None
        self._constant_prob: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        raw_calib_probs: Sequence[float],
        y_calib: Sequence[Any],
    ) -> "ProbabilityCalibrator":
        """Ajusta el calibrador exclusivamente sobre `(raw_calib_probs, y_calib)`."""
        p_cal = _clip_probs(raw_calib_probs, eps=PROB_EPS)
        y_arr = np.asarray(y_calib, dtype=float).ravel()

        if y_arr.size != p_cal.size:
            raise ValueError(
                f"Longitudes incompatibles: len(raw_calib_probs)={p_cal.size} "
                f"vs len(y_calib)={y_arr.size}."
            )
        if np.isnan(y_arr).any():
            raise ValueError("y_calib contiene valores NaN.")

        self._model = None
        self._constant_prob = None

        if self.method == "none":
            self.is_fitted = True
            return self

        y_int = y_arr.astype(int)
        if self.method == "platt":
            if len(np.unique(y_int)) < 2:
                self._constant_prob = float(np.clip(np.mean(y_arr), PROB_EPS, 1.0 - PROB_EPS))
            else:
                logits = _logit(p_cal).reshape(-1, 1)
                lr = LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.random_state,
                )
                lr.fit(logits, y_int)
                self._model = lr
            self.is_fitted = True
            return self

        if self.method == "isotonic":
            # Plan Maestro §8: sin techo duro arbitrario 0.75 ni winsorización [-4]
            iso = IsotonicRegression(
                y_min=PROB_EPS,
                y_max=1.0 - PROB_EPS,
                out_of_bounds="clip",
            )
            iso.fit(p_cal, y_arr)
            self._model = iso
            self.is_fitted = True
            return self

        raise ValueError(f"Método de calibración no soportado: {self.method}")

    def predict(self, raw_probs: Sequence[float]) -> np.ndarray:
        """Transforma probabilidades crudas en probabilidades calibradas en `[1e-6, 1 - 1e-6]`."""
        if not self.is_fitted:
            raise RuntimeError(
                "ProbabilityCalibrator debe ajustarse con `fit` antes de predecir."
            )
        p_raw = _clip_probs(raw_probs, eps=PROB_EPS)

        if self.method == "none":
            return p_raw

        if self.method == "platt":
            if self._constant_prob is not None:
                return np.full(p_raw.size, self._constant_prob, dtype=float)
            logits = _logit(p_raw).reshape(-1, 1)
            probs = self._model.predict_proba(logits)[:, 1]
            return np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)

        if self.method == "isotonic":
            probs = np.asarray(self._model.predict(p_raw), dtype=float)
            return np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)

        raise ValueError(f"Método de calibración no soportado: {self.method}")

    def transform(self, raw_probs: Sequence[float]) -> np.ndarray:
        """Alias de `predict` para ergonomía compatible con transformadores sklearn."""
        return self.predict(raw_probs)


def compute_reliability_table(
    y_true: Sequence[Any],
    p_pred: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Construye una tabla de fiabilidad de `n_bins` intervalos equidistantes en `[0, 1]`
    con columnas: `bin_low`, `bin_high`, `count`, `mean_pred`, `empirical_freq`, `abs_gap`.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins debe ser >= 1, recibido {n_bins}.")

    y_arr = np.asarray(y_true, dtype=float).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        raise ValueError("y_true y p_pred deben ser no vacíos y de igual longitud.")
    if np.isnan(y_arr).any() or np.isnan(p_arr).any() or np.isinf(p_arr).any():
        raise ValueError("y_true o p_pred contienen valores NaN o infinitos.")

    p_unit = np.clip(p_arr, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for k in range(n_bins):
        low = float(edges[k])
        high = float(edges[k + 1])
        if k < n_bins - 1:
            mask = (p_unit >= low) & (p_unit < high)
        else:
            mask = (p_unit >= low) & (p_unit <= high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(p_unit[mask]))
            emp_freq = float(np.mean(y_arr[mask]))
            abs_gap = float(abs(mean_pred - emp_freq))
        else:
            mean_pred = float("nan")
            emp_freq = float("nan")
            abs_gap = float("nan")

        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "mean_pred": mean_pred,
                "empirical_freq": emp_freq,
                "abs_gap": abs_gap,
            }
        )

    return pd.DataFrame(rows)


def _serialize_reliability_table(rel_df: pd.DataFrame) -> list[Dict[str, Any]]:
    records = []
    for row in rel_df.to_dict(orient="records"):
        serialized = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                serialized[k] = None
            elif isinstance(v, (np.integer, int)):
                serialized[k] = int(v)
            else:
                serialized[k] = float(v)
        records.append(serialized)
    return records


def compute_calibration_metrics(
    y_true: Sequence[Any],
    p_pred: Sequence[float],
    n_bins: int = 10,
    base_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas integrales de calibración:
    - `brier_score`
    - `brier_skill_score`
    - `log_loss`
    - `ece` (Expected Calibration Error)
    - `mce` (Maximum Calibration Error)
    - `n_samples`
    - `reliability_table` (lista de dicts serializable a JSON)
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()

    if y_arr.size == 0 or y_arr.size != p_arr.size:
        raise ValueError("y_true y p_pred deben ser no vacíos y de igual longitud.")
    if np.isnan(p_arr).any() or np.isinf(p_arr).any():
        raise ValueError("p_pred contiene valores NaN o infinitos.")

    n_samples = int(y_arr.size)
    p_unit = np.clip(p_arr, 0.0, 1.0)
    p_clip = np.clip(p_arr, PROB_EPS, 1.0 - PROB_EPS)

    brier = float(brier_score_loss(y_arr, p_unit))
    ref_rate = float(np.mean(y_arr)) if base_rate is None else float(base_rate)
    brier_base = float(np.mean((y_arr - ref_rate) ** 2))
    if brier_base <= 0.0:
        bss = 0.0 if brier == 0.0 else float("nan")
    else:
        bss = float(1.0 - (brier / brier_base))

    ll = float(log_loss(y_arr, p_clip, labels=[0, 1]))

    rel_df = compute_reliability_table(y_arr, p_unit, n_bins=n_bins)
    non_empty = rel_df[rel_df["count"] > 0]
    if non_empty.empty:
        ece = 0.0
        mce = 0.0
    else:
        weights = non_empty["count"].to_numpy(dtype=float) / float(n_samples)
        gaps = non_empty["abs_gap"].to_numpy(dtype=float)
        ece = float(np.sum(weights * gaps))
        mce = float(np.max(gaps))

    return {
        "brier_score": brier,
        "brier_skill_score": bss,
        "log_loss": ll,
        "ece": ece,
        "mce": mce,
        "n_samples": n_samples,
        "reliability_table": _serialize_reliability_table(rel_df),
    }


def evaluate_calibration_by_regime(
    y_true: Sequence[Any],
    p_pred: Sequence[float],
    regime_series: Sequence[Any],
    n_bins: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa la calibración segmentada por régimen de mercado (ej. baja/media/alta
    volatilidad o tendencia/rango). Ignora etiquetas de régimen nulas (`NaN`/`None`).
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()
    regimes = pd.Series(regime_series).reset_index(drop=True)

    if not (y_arr.size == p_arr.size == len(regimes)):
        raise ValueError(
            f"Longitudes incompatibles: y_true={y_arr.size}, p_pred={p_arr.size}, "
            f"regime_series={len(regimes)}."
        )

    valid_regimes = sorted({str(r) for r in regimes.dropna().unique()})
    results: Dict[str, Dict[str, Any]] = {}

    for regime_label in valid_regimes:
        mask = regimes.notna() & (regimes.astype(str) == regime_label)
        mask_arr = mask.to_numpy(dtype=bool)
        if np.sum(mask_arr) == 0:
            continue
        results[regime_label] = compute_calibration_metrics(
            y_true=y_arr[mask_arr],
            p_pred=p_arr[mask_arr],
            n_bins=n_bins,
        )

    return results


def compare_calibrators(
    raw_calib_probs: Sequence[float],
    y_calib: Sequence[Any],
    raw_val_probs: Sequence[float],
    y_val: Sequence[Any],
    methods: Sequence[str] = ("none", "platt", "isotonic"),
    n_bins: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Ajusta cada calibrador exclusivamente sobre `(raw_calib_probs, y_calib)` y evalúa
    sobre `(raw_val_probs, y_val)`. Selecciona el mejor calibrador por menor
    `(brier_score, log_loss, ece)` en validación.
    """
    method_list = list(methods)
    if not method_list:
        raise ValueError("methods no puede ser una secuencia vacía.")

    y_cal_arr = np.asarray(y_calib, dtype=int).ravel()
    calib_base_rate = float(np.mean(y_cal_arr))

    calibrators: Dict[str, ProbabilityCalibrator] = {}
    val_metrics: Dict[str, Dict[str, Any]] = {}
    val_predictions: Dict[str, np.ndarray] = {}

    for method in method_list:
        cal = ProbabilityCalibrator(method=method, random_state=random_state)
        cal.fit(raw_calib_probs, y_calib)
        p_val_cal = cal.predict(raw_val_probs)

        calibrators[method] = cal
        val_predictions[method] = p_val_cal
        val_metrics[method] = compute_calibration_metrics(
            y_true=y_val,
            p_pred=p_val_cal,
            n_bins=n_bins,
            base_rate=calib_base_rate,
        )

    best_method = min(
        method_list,
        key=lambda m: (
            float(val_metrics[m]["brier_score"]),
            float(val_metrics[m]["log_loss"]),
            float(val_metrics[m]["ece"]),
        ),
    )

    return {
        "calibrators": calibrators,
        "val_metrics": val_metrics,
        "val_predictions": val_predictions,
        "best_method": best_method,
        "best_calibrator": calibrators[best_method],
    }


def evaluate_thresholds(
    y_true: Sequence[Any],
    p_cal: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Análisis de threshold puramente probabilístico/estadístico, estrictamente
    separado de la calibración y de la capa económica (Plan Maestro §8).
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    p_arr = np.clip(np.asarray(p_cal, dtype=float).ravel(), 0.0, 1.0)

    if y_arr.size == 0 or y_arr.size != p_arr.size:
        raise ValueError("y_true y p_cal deben ser no vacíos y de igual longitud.")

    if thresholds is None:
        thresh_values = [round(float(t), 4) for t in np.arange(0.10, 0.91, 0.05)]
    else:
        thresh_values = [float(t) for t in thresholds]
        if not thresh_values:
            raise ValueError("thresholds no puede estar vacío.")

    n = int(y_arr.size)
    rows = []
    for thr in thresh_values:
        pred_pos = p_arr >= thr
        y_hat = pred_pos.astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_hat, labels=[0, 1]).ravel()
        n_signals = int(tp + fp)
        signal_rate = float(n_signals / n)
        precision = float(tp / n_signals) if n_signals > 0 else float("nan")
        n_pos = int(tp + fn)
        recall = float(tp / n_pos) if n_pos > 0 else float("nan")
        if (
            not np.isnan(precision)
            and not np.isnan(recall)
            and (precision + recall) > 0.0
        ):
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")
        accuracy = float((tp + tn) / n)
        mean_prob_sel = float(np.mean(p_arr[pred_pos])) if n_signals > 0 else float("nan")

        rows.append(
            {
                "threshold": thr,
                "n_signals": n_signals,
                "signal_rate": signal_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "mean_prob_selected": mean_prob_sel,
            }
        )

    return pd.DataFrame(rows)
