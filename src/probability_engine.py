"""
Probability Engine (Fase 5) — Jerarquía determinista de modelos de probabilidad
para estimación de P(TP antes que SL) en direcciones LONG y SHORT.

Reglas de integridad:
- Determinismo estricto (`random_state=seed`, `n_jobs=1`, LightGBM determinista).
- Entrenamiento sobre `df_train` y selección EXCLUSIVA sobre `df_val`.
- Jamás accede ni recibe el Test Final Sellado.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_HIERARCHY: Tuple[str, ...] = (
    "base_rate",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "lightgbm",
)

VALID_DIRECTIONS: Tuple[str, ...] = ("LONG", "SHORT")
PROB_EPS: float = 1e-6


class BaseRateClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador baseline que predice la frecuencia empírica de la clase positiva
    observada en `y_train`. Compatible con la interfaz `fit` / `predict_proba`.
    """

    def __init__(self) -> None:
        self.base_rate_: float = 0.5
        self.classes_: np.ndarray = np.array([0, 1], dtype=int)

    def fit(self, X: Any, y: Any) -> "BaseRateClassifier":
        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.size == 0:
            raise ValueError("y_train no puede estar vacío al entrenar BaseRateClassifier.")
        if np.isnan(y_arr).any():
            raise ValueError("y_train contiene valores NaN.")
        self.base_rate_ = float(np.clip(np.mean(y_arr), 0.0, 1.0))
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, self.base_rate_, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """
    Construye la jerarquía completa de 5 modelos (§7.1) con determinismo estricto
    (`random_state=seed`, `n_jobs=1`).
    """
    seed = int(random_state)
    return {
        "base_rate": BaseRateClassifier(),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.03,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            n_jobs=1,
        ),
    }


def _extract_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extrae de forma segura la probabilidad de la clase positiva (clase 1)."""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        return proba
    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes_arr = np.asarray(classes)
        matches = np.where(classes_arr == 1)[0]
        if len(matches) == 1:
            return proba[:, int(matches[0])]
        return np.zeros(len(X), dtype=float)
    return proba[:, 1]


def compute_classification_metrics(
    y_true: Sequence[Any],
    p_pred: Sequence[float],
    base_rate: Optional[float] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calcula métricas probabilísticas y de clasificación completas sobre `(y_true, p_pred)`:
    - `roc_auc` (`float("nan")` si `y_true` contiene una sola clase)
    - `pr_auc` (`average_precision_score`; `float("nan")` si `y_true` contiene una sola clase)
    - `brier_score`
    - `brier_skill_score` (`1 - brier / brier_base`)
    - `log_loss`
    - `accuracy`
    - `confusion_matrix` (`{"tn", "fp", "fn", "tp"}`)
    - `positive_rate`
    - `n_samples`
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    p_arr = np.asarray(p_pred, dtype=float).ravel()

    if y_arr.size == 0 or p_arr.size == 0:
        raise ValueError("y_true y p_pred no pueden estar vacíos.")
    if y_arr.size != p_arr.size:
        raise ValueError(
            f"Longitudes incompatibles: len(y_true)={y_arr.size} vs len(p_pred)={p_arr.size}."
        )
    if np.isnan(p_arr).any() or np.isinf(p_arr).any():
        raise ValueError("p_pred contiene valores NaN o infinitos.")

    n_samples = int(y_arr.size)
    positive_rate = float(np.mean(y_arr))
    unique_classes = np.unique(y_arr)

    p_unit = np.clip(p_arr, 0.0, 1.0)
    p_clip = np.clip(p_arr, PROB_EPS, 1.0 - PROB_EPS)

    if len(unique_classes) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_arr, p_unit))
        pr_auc = float(average_precision_score(y_arr, p_unit))

    brier = float(brier_score_loss(y_arr, p_unit))
    ref_rate = positive_rate if base_rate is None else float(base_rate)
    brier_base = float(np.mean((y_arr - ref_rate) ** 2))
    if brier_base <= 0.0:
        bss = 0.0 if brier == 0.0 else float("nan")
    else:
        bss = float(1.0 - (brier / brier_base))

    ll = float(log_loss(y_arr, p_clip, labels=[0, 1]))

    y_hat = (p_unit >= float(threshold)).astype(int)
    acc = float(accuracy_score(y_arr, y_hat))
    tn, fp, fn, tp = confusion_matrix(y_arr, y_hat, labels=[0, 1]).ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "brier_skill_score": bss,
        "log_loss": ll,
        "accuracy": acc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "positive_rate": positive_rate,
        "n_samples": n_samples,
    }


def _selection_sort_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    """Clave de selección: menor Brier, luego menor LogLoss, luego mayor ROC-AUC."""
    brier = float(metrics["brier_score"])
    ll = float(metrics["log_loss"])
    auc = float(metrics["roc_auc"])
    neg_auc = float("inf") if math.isnan(auc) else -auc
    return (brier, ll, neg_auc)


class ProbabilityEngine:
    """
    Motor de entrenamiento y comparación de modelos de probabilidad.
    Soporta ejecución independiente para `direction="LONG"` y `direction="SHORT"`.
    """

    def __init__(self, direction: str = "LONG", random_state: int = 42) -> None:
        direction_upper = direction.upper().strip()
        if direction_upper not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction inválido: '{direction}'. Valores permitidos: {VALID_DIRECTIONS}."
            )
        self.direction: str = direction_upper
        self.random_state: int = int(random_state)
        self.models_: Dict[str, Any] = {}
        self.val_metrics_: Dict[str, Dict[str, Any]] = {}
        self.best_model_name_: Optional[str] = None
        self.feature_cols_: List[str] = []

    def train_and_compare(
        self,
        df_train: pd.DataFrame,
        y_train: pd.Series,
        df_val: pd.DataFrame,
        y_val: pd.Series,
        feature_cols: Sequence[str],
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Entrena cada modelo de la jerarquía sobre `df_train` y evalúa EXCLUSIVAMENTE
        sobre `df_val` (jamás recibe ni toca el Test Final Sellado).

        Selecciona `best_model_name` según `(brier_score asc, log_loss asc, -roc_auc asc)`.
        """
        if df_train.empty or df_val.empty:
            raise ValueError("df_train y df_val no pueden estar vacíos.")

        if isinstance(df_train.index, pd.DatetimeIndex) and isinstance(
            df_val.index, pd.DatetimeIndex
        ):
            if df_val.index.min() <= df_train.index.max():
                raise ValueError(
                    "Solapamiento temporal detectado entre df_train y df_val: "
                    f"df_val.index.min() ({df_val.index.min()}) <= "
                    f"df_train.index.max() ({df_train.index.max()})."
                )

        feat_list = list(feature_cols)
        if not feat_list:
            raise ValueError("feature_cols no puede estar vacío.")

        missing_tr = [c for c in feat_list if c not in df_train.columns]
        missing_va = [c for c in feat_list if c not in df_val.columns]
        if missing_tr:
            raise ValueError(f"Columnas faltantes en df_train: {missing_tr}")
        if missing_va:
            raise ValueError(f"Columnas faltantes en df_val: {missing_va}")

        registry = build_model_registry(random_state=self.random_state)
        if model_names is None:
            selected_names = list(MODEL_HIERARCHY)
        else:
            selected_names = list(model_names)
            if not selected_names:
                raise ValueError("model_names no puede ser una secuencia vacía.")
            unknown = [m for m in selected_names if m not in registry]
            if unknown:
                raise ValueError(
                    f"Modelos desconocidos: {unknown}. Disponibles: {MODEL_HIERARCHY}."
                )

        X_tr = df_train[feat_list]
        y_tr = np.asarray(y_train, dtype=int).ravel()
        X_va = df_val[feat_list]
        y_va = np.asarray(y_val, dtype=int).ravel()

        train_base_rate = float(np.mean(y_tr))

        trained_models: Dict[str, Any] = {}
        val_metrics: Dict[str, Dict[str, Any]] = {}
        val_predictions: Dict[str, np.ndarray] = {}

        for name in selected_names:
            model = registry[name]
            model.fit(X_tr, y_tr)
            p_val = _extract_positive_proba(model, X_va)

            trained_models[name] = model
            val_predictions[name] = p_val
            val_metrics[name] = compute_classification_metrics(
                y_true=y_va,
                p_pred=p_val,
                base_rate=train_base_rate,
            )

        best_model_name = min(
            selected_names,
            key=lambda m: _selection_sort_key(val_metrics[m]),
        )

        self.models_ = trained_models
        self.val_metrics_ = val_metrics
        self.best_model_name_ = best_model_name
        self.feature_cols_ = feat_list

        return {
            "direction": self.direction,
            "random_state": self.random_state,
            "feature_cols": feat_list,
            "models": trained_models,
            "val_metrics": val_metrics,
            "val_predictions": val_predictions,
            "best_model_name": best_model_name,
            "best_model": trained_models[best_model_name],
        }

    def predict_proba(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Predice probabilidades crudas de la clase positiva con el modelo elegido (o el mejor)."""
        if not self.models_ or self.best_model_name_ is None:
            raise RuntimeError(
                "ProbabilityEngine debe entrenarse con `train_and_compare` antes de predecir."
            )
        target_name = model_name or self.best_model_name_
        if target_name not in self.models_:
            raise KeyError(
                f"Modelo '{target_name}' no entrenado. Disponibles: {list(self.models_.keys())}."
            )
        return _extract_positive_proba(self.models_[target_name], df[self.feature_cols_])
