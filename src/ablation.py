"""
Ablation Studies & Interpretability Engine (Fase 10) — Comparación progresiva
M0–M4 sobre Walk-Forward Validation e interpretabilidad determinista (§12).

Reglas de integridad:
- Evalúa M0–M4 sobre los mismos folds walk-forward y bajo las mismas hipótesis.
- Selecciona `best_model_name` EXCLUSIVAMENTE sobre métricas de validación walk-forward.
- JAMÁS accede al Test Final Sellado (`test_period="SEALED_NOT_TOUCHED"`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from contracts import CostModel
from experiment_tracker import ExperimentTracker
from splits import TemporalDatasetPartitioner
from validation import deflated_sharpe_ratio, run_walk_forward_validation


ABLATION_ORDER: Tuple[str, ...] = (
    "M0_Baseline",
    "M1_Technical",
    "M2_TA+Microstructure",
    "M3_TA+Micro+Derivatives",
    "M4_Full_Context",
)

INTERPRETABILITY_CAUSALITY_WARNING: str = (
    "ADVERTENCIA (Plan Maestro §12): La importancia de features (nativa o por "
    "permutación) mide únicamente asociación/contribución predictiva dentro del "
    "modelo evaluado y NO debe interpretarse como causalidad económica ni estructural."
)


def build_ablation_configs(
    feature_columns: Sequence[str],
    model_name: str = "lightgbm",
) -> Dict[str, Dict[str, Any]]:
    """
    Construye las 5 configuraciones progresivas del Plan Maestro §12:
    - `M0_Baseline`            : modelo `base_rate` (frecuencia histórica)
    - `M1_Technical`           : `ta_*`
    - `M2_TA+Microstructure`   : `ta_* + sm_*`
    - `M3_TA+Micro+Derivatives`: `ta_* + sm_* + deriv_*`
    - `M4_Full_Context`        : `ta_* + sm_* + deriv_* + ctx_*`
    """
    cols = list(feature_columns)
    ta_cols = [c for c in cols if c.startswith("ta_")]
    sm_cols = [c for c in cols if c.startswith("sm_")]
    deriv_cols = [c for c in cols if c.startswith("deriv_")]
    ctx_cols = [c for c in cols if c.startswith("ctx_")]

    if not ta_cols:
        raise ValueError(
            "No se encontraron columnas técnicas con prefijo 'ta_' en features."
        )

    return {
        "M0_Baseline": {
            "model_name": "base_rate",
            "feature_cols": [],
            "families": [],
        },
        "M1_Technical": {
            "model_name": model_name,
            "feature_cols": list(ta_cols),
            "families": ["ta"],
        },
        "M2_TA+Microstructure": {
            "model_name": model_name,
            "feature_cols": list(ta_cols + sm_cols),
            "families": ["ta", "sm"],
        },
        "M3_TA+Micro+Derivatives": {
            "model_name": model_name,
            "feature_cols": list(ta_cols + sm_cols + deriv_cols),
            "families": ["ta", "sm", "deriv"],
        },
        "M4_Full_Context": {
            "model_name": model_name,
            "feature_cols": list(ta_cols + sm_cols + deriv_cols + ctx_cols),
            "families": ["ta", "sm", "deriv", "ctx"],
        },
    }


def _selection_key(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Criterio de selección exclusivo sobre validación Walk-Forward:
    menor `mean_brier`, luego menor `mean_log_loss`, luego mayor `mean_auc`.
    """
    brier = float(summary.get("mean_brier", float("inf")))
    ll = float(summary.get("mean_log_loss", float("inf")))
    auc = float(summary.get("mean_auc", float("nan")))
    if math.isnan(brier):
        brier = float("inf")
    if math.isnan(ll):
        ll = float("inf")
    neg_auc = float("inf") if math.isnan(auc) else -auc
    return (brier, ll, neg_auc)


def compute_feature_importance(
    model: Any,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Extrae la importancia nativa de features de un modelo entrenado
    (árboles, LightGBM, modelos lineales o `Pipeline`).

    ADVERTENCIA (Plan Maestro §12): La importancia nativa mide asociación/contribución
    predictiva dentro del modelo y NO implica causalidad.
    """
    feat_list = list(feature_cols)
    if not feat_list:
        return pd.DataFrame(columns=["feature", "importance", "importance_norm", "rank"])

    estimator = model
    if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and model.steps:
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        raw_imp = np.asarray(estimator.feature_importances_, dtype=float).ravel()
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        raw_imp = np.mean(np.abs(coef), axis=0).ravel() if coef.ndim > 1 else np.abs(coef).ravel()
    else:
        raw_imp = np.zeros(len(feat_list), dtype=float)

    if len(raw_imp) != len(feat_list):
        raise ValueError(
            f"Longitud de importancias ({len(raw_imp)}) != len(feature_cols) ({len(feat_list)})."
        )

    total = float(np.sum(np.abs(raw_imp)))
    imp_norm = raw_imp / total if total > 0.0 else np.zeros_like(raw_imp)

    df = pd.DataFrame(
        {
            "feature": feat_list,
            "importance": raw_imp,
            "importance_norm": imp_norm,
        }
    )
    df = df.sort_values(by=["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def compute_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: Sequence[Any],
    feature_cols: Sequence[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula la importancia por permutación sobre el conjunto de validación usando
    `sklearn.inspection.permutation_importance` con `scoring="neg_brier_score"`.

    ADVERTENCIA (Plan Maestro §12): La importancia por permutación mide asociación/
    contribución predictiva fuera de muestra y NO debe interpretarse como causalidad.
    """
    feat_list = list(feature_cols)
    if not feat_list:
        return pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std", "rank"]
        )
    if X_val.empty:
        raise ValueError("X_val no puede estar vacío en compute_permutation_importance.")

    X_eval = X_val[feat_list]
    y_eval = np.asarray(y_val, dtype=int).ravel()
    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"Longitudes incompatibles: len(X_val)={len(X_eval)} vs len(y_val)={len(y_eval)}."
        )

    perm = permutation_importance(
        estimator=model,
        X=X_eval,
        y=y_eval,
        scoring="neg_brier_score",
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=1,
    )

    df = pd.DataFrame(
        {
            "feature": feat_list,
            "importance_mean": np.asarray(perm.importances_mean, dtype=float),
            "importance_std": np.asarray(perm.importances_std, dtype=float),
        }
    )
    df = df.sort_values(
        by=["importance_mean", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def compute_feature_stability(
    fold_importance_dfs: List[pd.DataFrame],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Calcula la estabilidad del ranking de importancia de features entre folds
    walk-forward: correlación de Spearman promedio entre pares de folds y
    solapamiento promedio en el top-k.

    ADVERTENCIA (Plan Maestro §12): La estabilidad de importancia refleja consistencia
    de asociación predictiva entre folds y NO implica causalidad.
    """
    if top_k < 1:
        raise ValueError(f"top_k debe ser >= 1, recibido {top_k}.")

    n_folds = len(fold_importance_dfs)
    if n_folds == 0:
        return {
            "mean_spearman_corr": float("nan"),
            "min_spearman_corr": float("nan"),
            "mean_top_k_overlap": 0.0,
            "top_k": int(top_k),
            "n_folds": 0,
            "consensus_top_features": [],
            "causality_warning": INTERPRETABILITY_CAUSALITY_WARNING,
        }

    # Determinar columna de importancia disponible y alinear features comunes
    imp_maps: List[pd.Series] = []
    rank_maps: List[pd.Series] = []
    top_sets: List[set[str]] = []

    for df in fold_importance_dfs:
        if "feature" not in df.columns:
            raise ValueError("Cada DataFrame de importancia debe contener la columna 'feature'.")
        if "importance_mean" in df.columns:
            val_col = "importance_mean"
        elif "importance" in df.columns:
            val_col = "importance"
        else:
            raise ValueError(
                "Cada DataFrame de importancia debe contener 'importance' o 'importance_mean'."
            )

        ordered = df.sort_values(by=[val_col, "feature"], ascending=[False, True]).reset_index(
            drop=True
        )
        imp_series = pd.Series(
            ordered[val_col].to_numpy(dtype=float),
            index=ordered["feature"].astype(str),
        )
        rank_series = pd.Series(
            np.arange(1, len(ordered) + 1, dtype=float),
            index=ordered["feature"].astype(str),
        )
        imp_maps.append(imp_series)
        rank_maps.append(rank_series)
        effective_k = min(int(top_k), len(ordered))
        top_sets.append(set(ordered["feature"].astype(str).iloc[:effective_k].tolist()))

    common_features = sorted(set.intersection(*(set(s.index) for s in imp_maps)))
    effective_k = min(int(top_k), len(common_features))

    if n_folds == 1:
        consensus = (
            rank_maps[0].loc[common_features].sort_values().index[:effective_k].tolist()
            if common_features
            else []
        )
        return {
            "mean_spearman_corr": 1.0,
            "min_spearman_corr": 1.0,
            "mean_top_k_overlap": 1.0 if effective_k > 0 else 0.0,
            "top_k": int(top_k),
            "n_folds": 1,
            "consensus_top_features": consensus,
            "causality_warning": INTERPRETABILITY_CAUSALITY_WARNING,
        }

    spearman_corrs: List[float] = []
    overlaps: List[float] = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            if len(common_features) >= 2:
                v_i = imp_maps[i].loc[common_features].to_numpy(dtype=float)
                v_j = imp_maps[j].loc[common_features].to_numpy(dtype=float)
                const_i = bool(np.allclose(v_i, v_i[0]))
                const_j = bool(np.allclose(v_j, v_j[0]))
                if const_i and const_j:
                    corr = 1.0
                elif const_i or const_j:
                    corr = 0.0
                else:
                    res = spearmanr(v_i, v_j)
                    corr = float(res.statistic) if np.isfinite(res.statistic) else 0.0
            else:
                corr = 1.0
            spearman_corrs.append(corr)

            if effective_k > 0:
                overlap_ratio = len(top_sets[i].intersection(top_sets[j])) / float(effective_k)
            else:
                overlap_ratio = 0.0
            overlaps.append(overlap_ratio)

    mean_ranks = pd.concat([r.reindex(common_features) for r in rank_maps], axis=1).mean(axis=1)
    consensus_top = mean_ranks.sort_values().index[:effective_k].tolist()

    return {
        "mean_spearman_corr": float(np.mean(spearman_corrs)),
        "min_spearman_corr": float(np.min(spearman_corrs)),
        "mean_top_k_overlap": float(np.mean(overlaps)),
        "top_k": int(top_k),
        "n_folds": n_folds,
        "consensus_top_features": consensus_top,
        "causality_warning": INTERPRETABILITY_CAUSALITY_WARNING,
    }


def run_ablation_study(
    df_ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    partitioner: TemporalDatasetPartitioner,
    direction: str = "LONG",
    model_name: str = "lightgbm",
    calibration_method: str = "isotonic",
    cost_model: CostModel = CostModel(),
    ev_min_threshold: float = 0.12,
    n_splits: int = 3,
    random_state: int = 42,
    tracker: Optional[ExperimentTracker] = None,
    dataset_version: str = "v1.0",
    feature_version: str = "v2.0",
    label_version: str = "v2.0",
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon_bars: int = 24,
    risk_per_trade_pct: float = 0.01,
    initial_balance: float = 10000.0,
    df_funding: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el estudio de ablación progresiva M0–M4 sobre los mismos folds
    Walk-Forward y selecciona el mejor candidato EXCLUSIVAMENTE en validación.

    JAMÁS toca el Test Final Sellado (`test_period="SEALED_NOT_TOUCHED"`).
    """
    direction_upper = direction.upper().strip()
    configs = build_ablation_configs(features.columns, model_name=model_name)
    n_candidates = len(configs)

    results: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []
    experiment_ids: List[str] = []

    for config_name in ABLATION_ORDER:
        cfg = configs[config_name]
        wf_res = run_walk_forward_validation(
            df_ohlcv=df_ohlcv,
            features=features,
            outcomes=outcomes,
            partitioner=partitioner,
            feature_cols=cfg["feature_cols"],
            direction=direction_upper,
            model_name=cfg["model_name"],
            calibration_method=calibration_method,
            cost_model=cost_model,
            ev_min_threshold=ev_min_threshold,
            n_splits=n_splits,
            random_state=random_state,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            horizon_bars=horizon_bars,
            risk_per_trade_pct=risk_per_trade_pct,
            initial_balance=initial_balance,
            df_funding=df_funding,
        )

        # Ajuste de DSR considerando los n_candidates probados en la ablación
        oos_trades = wf_res["oos_trades"]
        realized_r = (
            oos_trades["realized_r_net"].to_numpy(dtype=float)
            if not oos_trades.empty
            else np.array([], dtype=float)
        )
        wf_res["summary"]["dsr_ablation"] = deflated_sharpe_ratio(
            returns=realized_r,
            n_trials=n_candidates,
        )
        results[config_name] = wf_res

        summ = wf_res["summary"]
        comparison_rows.append(
            {
                "config": config_name,
                "model_name": cfg["model_name"],
                "n_features": len(cfg["feature_cols"]),
                "mean_brier": summ["mean_brier"],
                "std_brier": summ["std_brier"],
                "mean_log_loss": summ["mean_log_loss"],
                "mean_auc": summ["mean_auc"],
                "mean_ece": summ["mean_ece"],
                "total_trades": summ["total_trades"],
                "mean_expectancy_r": summ["mean_expectancy_r"],
                "positive_ev_fold_ratio": summ["positive_ev_fold_ratio"],
                "dsr_probability": summ["dsr_ablation"]["dsr_probability"],
            }
        )

        if tracker is not None:
            fold_metrics = wf_res["fold_metrics"]
            train_period = (
                f"{fold_metrics[0]['train_start']} -> {fold_metrics[-1]['train_end']}"
                if fold_metrics
                else "N/A"
            )
            val_period = (
                f"{fold_metrics[0]['val_start']} -> {fold_metrics[-1]['val_end']}"
                if fold_metrics
                else "N/A"
            )
            rec = tracker.log_experiment(
                hypothesis=f"Ablation {direction_upper} - {config_name} ({cfg['families']})",
                dataset_version=dataset_version,
                feature_version=feature_version,
                label_version=label_version,
                model_name=f"{config_name}:{cfg['model_name']}",
                parameters={
                    "direction": direction_upper,
                    "ablation_config": config_name,
                    "base_model": cfg["model_name"],
                    "calibration_method": calibration_method,
                    "ev_min_threshold": ev_min_threshold,
                    "n_splits": n_splits,
                    "n_features": len(cfg["feature_cols"]),
                    "feature_cols": cfg["feature_cols"],
                    "tp_mult": tp_mult,
                    "sl_mult": sl_mult,
                    "horizon_bars": horizon_bars,
                },
                train_period=train_period,
                val_period=val_period,
                test_period="SEALED_NOT_TOUCHED",
                cost_assumptions=cost_model.to_dict(),
                metrics=summ,
                random_seed=random_state,
                result="COMPLETED",
                conclusion=(
                    f"WF Validation ({n_splits} folds): Brier={summ['mean_brier']:.4f}, "
                    f"LogLoss={summ['mean_log_loss']:.4f}, AUC={summ['mean_auc']:.4f}, "
                    f"Trades={summ['total_trades']}, E[R]={summ['mean_expectancy_r']:.4f}."
                ),
            )
            experiment_ids.append(rec.experiment_id)

    best_model_name = min(
        ABLATION_ORDER,
        key=lambda name: _selection_key(results[name]["summary"]),
    )

    comparison_table = pd.DataFrame(comparison_rows)

    return {
        "direction": direction_upper,
        "configs": configs,
        "results": results,
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "best_feature_cols": list(configs[best_model_name]["feature_cols"]),
        "best_summary": results[best_model_name]["summary"],
        "experiment_ids": experiment_ids,
        "causality_warning": INTERPRETABILITY_CAUSALITY_WARNING,
    }
