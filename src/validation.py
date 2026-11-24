"""
Validation & Robustness Engine (Fase 9) — Validación Walk-Forward estricta,
Bootstrap CI, Monte Carlo Drawdown, Deflated Sharpe Ratio y Análisis por Régimen.

Reglas de integridad (Plan Maestro §11):
- Ejecuta EXCLUSIVAMENTE sobre `partitioner.get_walk_forward_folds(...)`.
- JAMÁS invoca `partitioner.get_sealed_test(...)`.
- Estima `TimeoutEconomics` exclusivamente sobre `train + calib` en cada fold.
- Garantiza determinismo total con `random_state` fijo.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from backtest_engine import BacktestConfig, BacktestEngine, TRADE_COLUMNS
from calibration_engine import ProbabilityCalibrator, compute_calibration_metrics
from contracts import CostModel
from ev_engine import EVEngine, estimate_timeout_economics
from probability_engine import ProbabilityEngine, compute_classification_metrics
from splits import TemporalDatasetPartitioner


EULER_MASCHERONI: float = 0.5772156649015329


def _clean_finite_1d(values: Any) -> np.ndarray:
    """Convierte a np.ndarray 1D float64 filtrando NaN e infinitos."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_expectancy_ci(
    realized_r: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Estima el intervalo de confianza bootstrap no paramétrico para la esperanza
    en unidades R (`E[R]`) y la probabilidad bootstrap de `E[R] > 0`.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples debe ser >= 1, recibido {n_resamples}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha debe estar en (0, 1), recibido {alpha}.")

    r_arr = _clean_finite_1d(realized_r)
    n_trades = int(r_arr.size)
    if n_trades == 0:
        return {
            "mean_r": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "prob_positive_ev": 0.0,
            "n_trades": 0,
        }

    rng = np.random.default_rng(int(random_state))
    indices = rng.integers(0, n_trades, size=(int(n_resamples), n_trades))
    boot_means = np.mean(r_arr[indices], axis=1)

    return {
        "mean_r": float(np.mean(r_arr)),
        "ci_low": float(np.quantile(boot_means, alpha / 2.0)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha / 2.0)),
        "prob_positive_ev": float(np.mean(boot_means > 0.0)),
        "n_trades": n_trades,
    }


def monte_carlo_drawdown(
    realized_r: np.ndarray,
    risk_per_trade: float = 0.01,
    n_simulations: int = 1000,
    ruin_threshold_pct: float = 20.0,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Simula trayectorias de equity mediante Monte Carlo (remuestreo bootstrap de R-múltiplos
    con capitalización compuesta) para cuantificar distribución de Max Drawdown y
    probabilidad de ruina por encima de `ruin_threshold_pct`.
    """
    if not (0.0 < risk_per_trade <= 1.0):
        raise ValueError(f"risk_per_trade debe estar en (0, 1], recibido {risk_per_trade}.")
    if n_simulations < 1:
        raise ValueError(f"n_simulations debe ser >= 1, recibido {n_simulations}.")
    if ruin_threshold_pct <= 0.0:
        raise ValueError(
            f"ruin_threshold_pct debe ser > 0, recibido {ruin_threshold_pct}."
        )

    r_arr = _clean_finite_1d(realized_r)
    n_trades = int(r_arr.size)
    if n_trades == 0:
        return {
            "median_max_dd_pct": 0.0,
            "p95_max_dd_pct": 0.0,
            "p99_max_dd_pct": 0.0,
            "prob_ruin_pct": 0.0,
        }

    rng = np.random.default_rng(int(random_state))
    indices = rng.integers(0, n_trades, size=(int(n_simulations), n_trades))
    sampled_r = r_arr[indices]

    step_factors = np.maximum(1.0 + sampled_r * float(risk_per_trade), 0.0)
    equity_paths = np.cumprod(step_factors, axis=1)
    equity_with_init = np.concatenate(
        [np.ones((int(n_simulations), 1), dtype=float), equity_paths],
        axis=1,
    )

    running_peaks = np.maximum.accumulate(equity_with_init, axis=1)
    safe_peaks = np.where(running_peaks > 0.0, running_peaks, 1.0)
    drawdowns_pct = ((running_peaks - equity_with_init) / safe_peaks) * 100.0
    max_dds_pct = np.max(drawdowns_pct, axis=1)

    return {
        "median_max_dd_pct": float(np.percentile(max_dds_pct, 50.0)),
        "p95_max_dd_pct": float(np.percentile(max_dds_pct, 95.0)),
        "p99_max_dd_pct": float(np.percentile(max_dds_pct, 99.0)),
        "prob_ruin_pct": float(np.mean(max_dds_pct >= float(ruin_threshold_pct)) * 100.0),
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    benchmark_sharpe: float = 0.0,
) -> Dict[str, float]:
    """
    Calcula el Deflated Sharpe Ratio (Bailey & López de Prado, 2014), ajustando
    el Sharpe observado por sesgo de selección múltiple (`n_trials`), asimetría
    (skewness) y curtosis (kurtosis) de los retornos.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials debe ser >= 1, recibido {n_trials}.")

    r_arr = _clean_finite_1d(returns)
    n_obs = int(r_arr.size)
    if n_obs < 2:
        return {
            "observed_sharpe": 0.0,
            "expected_max_sharpe": float(benchmark_sharpe),
            "dsr_probability": 0.0,
            "n_trials": int(n_trials),
            "n_obs": n_obs,
        }

    std_ret = float(np.std(r_arr, ddof=1))
    if std_ret <= 1e-12:
        return {
            "observed_sharpe": 0.0,
            "expected_max_sharpe": float(benchmark_sharpe),
            "dsr_probability": 0.0,
            "n_trials": int(n_trials),
            "n_obs": n_obs,
        }

    observed_sharpe = float(np.mean(r_arr) / std_ret)

    skewness = float(stats.skew(r_arr, bias=True)) if n_obs >= 3 else 0.0
    kurtosis_raw = float(stats.kurtosis(r_arr, fisher=False, bias=True)) if n_obs >= 4 else 3.0
    if not np.isfinite(skewness):
        skewness = 0.0
    if not np.isfinite(kurtosis_raw):
        kurtosis_raw = 3.0

    # Varianza asintótica del estimador de Sharpe (Mertens 2002 / Bailey & LdP 2014)
    sr_var_num = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis_raw - 1.0) / 4.0) * (observed_sharpe ** 2)
    )
    sr_std = math.sqrt(max(sr_var_num, 1e-12) / float(n_obs - 1))

    # Umbral de Sharpe máximo esperado bajo H0 dado n_trials
    if n_trials == 1:
        expected_max_sharpe = float(benchmark_sharpe)
    else:
        n_t = float(n_trials)
        z1 = float(stats.norm.ppf(1.0 - 1.0 / n_t))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_t * math.e)))
        e_max_z = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
        # Escala bajo H0: sigma_SR0 = 1 / sqrt(T - 1)
        sigma_sr0 = 1.0 / math.sqrt(float(n_obs - 1))
        expected_max_sharpe = float(benchmark_sharpe) + sigma_sr0 * e_max_z

    z_stat = (observed_sharpe - expected_max_sharpe) / max(sr_std, 1e-12)
    dsr_probability = float(stats.norm.cdf(z_stat))

    return {
        "observed_sharpe": float(observed_sharpe),
        "expected_max_sharpe": float(expected_max_sharpe),
        "dsr_probability": float(dsr_probability),
        "n_trials": int(n_trials),
        "n_obs": n_obs,
    }


def analyze_by_regime(
    trades_df: pd.DataFrame,
    regime_series: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Desglosa número de trades, win rate, expectancy en R, net PnL y profit factor
    por cada régimen de mercado presente en `regime_series`.
    """
    if trades_df is None or trades_df.empty:
        return {}

    if not isinstance(regime_series, pd.Series):
        regime_series = pd.Series(regime_series)

    # Alineación robusta de regímenes contra trades_df
    if "signal_time" in trades_df.columns and isinstance(regime_series.index, pd.DatetimeIndex):
        aligned_regimes = regime_series.reindex(trades_df["signal_time"]).to_numpy(dtype=object)
    elif len(regime_series) == len(trades_df):
        aligned_regimes = regime_series.to_numpy(dtype=object)
    elif "signal_idx" in trades_df.columns and len(regime_series) > 0:
        sig_idx = trades_df["signal_idx"].astype(int).to_numpy()
        aligned_regimes = np.array(
            [
                regime_series.iloc[idx] if 0 <= idx < len(regime_series) else None
                for idx in sig_idx
            ],
            dtype=object,
        )
    else:
        aligned_regimes = regime_series.reindex(trades_df.index).to_numpy(dtype=object)

    reg_ser = pd.Series(aligned_regimes, index=trades_df.index)
    valid_labels = sorted({str(val) for val in reg_ser.dropna().unique()})

    r_col = "realized_r_net" if "realized_r_net" in trades_df.columns else "realized_r_gross"
    pnl_col = "net_pnl_usdt" if "net_pnl_usdt" in trades_df.columns else r_col

    breakdown: Dict[str, Dict[str, Any]] = {}
    for label in valid_labels:
        mask = reg_ser.notna() & (reg_ser.astype(str) == label)
        sub = trades_df.loc[mask]
        if sub.empty:
            continue

        r_vals = pd.to_numeric(sub[r_col], errors="coerce").dropna().to_numpy(dtype=float)
        pnl_vals = pd.to_numeric(sub[pnl_col], errors="coerce").dropna().to_numpy(dtype=float)

        n_trades = int(len(sub))
        win_rate_pct = float(np.mean(r_vals > 0.0) * 100.0) if r_vals.size > 0 else 0.0
        expectancy_r = float(np.mean(r_vals)) if r_vals.size > 0 else 0.0
        net_pnl = float(np.sum(pnl_vals)) if pnl_vals.size > 0 else 0.0

        gross_wins = float(np.sum(pnl_vals[pnl_vals > 0.0])) if pnl_vals.size > 0 else 0.0
        gross_losses = float(np.abs(np.sum(pnl_vals[pnl_vals < 0.0]))) if pnl_vals.size > 0 else 0.0
        if gross_losses > 0.0:
            profit_factor = float(gross_wins / gross_losses)
        elif gross_wins > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        breakdown[label] = {
            "n_trades": n_trades,
            "win_rate_pct": win_rate_pct,
            "expectancy_r": expectancy_r,
            "net_pnl": net_pnl,
            "profit_factor": profit_factor,
        }

    return breakdown


def _safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


def _safe_nanstd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.std(finite, ddof=0)) if finite.size > 0 else float("nan")


def run_walk_forward_validation(
    df_ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    partitioner: TemporalDatasetPartitioner,
    feature_cols: Sequence[str],
    direction: str = "LONG",
    model_name: str = "lightgbm",
    calibration_method: str = "isotonic",
    cost_model: CostModel = CostModel(),
    ev_min_threshold: float = 0.12,
    n_splits: int = 3,
    random_state: int = 42,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon_bars: int = 24,
    risk_per_trade_pct: float = 0.01,
    initial_balance: float = 10000.0,
    df_funding: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Ejecuta validación Walk-Forward purgada y con embargo EXCLUSIVAMENTE sobre
    `partitioner.get_walk_forward_folds(n_splits=n_splits)`.

    JAMÁS accede al Test Final Sellado (`get_sealed_test`).
    """
    direction_upper = direction.upper().strip()
    if direction_upper not in ("LONG", "SHORT"):
        raise ValueError(f"direction inválido: '{direction}'. Debe ser 'LONG' o 'SHORT'.")

    if "atr_pct" not in features.columns:
        raise ValueError("features debe contener la columna 'atr_pct'.")
    if "target_tp_before_sl" not in outcomes.columns:
        raise ValueError("outcomes debe contener la columna 'target_tp_before_sl'.")

    requested_cols = list(feature_cols)
    if not requested_cols:
        if model_name == "base_rate":
            engine_cols = ["atr_pct"]
        else:
            raise ValueError("feature_cols no puede estar vacío para modelos con features.")
    else:
        missing = [c for c in requested_cols if c not in features.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en features: {missing}")
        engine_cols = requested_cols

    # EXCLUSIVAMENTE sobre el bloque de desarrollo (jamás toca get_sealed_test)
    folds = partitioner.get_walk_forward_folds(n_splits=n_splits)

    ev_engine = EVEngine(
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        cost_model=cost_model,
        ev_min_threshold=ev_min_threshold,
        direction=direction_upper,
    )
    bt_config = BacktestConfig(
        initial_balance=initial_balance,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        horizon_bars=horizon_bars,
        risk_per_trade_pct=risk_per_trade_pct,
        cost_model=cost_model,
    )
    bt_engine = BacktestEngine(config=bt_config)

    fold_metrics: List[Dict[str, Any]] = []
    fold_artifacts: List[Dict[str, Any]] = []
    oos_trades_list: List[pd.DataFrame] = []

    briers: List[float] = []
    log_losses: List[float] = []
    aucs: List[float] = []
    eces: List[float] = []
    positive_ev_folds = 0

    for fold_idx, (df_tr_part, df_ca_part, df_va_part) in enumerate(folds):
        tr_idx = df_tr_part.index
        ca_idx = df_ca_part.index
        va_idx = df_va_part.index

        # 1. Preparación supervisada limpia (sin NaNs) en train, calib y val
        X_tr_raw = features.reindex(tr_idx)[engine_cols]
        y_tr_raw = outcomes.reindex(tr_idx)["target_tp_before_sl"]
        tr_mask = X_tr_raw.notna().all(axis=1) & y_tr_raw.notna()
        X_tr = X_tr_raw.loc[tr_mask]
        y_tr = y_tr_raw.loc[tr_mask].astype(int)

        X_ca_raw = features.reindex(ca_idx)[engine_cols]
        y_ca_raw = outcomes.reindex(ca_idx)["target_tp_before_sl"]
        ca_mask = X_ca_raw.notna().all(axis=1) & y_ca_raw.notna()
        X_ca = X_ca_raw.loc[ca_mask]
        y_ca = y_ca_raw.loc[ca_mask].astype(int)

        X_va_raw = features.reindex(va_idx)[engine_cols]
        y_va_raw = outcomes.reindex(va_idx)["target_tp_before_sl"]
        va_mask = X_va_raw.notna().all(axis=1) & y_va_raw.notna()
        X_va = X_va_raw.loc[va_mask]
        y_va = y_va_raw.loc[va_mask].astype(int)

        if X_tr.empty or X_ca.empty or X_va.empty:
            raise ValueError(
                f"Fold {fold_idx} sin filas válidas tras filtrar NaNs: "
                f"train={len(X_tr)}, calib={len(X_ca)}, val={len(X_va)}."
            )

        # 2. Entrenamiento del modelo en train y predicción cruda en calib
        prob_engine = ProbabilityEngine(
            direction=direction_upper,
            random_state=random_state,
        )
        train_res = prob_engine.train_and_compare(
            df_train=X_tr,
            y_train=y_tr,
            df_val=X_ca,
            y_val=y_ca,
            feature_cols=engine_cols,
            model_names=[model_name],
        )
        trained_model = train_res["models"][model_name]
        raw_calib_probs = train_res["val_predictions"][model_name]

        # 3. Estimación de TimeoutEconomics EXCLUSIVAMENTE sobre train + calib
        outcomes_tr_ca = pd.concat(
            [outcomes.reindex(tr_idx), outcomes.reindex(ca_idx)],
            axis=0,
        )
        timeout_econ = estimate_timeout_economics(outcomes_tr_ca)

        # 4. Calibración post-hoc exclusivamente en calib
        calibrator = ProbabilityCalibrator(
            method=calibration_method,
            random_state=random_state,
        )
        calibrator.fit(raw_calib_probs, y_ca.to_numpy(dtype=float))

        # 5. Evaluación probabilística y de calibración en val
        raw_val_probs = prob_engine.predict_proba(X_va, model_name=model_name)
        cal_val_probs = calibrator.predict(raw_val_probs)
        train_base_rate = float(np.mean(y_tr.to_numpy(dtype=float)))

        cls_metrics = compute_classification_metrics(
            y_true=y_va,
            p_pred=cal_val_probs,
            base_rate=train_base_rate,
        )
        cal_metrics = compute_calibration_metrics(
            y_true=y_va,
            p_pred=cal_val_probs,
            base_rate=train_base_rate,
        )

        # 6. Evaluación EV y BacktestEngine sobre la serie completa OHLCV de val
        df_ohlcv_val = df_ohlcv.reindex(va_idx)
        val_feat_valid_mask = X_va_raw.notna().all(axis=1)
        val_valid_idx = X_va_raw.index[val_feat_valid_mask]

        p_raw_full = pd.Series(np.nan, index=va_idx, dtype=float)
        p_cal_full = pd.Series(np.nan, index=va_idx, dtype=float)
        if len(val_valid_idx) > 0:
            raw_for_bt = prob_engine.predict_proba(
                X_va_raw.loc[val_valid_idx], model_name=model_name
            )
            cal_for_bt = calibrator.predict(raw_for_bt)
            p_raw_full.loc[val_valid_idx] = raw_for_bt
            p_cal_full.loc[val_valid_idx] = cal_for_bt

        atr_val_full = features.reindex(va_idx)["atr_pct"]
        ev_df = ev_engine.evaluate(
            p_calibrated=p_cal_full,
            atr_pct=atr_val_full,
            p_raw=p_raw_full,
            timeout_economics=timeout_econ,
        )

        signals_val = pd.DataFrame(
            {
                "side": np.where(ev_df["accepted"].to_numpy(dtype=bool), direction_upper, None),
                "atr_pct": atr_val_full.to_numpy(dtype=float),
                "p_calibrated": p_cal_full.to_numpy(dtype=float),
                "ev_net_r": ev_df["ev_net_r"].to_numpy(dtype=float),
            },
            index=va_idx,
        )

        bt_result = bt_engine.run(
            df_ohlcv=df_ohlcv_val,
            signals=signals_val,
            df_funding=df_funding,
        )

        fold_trades = bt_result.trades.copy()
        if not fold_trades.empty:
            fold_trades["fold"] = int(fold_idx)
            oos_trades_list.append(fold_trades)

        fold_brier = float(cls_metrics["brier_score"])
        fold_ll = float(cls_metrics["log_loss"])
        fold_auc = float(cls_metrics["roc_auc"])
        fold_ece = float(cal_metrics["ece"])
        fold_exp_r = float(bt_result.metrics["expectancy_r"])
        fold_n_trades = int(bt_result.metrics["total_trades"])

        briers.append(fold_brier)
        log_losses.append(fold_ll)
        aucs.append(fold_auc)
        eces.append(fold_ece)
        if fold_n_trades > 0 and fold_exp_r > 0.0:
            positive_ev_folds += 1

        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "train_start": tr_idx[0].isoformat(),
                "train_end": tr_idx[-1].isoformat(),
                "calib_start": ca_idx[0].isoformat(),
                "calib_end": ca_idx[-1].isoformat(),
                "val_start": va_idx[0].isoformat(),
                "val_end": va_idx[-1].isoformat(),
                "n_train": int(len(X_tr)),
                "n_calib": int(len(X_ca)),
                "n_val": int(len(X_va)),
                "brier_score": fold_brier,
                "log_loss": fold_ll,
                "roc_auc": fold_auc,
                "ece": fold_ece,
                "n_signals_accepted": int(ev_df["accepted"].sum()),
                "n_trades": fold_n_trades,
                "expectancy_r": fold_exp_r,
                "timeout_economics": timeout_econ.to_dict(),
                "classification_metrics": cls_metrics,
                "calibration_metrics": cal_metrics,
                "backtest_metrics": bt_result.metrics,
            }
        )

        fold_artifacts.append(
            {
                "fold": int(fold_idx),
                "model": trained_model,
                "calibrator": calibrator,
                "X_val": X_va,
                "y_val": y_va,
                "feature_cols": list(engine_cols),
            }
        )

    if oos_trades_list:
        oos_trades = pd.concat(oos_trades_list, ignore_index=True)
        all_realized_r = oos_trades["realized_r_net"].to_numpy(dtype=float)
    else:
        oos_trades = pd.DataFrame(columns=list(TRADE_COLUMNS) + ["fold"])
        all_realized_r = np.array([], dtype=float)

    total_trades = int(len(oos_trades))
    mean_expectancy_r = float(np.mean(all_realized_r)) if total_trades > 0 else 0.0
    positive_ev_fold_ratio = (
        float(positive_ev_folds / len(folds)) if len(folds) > 0 else 0.0
    )

    bootstrap_ci = bootstrap_expectancy_ci(
        realized_r=all_realized_r,
        random_state=random_state,
    )
    mc_dd = monte_carlo_drawdown(
        realized_r=all_realized_r,
        risk_per_trade=risk_per_trade_pct,
        random_state=random_state,
    )
    dsr = deflated_sharpe_ratio(
        returns=all_realized_r,
        n_trials=1,
    )

    summary = {
        "mean_brier": _safe_nanmean(briers),
        "std_brier": _safe_nanstd(briers),
        "mean_log_loss": _safe_nanmean(log_losses),
        "mean_auc": _safe_nanmean(aucs),
        "mean_ece": _safe_nanmean(eces),
        "total_trades": total_trades,
        "mean_expectancy_r": mean_expectancy_r,
        "positive_ev_fold_ratio": positive_ev_fold_ratio,
        "bootstrap_ci": bootstrap_ci,
        "monte_carlo_dd": mc_dd,
        "dsr": dsr,
    }

    return {
        "direction": direction_upper,
        "model_name": model_name,
        "calibration_method": calibration_method,
        "feature_cols": requested_cols,
        "n_splits": int(len(folds)),
        "random_state": int(random_state),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "oos_trades": oos_trades,
        "fold_artifacts": fold_artifacts,
    }
