"""
Pruebas reales (sin mocks sustitutivos) para Fase 6 — Calibration Engine.
Verifica calibradores none/platt/isotonic, ausencia del techo arbitrario 0.75,
cálculo exacto de ECE/MCE/reliability table, evaluación por régimen,
comparación sobre calib->val y análisis de thresholds desacoplado.
"""
import json

import numpy as np
import pandas as pd
import pytest

from calibration_engine import (
    PROB_EPS,
    ProbabilityCalibrator,
    compare_calibrators,
    compute_calibration_metrics,
    compute_reliability_table,
    evaluate_calibration_by_regime,
    evaluate_thresholds,
)
from contracts import LabelSpec
from feature_engine import FeatureEngine
from label_engine import LabelEngine
from probability_engine import ProbabilityEngine
from splits import TemporalDatasetPartitioner


def test_calibrator_none_platt_isotonic_and_no_arbitrary_075_cap():
    # Datos con señal fuerte en cola superior (frecuencia empírica > 0.90)
    raw_calib = np.linspace(0.05, 0.98, 200)
    rng = np.random.default_rng(42)
    y_calib = (rng.uniform(0.0, 1.0, size=200) < raw_calib).astype(int)
    # Asegurar cola superior claramente positiva (100% positivos en p >= 0.85)
    y_calib[raw_calib >= 0.85] = 1
    y_calib[raw_calib <= 0.15] = 0

    test_probs = np.array([0.0, 0.10, 0.50, 0.90, 0.99, 1.0])

    # 1. none: identidad recortada solo a [1e-6, 1 - 1e-6]
    cal_none = ProbabilityCalibrator(method="none").fit(raw_calib, y_calib)
    p_none = cal_none.predict(test_probs)
    assert p_none[0] == pytest.approx(PROB_EPS)
    assert p_none[-1] == pytest.approx(1.0 - PROB_EPS)
    assert p_none[2] == pytest.approx(0.50)

    # 2. platt: Platt scaling sobre log-odds
    cal_platt = ProbabilityCalibrator(method="platt").fit(raw_calib, y_calib)
    p_platt = cal_platt.predict(test_probs)
    assert np.all(p_platt >= PROB_EPS) and np.all(p_platt <= 1.0 - PROB_EPS)
    assert p_platt[-1] > p_platt[2] > p_platt[0]
    assert np.max(p_platt) > 0.75

    # 3. isotonic: SIN techo duro arbitrario de 0.75 ni recorte [-4]
    cal_iso = ProbabilityCalibrator(method="isotonic").fit(raw_calib, y_calib)
    p_iso = cal_iso.predict(test_probs)
    assert np.all(p_iso >= PROB_EPS) and np.all(p_iso <= 1.0 - PROB_EPS)
    assert p_iso[-1] > 0.90, (
        f"La calibración isotónica no debe tener techo arbitrario 0.75; obtenido {p_iso[-1]}."
    )
    assert np.all(np.diff(p_iso) >= -1e-12)  # monotonicidad no decreciente


def test_exact_reliability_table_ece_and_mce():
    # Construcción exacta con 2 bins conocidos en n_bins=2 ([0.0, 0.5) y [0.5, 1.0]):
    # Bin 0: 4 muestras con p=0.20, 1 positiva -> mean_pred=0.20, emp_freq=0.25, abs_gap=0.05
    # Bin 1: 6 muestras con p=0.80, 3 positivas -> mean_pred=0.80, emp_freq=0.50, abs_gap=0.30
    # ECE = (4/10)*0.05 + (6/10)*0.30 = 0.02 + 0.18 = 0.20
    # MCE = max(0.05, 0.30) = 0.30
    y_true = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    p_pred = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    rel = compute_reliability_table(y_true, p_pred, n_bins=2)
    assert list(rel.columns) == [
        "bin_low",
        "bin_high",
        "count",
        "mean_pred",
        "empirical_freq",
        "abs_gap",
    ]
    assert len(rel) == 2
    assert rel.loc[0, "count"] == 4
    assert rel.loc[0, "mean_pred"] == pytest.approx(0.20)
    assert rel.loc[0, "empirical_freq"] == pytest.approx(0.25)
    assert rel.loc[0, "abs_gap"] == pytest.approx(0.05)

    assert rel.loc[1, "count"] == 6
    assert rel.loc[1, "mean_pred"] == pytest.approx(0.80)
    assert rel.loc[1, "empirical_freq"] == pytest.approx(0.50)
    assert rel.loc[1, "abs_gap"] == pytest.approx(0.30)

    metrics = compute_calibration_metrics(y_true, p_pred, n_bins=2)
    assert metrics["ece"] == pytest.approx(0.20)
    assert metrics["mce"] == pytest.approx(0.30)
    assert metrics["n_samples"] == 10
    # Debe ser serializable a JSON sin errores
    serialized = json.dumps(metrics["reliability_table"])
    assert len(json.loads(serialized)) == 2


def test_evaluate_calibration_by_regime():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    p_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.8, 0.1, 0.2])
    regimes = pd.Series(["low_vol", "low_vol", "low_vol", "low_vol", "high_vol", "high_vol", "high_vol", "high_vol"])

    by_regime = evaluate_calibration_by_regime(y_true, p_pred, regimes, n_bins=4)
    assert set(by_regime.keys()) == {"high_vol", "low_vol"}
    assert by_regime["low_vol"]["n_samples"] == 4
    assert by_regime["high_vol"]["n_samples"] == 4
    assert by_regime["high_vol"]["brier_score"] < by_regime["low_vol"]["brier_score"]


def test_compare_calibrators_end_to_end_real_pipeline(sample_ohlcv_and_funding):
    df_btc, _df_eth, df_funding = sample_ohlcv_and_funding
    features = FeatureEngine().build_features(df_btc, df_funding)
    outcomes = LabelEngine(
        LabelSpec(tp_mult=1.5, sl_mult=1.0, horizon_bars=24, entry_mode="next_open")
    ).compute_outcomes(df_btc, features["atr_pct"], direction="LONG")

    feature_cols = [
        c for c in features.columns if c.startswith(("ta_", "sm_", "deriv_"))
    ]
    ds = features.copy()
    ds["target"] = outcomes["target_tp_before_sl"]
    ds = ds.dropna(subset=feature_cols + ["target"]).copy()
    ds["target"] = ds["target"].astype(int)

    partitioner = TemporalDatasetPartitioner(
        ds, purge_bars=24, embargo_bars=24, horizon_bars=24, sealed_test_ratio=0.15
    )
    df_train, df_calib, df_val = partitioner.get_dev_split(
        train_ratio=0.60, calib_ratio=0.20
    )

    prob_engine = ProbabilityEngine(direction="LONG", random_state=42)
    prob_res = prob_engine.train_and_compare(
        df_train=df_train,
        y_train=df_train["target"],
        df_val=df_val,
        y_val=df_val["target"],
        feature_cols=feature_cols,
        model_names=("logistic_regression", "lightgbm"),
    )

    best_name = prob_res["best_model_name"]
    raw_calib_probs = prob_engine.predict_proba(df_calib, model_name=best_name)
    raw_val_probs = prob_engine.predict_proba(df_val, model_name=best_name)

    comp = compare_calibrators(
        raw_calib_probs=raw_calib_probs,
        y_calib=df_calib["target"],
        raw_val_probs=raw_val_probs,
        y_val=df_val["target"],
        methods=("none", "platt", "isotonic"),
        n_bins=5,
    )

    assert set(comp["calibrators"].keys()) == {"none", "platt", "isotonic"}
    assert set(comp["val_metrics"].keys()) == {"none", "platt", "isotonic"}
    assert comp["best_method"] in {"none", "platt", "isotonic"}

    expected_best = min(
        ("none", "platt", "isotonic"),
        key=lambda m: (
            comp["val_metrics"][m]["brier_score"],
            comp["val_metrics"][m]["log_loss"],
            comp["val_metrics"][m]["ece"],
        ),
    )
    assert comp["best_method"] == expected_best

    # Evaluar calibración por régimen de volatilidad (atr_pct) en validación
    med_atr = float(df_val["atr_pct"].median())
    vol_regime = np.where(df_val["atr_pct"] >= med_atr, "high_vol", "low_vol")
    p_val_best = comp["val_predictions"][comp["best_method"]]
    regime_metrics = evaluate_calibration_by_regime(
        df_val["target"], p_val_best, vol_regime, n_bins=5
    )
    assert set(regime_metrics.keys()).issubset({"high_vol", "low_vol"})
    assert sum(m["n_samples"] for m in regime_metrics.values()) == len(df_val)

    # Test Final Sellado intacto
    assert partitioner._sealed_access_log == []


def test_evaluate_thresholds_separated_from_calibration_and_economics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    p_cal = np.array([0.15, 0.35, 0.55, 0.60, 0.80, 0.90])

    df_thr = evaluate_thresholds(y_true, p_cal, thresholds=[0.20, 0.50, 0.75, 0.95])
    assert list(df_thr["threshold"]) == [0.20, 0.50, 0.75, 0.95]
    # El número de señales debe ser monótono no creciente al subir el threshold
    assert np.all(np.diff(df_thr["n_signals"].to_numpy()) <= 0)

    # Verificar columnas puramente estadísticas (sin columnas económicas/EV/Kelly)
    assert "ev_in_r" not in df_thr.columns
    assert "risk_fraction_kelly" not in df_thr.columns

    # Para threshold=0.50: pred_pos = [F, F, T, T, T, T] -> tp=3, fp=1, tn=2, fn=0
    row_05 = df_thr.loc[df_thr["threshold"] == 0.50].iloc[0]
    assert row_05["n_signals"] == 4
    assert row_05["tp"] == 3
    assert row_05["fp"] == 1
    assert row_05["tn"] == 2
    assert row_05["fn"] == 0
    assert row_05["precision"] == pytest.approx(0.75)
    assert row_05["recall"] == pytest.approx(1.0)

    # Para threshold=0.95: 0 señales -> precision es NaN sin error
    row_95 = df_thr.loc[df_thr["threshold"] == 0.95].iloc[0]
    assert row_95["n_signals"] == 0
    assert np.isnan(row_95["precision"])
