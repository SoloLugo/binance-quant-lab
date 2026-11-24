"""
Pruebas reales (sin mocks sustitutivos) para Fase 5 — Probability Engine
y el adaptador determinista BenterQuantBrain.
"""
import math

import numpy as np
import pandas as pd
import pytest

from brain import BenterQuantBrain
from contracts import LabelSpec
from feature_engine import FeatureEngine
from label_engine import LabelEngine
from probability_engine import (
    MODEL_HIERARCHY,
    BaseRateClassifier,
    ProbabilityEngine,
    compute_classification_metrics,
)
from splits import TemporalDatasetPartitioner


def _build_real_dataset(sample_ohlcv_and_funding, direction: str = "LONG"):
    df_btc, _df_eth, df_funding = sample_ohlcv_and_funding
    feat_engine = FeatureEngine()
    features = feat_engine.build_features(df_btc, df_funding)
    label_engine = LabelEngine(
        LabelSpec(tp_mult=1.5, sl_mult=1.0, horizon_bars=24, entry_mode="next_open")
    )
    outcomes = label_engine.compute_outcomes(df_btc, features["atr_pct"], direction=direction)

    feature_cols = [
        c for c in features.columns if c.startswith(("ta_", "sm_", "deriv_"))
    ]
    ds = features.copy()
    ds["target"] = outcomes["target_tp_before_sl"]
    ds["realized_r_gross"] = outcomes["realized_r_gross"]
    ds["exit_bars"] = outcomes["exit_bars"]
    ds = ds.dropna(subset=feature_cols + ["target"]).copy()
    ds["target"] = ds["target"].astype(int)
    return ds, feature_cols


def test_base_rate_classifier_predicts_empirical_frequency():
    X_tr = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_tr = np.array([1, 0, 1, 0, 0])
    clf = BaseRateClassifier().fit(X_tr, y_tr)

    X_te = pd.DataFrame({"f1": [10.0, 20.0, 30.0]})
    proba = clf.predict_proba(X_te)
    assert proba.shape == (3, 2)
    assert np.allclose(proba[:, 1], 0.4)
    assert np.allclose(proba[:, 0], 0.6)
    assert np.array_equal(clf.predict(X_te), np.array([0, 0, 0]))


def test_compute_classification_metrics_complete_and_single_class():
    y_true = np.array([0, 0, 1, 1])
    p_pred = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_classification_metrics(y_true, p_pred, threshold=0.5)
    expected_keys = {
        "roc_auc",
        "pr_auc",
        "brier_score",
        "brier_skill_score",
        "log_loss",
        "accuracy",
        "confusion_matrix",
        "positive_rate",
        "n_samples",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["n_samples"] == 4
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    # Brier = mean([0.01, 0.16, 0.16, 0.01]) = 0.085; brier_base = 0.25 -> BSS = 1 - 0.085/0.25 = 0.66
    assert metrics["brier_score"] == pytest.approx(0.085)
    assert metrics["brier_skill_score"] == pytest.approx(0.66)

    # Clase única -> roc_auc debe ser NaN sin lanzar excepción
    single_metrics = compute_classification_metrics(np.array([0, 0, 0, 0]), p_pred)
    assert math.isnan(single_metrics["roc_auc"])
    assert math.isnan(single_metrics["pr_auc"])
    assert single_metrics["n_samples"] == 4
    assert single_metrics["confusion_matrix"] == {"tn": 2, "fp": 2, "fn": 0, "tp": 0}


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_probability_engine_five_models_val_only_and_sealed_locked(
    sample_ohlcv_and_funding, direction
):
    ds, feature_cols = _build_real_dataset(sample_ohlcv_and_funding, direction=direction)
    partitioner = TemporalDatasetPartitioner(
        ds,
        purge_bars=24,
        embargo_bars=24,
        horizon_bars=24,
        sealed_test_ratio=0.15,
    )
    # Test Final Sellado permanece estrictamente bloqueado durante toda la selección
    with pytest.raises(PermissionError):
        partitioner.get_sealed_test()

    df_train, _df_calib, df_val = partitioner.get_dev_split(
        train_ratio=0.65, calib_ratio=0.15
    )

    engine = ProbabilityEngine(direction=direction, random_state=42)
    result = engine.train_and_compare(
        df_train=df_train,
        y_train=df_train["target"],
        df_val=df_val,
        y_val=df_val["target"],
        feature_cols=feature_cols,
    )

    assert result["direction"] == direction
    assert set(result["models"].keys()) == set(MODEL_HIERARCHY)
    assert set(result["val_metrics"].keys()) == set(MODEL_HIERARCHY)
    assert result["best_model_name"] in MODEL_HIERARCHY

    for name in MODEL_HIERARCHY:
        m = result["val_metrics"][name]
        assert m["n_samples"] == len(df_val)
        assert 0.0 <= m["brier_score"] <= 1.0
        assert m["log_loss"] > 0.0
        assert 0.0 <= m["accuracy"] <= 1.0
        cm = m["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(df_val)

    # Verificar que la selección respeta (brier_score asc, log_loss asc, -roc_auc asc)
    def _key(name: str):
        m = result["val_metrics"][name]
        auc = m["roc_auc"]
        return (m["brier_score"], m["log_loss"], float("inf") if math.isnan(auc) else -auc)

    expected_best = min(MODEL_HIERARCHY, key=_key)
    assert result["best_model_name"] == expected_best

    # Verificar que el Test Final Sellado jamás fue accedido
    assert partitioner._sealed_access_log == []


def test_probability_engine_determinism_and_overlap_guard(sample_ohlcv_and_funding):
    ds, feature_cols = _build_real_dataset(sample_ohlcv_and_funding, direction="LONG")
    partitioner = TemporalDatasetPartitioner(
        ds, purge_bars=24, embargo_bars=24, horizon_bars=24, sealed_test_ratio=0.15
    )
    df_train, _df_calib, df_val = partitioner.get_dev_split()

    eng1 = ProbabilityEngine(direction="LONG", random_state=42)
    res1 = eng1.train_and_compare(
        df_train, df_train["target"], df_val, df_val["target"], feature_cols
    )

    eng2 = ProbabilityEngine(direction="LONG", random_state=42)
    res2 = eng2.train_and_compare(
        df_train, df_train["target"], df_val, df_val["target"], feature_cols
    )

    assert res1["best_model_name"] == res2["best_model_name"]
    for name in MODEL_HIERARCHY:
        assert np.allclose(res1["val_predictions"][name], res2["val_predictions"][name], atol=0.0)
        assert res1["val_metrics"][name]["brier_score"] == res2["val_metrics"][name]["brier_score"]

    # Solapamiento temporal entre train y val debe lanzar ValueError
    with pytest.raises(ValueError, match="Solapamiento temporal"):
        eng1.train_and_compare(
            df_train, df_train["target"], df_train.iloc[-20:], df_train["target"].iloc[-20:], feature_cols
        )


def test_brain_adapter_determinism_and_temporal_validation(sample_ohlcv_and_funding):
    ds, feature_cols = _build_real_dataset(sample_ohlcv_and_funding, direction="LONG")
    partitioner = TemporalDatasetPartitioner(
        ds, purge_bars=24, embargo_bars=24, horizon_bars=24, sealed_test_ratio=0.15
    )
    df_train, df_calib, df_val = partitioner.get_dev_split()

    brain1 = BenterQuantBrain(feature_cols=feature_cols, direction="LONG", random_state=42)
    brain1.base_clf.set_params(n_estimators=25, min_child_samples=10)
    brain1.fit_and_calibrate(
        df_train, df_train["target"], df_calib, df_calib["target"], purge_bars=24, enforce_purge=True
    )
    p1 = brain1.predict_calibrated_proba(df_val)

    brain2 = BenterQuantBrain(feature_cols=feature_cols, direction="LONG", random_state=42)
    brain2.base_clf.set_params(n_estimators=25, min_child_samples=10)
    brain2.fit_and_calibrate(
        df_train, df_train["target"], df_calib, df_calib["target"], purge_bars=24, enforce_purge=True
    )
    p2 = brain2.predict_calibrated_proba(df_val)

    assert np.array_equal(p1, p2)

    # Solapamiento temporal siempre debe ser rechazado
    with pytest.raises(ValueError, match="Solapamiento temporal"):
        brain1.fit_and_calibrate(
            df_train, df_train["target"], df_train.iloc[-10:], df_train["target"].iloc[-10:]
        )

    # Con enforce_purge=True, una brecha de 1 barra (menor que purge_bars=24) debe fallar
    n = len(ds)
    split_idx = int(n * 0.7)
    with pytest.raises(ValueError, match="Brecha temporal insuficiente"):
        brain1.fit_and_calibrate(
            ds.iloc[:split_idx],
            ds["target"].iloc[:split_idx],
            ds.iloc[split_idx:],
            ds["target"].iloc[split_idx:],
            purge_bars=24,
            enforce_purge=True,
        )
