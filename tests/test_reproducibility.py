"""
Suite de Reproducibilidad Bit-a-Bit (Sprint 2) — Binance Quant Lab.

Verifica determinismo estricto en ejecuciones independientes (sin mocks) sobre:
1. FeatureEngine + ContextEngine + LabelEngine (DataFrames bit-a-bit idénticos).
2. ProbabilityEngine (5 modelos: base_rate, logistic_regression, random_forest,
   gradient_boosting, lightgbm) con random_state=42.
3. compare_calibrators (none, platt, isotonic): probabilidades calibradas y métricas
   ECE, MCE, Brier, BSS y LogLoss exactamente iguales.
4. BenterQuantBrain (fit_and_calibrate): probabilidades calibradas, EV/Kelly y métricas
   de evaluación exactamente iguales.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from brain import BenterQuantBrain
from calibration_engine import VALID_CALIBRATION_METHODS, compare_calibrators
from context_engine import ContextEngine
from contracts import LabelSpec
from feature_engine import FeatureEngine
from label_engine import LabelEngine
from probability_engine import MODEL_HIERARCHY, ProbabilityEngine


def _make_deterministic_ohlcv(
    rng: np.random.Generator,
    index: pd.DatetimeIndex,
    base_price: float,
    volatility: float = 0.005,
) -> pd.DataFrame:
    """Genera un DataFrame OHLCV sintético determinista con velas válidas y volúmenes positivos."""
    n = len(index)
    log_rets = rng.normal(loc=0.0, scale=volatility, size=n)
    closes = base_price * np.exp(np.cumsum(log_rets))
    opens = np.empty(n, dtype=float)
    opens[0] = base_price
    opens[1:] = closes[:-1]

    wick_up = np.abs(rng.normal(loc=0.0018, scale=0.0012, size=n)) * closes
    wick_dn = np.abs(rng.normal(loc=0.0018, scale=0.0012, size=n)) * closes
    highs = np.maximum(opens, closes) + wick_up
    lows = np.maximum(1e-6, np.minimum(opens, closes) - wick_dn)

    volume = rng.uniform(50.0, 500.0, size=n)
    buy_ratio = rng.uniform(0.3, 0.7, size=n)
    taker_buy_volume = volume * buy_ratio
    quote_volume = volume * closes
    count = rng.integers(500, 5000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "quote_volume": quote_volume,
            "count": count,
            "taker_buy_volume": taker_buy_volume,
        },
        index=index,
    )


def _assert_metrics_exact(m1: Dict[str, Any], m2: Dict[str, Any], context: str = "") -> None:
    """Compara dos diccionarios de métricas exigiendo igualdad exacta (incluyendo NaNs y dicts anidados)."""
    assert set(m1.keys()) == set(m2.keys()), f"Claves distintas en {context}: {m1.keys()} vs {m2.keys()}"
    for key in m1:
        v1, v2 = m1[key], m2[key]
        if isinstance(v1, dict):
            assert isinstance(v2, dict), f"Tipo distinto para {context}.{key}"
            _assert_metrics_exact(v1, v2, context=f"{context}.{key}")
        elif isinstance(v1, list):
            assert isinstance(v2, list) and len(v1) == len(v2), f"Lista distinta para {context}.{key}"
            for idx, (item1, item2) in enumerate(zip(v1, v2)):
                if isinstance(item1, dict):
                    _assert_metrics_exact(item1, item2, context=f"{context}.{key}[{idx}]")
                else:
                    assert item1 == item2, f"Valor distinto en {context}.{key}[{idx}]: {item1} vs {item2}"
        elif isinstance(v1, float) and math.isnan(v1):
            assert isinstance(v2, float) and math.isnan(v2), f"NaN esperado en {context}.{key}, recibido {v2}"
        else:
            assert v1 == v2, f"Desviación en {context}.{key}: {v1!r} != {v2!r}"


@pytest.fixture(scope="module")
def deterministic_market_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dataset sintético determinista de 900 velas 15m UTC (seed=12345) suficientemente
    amplio para entrenar LightGBM (min_child_samples=100), RandomForest y calibradores
    con particiones temporalmente purgadas (train / calib / val) no degeneradas.
    """
    rng = np.random.default_rng(12345)
    idx_15m = pd.date_range("2026-01-01 00:00:00", periods=900, freq="15min", tz="UTC")

    df_btc = _make_deterministic_ohlcv(rng, idx_15m, base_price=70000.0, volatility=0.005)
    df_eth = _make_deterministic_ohlcv(rng, idx_15m, base_price=3500.0, volatility=0.006)

    idx_funding = pd.date_range(start=idx_15m[0], end=idx_15m[-1], freq="8h", tz="UTC")
    funding_rates = rng.normal(loc=0.0001, scale=0.00005, size=len(idx_funding))
    df_funding = pd.DataFrame({"funding_rate": funding_rates}, index=idx_funding)

    return df_btc, df_eth, df_funding


def _build_full_pipeline_dataset(
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    df_funding: pd.DataFrame,
    direction: str = "LONG",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Ejecuta FeatureEngine + ContextEngine + LabelEngine y retorna sus salidas y el dataset limpio."""
    feat_engine = FeatureEngine()
    ctx_engine = ContextEngine()
    label_engine = LabelEngine(
        LabelSpec(tp_mult=1.5, sl_mult=1.0, horizon_bars=24, entry_mode="next_open")
    )

    df_feat = feat_engine.build_features(df_btc, df_funding)
    df_ctx = ctx_engine.build_context(df_btc, df_eth=df_eth, atr_pct=df_feat["atr_pct"])
    df_labels = label_engine.compute_outcomes(df_btc, df_feat["atr_pct"], direction=direction)

    combined = pd.concat([df_feat, df_ctx], axis=1)
    feature_cols = [
        c for c in combined.columns if c.startswith(("ta_", "sm_", "deriv_", "ctx_"))
    ]
    ds = combined.copy()
    ds["target"] = df_labels["target_tp_before_sl"]
    ds["realized_r_gross"] = df_labels["realized_r_gross"]
    ds["exit_bars"] = df_labels["exit_bars"]
    ds = ds.dropna(subset=feature_cols + ["atr_pct", "target"]).copy()
    ds["target"] = ds["target"].astype(int)

    return df_feat, df_ctx, df_labels, ds, feature_cols


def _split_train_calib_val(
    ds: pd.DataFrame,
    purge_bars: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide `ds` cronológicamente en train / calib / val respetando `purge_bars` entre bloques."""
    n = len(ds)
    train_end = int(n * 0.55)
    calib_start = train_end + purge_bars
    calib_end = calib_start + int(n * 0.20)
    val_start = calib_end + purge_bars

    df_train = ds.iloc[:train_end].copy()
    df_calib = ds.iloc[calib_start:calib_end].copy()
    df_val = ds.iloc[val_start:].copy()

    assert not df_train.empty and not df_calib.empty and not df_val.empty
    assert df_calib.index.min() > df_train.index.max()
    assert df_val.index.min() > df_calib.index.max()
    return df_train, df_calib, df_val


# ---------------------------------------------------------------------------
# 1. Reproducibilidad bit-a-bit de FeatureEngine + ContextEngine + LabelEngine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_feature_context_label_engines_bit_for_bit_reproducible(
    deterministic_market_data, direction: str
):
    """
    Dos ejecuciones completamente independientes de FeatureEngine + ContextEngine +
    LabelEngine sobre copias profundas de los mismos datos deben producir DataFrames
    idénticos bit-a-bit (check_exact=True).
    """
    df_btc, df_eth, df_funding = deterministic_market_data

    # Ejecución 1 (instancias y copias independientes)
    feat_1, ctx_1, labels_1, ds_1, cols_1 = _build_full_pipeline_dataset(
        df_btc.copy(deep=True),
        df_eth.copy(deep=True),
        df_funding.copy(deep=True),
        direction=direction,
    )

    # Ejecución 2 (instancias y copias independientes)
    feat_2, ctx_2, labels_2, ds_2, cols_2 = _build_full_pipeline_dataset(
        df_btc.copy(deep=True),
        df_eth.copy(deep=True),
        df_funding.copy(deep=True),
        direction=direction,
    )

    assert cols_1 == cols_2 and len(cols_1) > 0
    pd.testing.assert_frame_equal(feat_1, feat_2, check_exact=True)
    pd.testing.assert_frame_equal(ctx_1, ctx_2, check_exact=True)
    pd.testing.assert_frame_equal(labels_1, labels_2, check_exact=True)
    pd.testing.assert_frame_equal(ds_1, ds_2, check_exact=True)

    summary_1 = LabelEngine.summarize_outcomes(labels_1)
    summary_2 = LabelEngine.summarize_outcomes(labels_2)
    _assert_metrics_exact(summary_1, summary_2, context=f"LabelEngine.summarize_outcomes[{direction}]")


# ---------------------------------------------------------------------------
# 2. Reproducibilidad bit-a-bit de ProbabilityEngine (los 5 modelos)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_probability_engine_five_models_bit_for_bit_reproducible(
    deterministic_market_data, direction: str
):
    """
    Dos ejecuciones independientes de ProbabilityEngine con random_state=42 sobre
    los 5 modelos (base_rate, logistic_regression, random_forest, gradient_boosting,
    lightgbm) deben producir probabilidades predichas y métricas idénticas bit-a-bit.
    """
    df_btc, df_eth, df_funding = deterministic_market_data
    _, _, _, ds, feature_cols = _build_full_pipeline_dataset(
        df_btc.copy(deep=True),
        df_eth.copy(deep=True),
        df_funding.copy(deep=True),
        direction=direction,
    )
    df_train, _, df_val = _split_train_calib_val(ds, purge_bars=24)
    y_train = df_train["target"]
    y_val = df_val["target"]

    # Asegurar que existen ambas clases en train y val para evaluación no degenerada
    assert set(y_train.unique()) == {0, 1}
    assert set(y_val.unique()) == {0, 1}

    eng_1 = ProbabilityEngine(direction=direction, random_state=42)
    res_1 = eng_1.train_and_compare(
        df_train=df_train.copy(deep=True),
        y_train=y_train.copy(deep=True),
        df_val=df_val.copy(deep=True),
        y_val=y_val.copy(deep=True),
        feature_cols=feature_cols,
        model_names=MODEL_HIERARCHY,
    )

    eng_2 = ProbabilityEngine(direction=direction, random_state=42)
    res_2 = eng_2.train_and_compare(
        df_train=df_train.copy(deep=True),
        y_train=y_train.copy(deep=True),
        df_val=df_val.copy(deep=True),
        y_val=y_val.copy(deep=True),
        feature_cols=feature_cols,
        model_names=MODEL_HIERARCHY,
    )

    expected_models = (
        "base_rate",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
    )
    assert tuple(res_1["models"].keys()) == expected_models
    assert tuple(res_2["models"].keys()) == expected_models
    assert res_1["best_model_name"] == res_2["best_model_name"]

    for model_name in expected_models:
        p_val_1 = res_1["val_predictions"][model_name]
        p_val_2 = res_2["val_predictions"][model_name]
        assert np.array_equal(p_val_1, p_val_2), (
            f"Predicciones de validación no idénticas bit-a-bit para {model_name} ({direction})"
        )

        # Verificar también predict_proba directo sobre el engine entrenado
        p_direct_1 = eng_1.predict_proba(df_val.copy(deep=True), model_name=model_name)
        p_direct_2 = eng_2.predict_proba(df_val.copy(deep=True), model_name=model_name)
        assert np.array_equal(p_direct_1, p_direct_2), (
            f"predict_proba directo no idéntico bit-a-bit para {model_name} ({direction})"
        )
        assert np.array_equal(p_val_1, p_direct_1)

        _assert_metrics_exact(
            res_1["val_metrics"][model_name],
            res_2["val_metrics"][model_name],
            context=f"ProbabilityEngine[{direction}].val_metrics[{model_name}]",
        )


# ---------------------------------------------------------------------------
# 3. Reproducibilidad bit-a-bit de compare_calibrators (none, platt, isotonic)
# ---------------------------------------------------------------------------
def test_compare_calibrators_bit_for_bit_reproducible(deterministic_market_data):
    """
    Dos ejecuciones independientes de compare_calibrators ('none', 'platt', 'isotonic')
    sobre probabilidades reales producidas por ProbabilityEngine deben generar
    probabilidades calibradas y métricas ECE, MCE, Brier, BSS y LogLoss idénticas.
    """
    df_btc, df_eth, df_funding = deterministic_market_data
    _, _, _, ds, feature_cols = _build_full_pipeline_dataset(
        df_btc.copy(deep=True),
        df_eth.copy(deep=True),
        df_funding.copy(deep=True),
        direction="LONG",
    )
    df_train, df_calib, df_val = _split_train_calib_val(ds, purge_bars=24)

    eng = ProbabilityEngine(direction="LONG", random_state=42)
    eng.train_and_compare(
        df_train=df_train,
        y_train=df_train["target"],
        df_val=df_calib,
        y_val=df_calib["target"],
        feature_cols=feature_cols,
        model_names=("lightgbm",),
    )
    raw_calib_probs = eng.predict_proba(df_calib, model_name="lightgbm")
    raw_val_probs = eng.predict_proba(df_val, model_name="lightgbm")
    y_calib = df_calib["target"].to_numpy(dtype=int)
    y_val = df_val["target"].to_numpy(dtype=int)

    cal_res_1 = compare_calibrators(
        raw_calib_probs=raw_calib_probs.copy(),
        y_calib=y_calib.copy(),
        raw_val_probs=raw_val_probs.copy(),
        y_val=y_val.copy(),
        methods=VALID_CALIBRATION_METHODS,
        n_bins=10,
        random_state=42,
    )
    cal_res_2 = compare_calibrators(
        raw_calib_probs=raw_calib_probs.copy(),
        y_calib=y_calib.copy(),
        raw_val_probs=raw_val_probs.copy(),
        y_val=y_val.copy(),
        methods=VALID_CALIBRATION_METHODS,
        n_bins=10,
        random_state=42,
    )

    assert VALID_CALIBRATION_METHODS == ("none", "platt", "isotonic")
    assert cal_res_1["best_method"] == cal_res_2["best_method"]

    for method in VALID_CALIBRATION_METHODS:
        p_cal_1 = cal_res_1["val_predictions"][method]
        p_cal_2 = cal_res_2["val_predictions"][method]
        assert np.array_equal(p_cal_1, p_cal_2), (
            f"Probabilidades calibradas no idénticas bit-a-bit para método '{method}'"
        )

        # Predicción directa con cada calibrador ajustado
        p_pred_1 = cal_res_1["calibrators"][method].predict(raw_val_probs.copy())
        p_pred_2 = cal_res_2["calibrators"][method].predict(raw_val_probs.copy())
        assert np.array_equal(p_pred_1, p_pred_2)
        assert np.array_equal(p_cal_1, p_pred_1)

        m1 = cal_res_1["val_metrics"][method]
        m2 = cal_res_2["val_metrics"][method]
        assert m1["ece"] == m2["ece"]
        assert m1["mce"] == m2["mce"]
        assert m1["brier_score"] == m2["brier_score"]
        _assert_metrics_exact(m1, m2, context=f"compare_calibrators[{method}]")


# ---------------------------------------------------------------------------
# 4. Reproducibilidad bit-a-bit de BenterQuantBrain (fit_and_calibrate + EV)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calibration_method", ["isotonic", "platt"])
def test_benter_quant_brain_fit_and_calibrate_bit_for_bit_reproducible(
    deterministic_market_data, calibration_method: str
):
    """
    Dos entrenamientos independientes de BenterQuantBrain con fit_and_calibrate y
    random_state=42 deben producir exactamente las mismas probabilidades calibradas,
    valores de EV / Kelly y métricas de evaluación fuera de muestra.
    """
    df_btc, df_eth, df_funding = deterministic_market_data
    _, _, _, ds, feature_cols = _build_full_pipeline_dataset(
        df_btc.copy(deep=True),
        df_eth.copy(deep=True),
        df_funding.copy(deep=True),
        direction="LONG",
    )
    df_train, df_calib, df_val = _split_train_calib_val(ds, purge_bars=24)

    def _train_brain() -> BenterQuantBrain:
        brain = BenterQuantBrain(
            feature_cols=feature_cols,
            tp_mult=1.5,
            sl_mult=1.0,
            roundtrip_fee=0.0010,
            direction="LONG",
            calibration_method=calibration_method,
            random_state=42,
        )
        brain.fit_and_calibrate(
            df_train=df_train.copy(deep=True),
            y_train=df_train["target"].copy(deep=True),
            df_calib=df_calib.copy(deep=True),
            y_calib=df_calib["target"].copy(deep=True),
            purge_bars=24,
            enforce_purge=True,
        )
        return brain

    brain_1 = _train_brain()
    brain_2 = _train_brain()

    p_cal_1 = brain_1.predict_calibrated_proba(df_val.copy(deep=True))
    p_cal_2 = brain_2.predict_calibrated_proba(df_val.copy(deep=True))
    assert np.array_equal(p_cal_1, p_cal_2), (
        f"Probabilidades calibradas de BenterQuantBrain no idénticas bit-a-bit ({calibration_method})"
    )

    atr_val = df_val["atr_pct"].to_numpy(dtype=float)
    ev_1 = brain_1.compute_ev_and_kelly(p_cal_1, atr_val.copy())
    ev_2 = brain_2.compute_ev_and_kelly(p_cal_2, atr_val.copy())
    pd.testing.assert_frame_equal(ev_1, ev_2, check_exact=True)
    assert np.array_equal(ev_1["ev_in_r"].to_numpy(), ev_2["ev_in_r"].to_numpy())
    assert np.array_equal(
        ev_1["risk_fraction_kelly"].to_numpy(),
        ev_2["risk_fraction_kelly"].to_numpy(),
    )

    eval_1 = brain_1.evaluate_model(df_val.copy(deep=True), df_val["target"].copy(deep=True))
    eval_2 = brain_2.evaluate_model(df_val.copy(deep=True), df_val["target"].copy(deep=True))
    _assert_metrics_exact(
        eval_1,
        eval_2,
        context=f"BenterQuantBrain.evaluate_model[{calibration_method}]",
    )
