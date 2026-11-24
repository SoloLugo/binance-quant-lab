"""
Tests de FeatureEngine (Fase 3 base) y ContextEngine (Fase 4 base).
Verifica:
- Activación y desactivación independiente de familias (`ta`, `microstructure`, `derivatives`).
- Ausencia absoluta de `.bfill()` cuando `df_funding` o `df_eth` empiezan más tarde que `df_btc`.
- Invarianza causal (modificar el futuro no altera ninguna feature pasada).
- Control Temporal Cuádruple (`available_at <= bar_timestamp`) en `ContextEngine`.
"""
import numpy as np
import pandas as pd
import pytest

from context_engine import ContextEngine, ExternalEvent
from feature_engine import ALL_FEATURE_FAMILIES, FeatureEngine
from pipeline import QuantFeaturePipeline


EXPECTED_TA_COLS = {
    "atr_pct",
    "ta_ret_1",
    "ta_ret_4",
    "ta_ret_16",
    "ta_dist_ema20",
    "ta_dist_ema50",
    "ta_rsi_14",
    "ta_bb_width",
    "ta_momentum_16",
    "ta_volatility_24",
    "ta_rel_volume_20",
    "ta_vwap_dist_24",
    "ta_dist_high_24",
    "ta_dist_low_24",
    "ta_body_ratio",
}

EXPECTED_SM_COLS = {
    "sm_ofi_1",
    "sm_ofi_cum_4",
    "sm_ofi_cum_16",
    "sm_avg_trade_size_zscore",
    "sm_whale_trade_zscore",
    "sm_lower_wick_ratio",
    "sm_upper_wick_ratio",
}

EXPECTED_DERIV_COLS = {
    "deriv_funding_rate",
    "deriv_funding_zscore",
    "deriv_funding_change_24h",
}


def test_independent_family_toggling(sample_ohlcv_and_funding):
    """Cada familia (`ta`, `microstructure`, `derivatives`) puede activarse o desactivarse por separado."""
    df_btc, _, df_funding = sample_ohlcv_and_funding

    # Todas activadas por defecto
    full = FeatureEngine().build_features(df_btc, df_funding)
    assert set(full.columns) == (EXPECTED_TA_COLS | EXPECTED_SM_COLS | EXPECTED_DERIV_COLS)
    assert (
        full["sm_avg_trade_size_zscore"].dropna()
        .equals(full["sm_whale_trade_zscore"].dropna())
    )

    # Solo TA
    only_ta = FeatureEngine(enabled_families=("ta",)).build_features(df_btc, df_funding)
    assert set(only_ta.columns) == EXPECTED_TA_COLS

    # Solo Microstructure
    only_sm = FeatureEngine(enabled_families=("microstructure",)).build_features(df_btc, df_funding)
    assert set(only_sm.columns) == EXPECTED_SM_COLS

    # Solo Derivatives
    only_deriv = FeatureEngine(enabled_families=("derivatives",)).build_features(df_btc, df_funding)
    assert set(only_deriv.columns) == EXPECTED_DERIV_COLS

    # Combinación parcial (ta + derivatives sin microstructure)
    ta_deriv = FeatureEngine(enabled_families=("ta", "derivatives")).build_features(df_btc, df_funding)
    assert set(ta_deriv.columns) == (EXPECTED_TA_COLS | EXPECTED_DERIV_COLS)

    # Ninguna familia habilitada -> DataFrame vacío con mismo índice
    empty = FeatureEngine(enabled_families=()).build_features(df_btc, df_funding)
    assert list(empty.columns) == []
    assert empty.index.equals(df_btc.index)

    # Familia desconocida -> ValueError
    with pytest.raises(ValueError, match="desconocidas"):
        FeatureEngine(enabled_families=("ta", "nonexistent_family"))


def test_no_bfill_when_funding_starts_later_than_btc(sample_ohlcv_and_funding):
    """
    Si `df_funding` comienza más tarde que `df_btc`, las barras anteriores a la primera
    observación de funding DEBEN permanecer como NaN (prohibido `.bfill()`).
    """
    df_btc, _, df_funding = sample_ohlcv_and_funding

    # Recortamos df_funding para que empiece 50 velas después del inicio de df_btc
    cutoff_ts = df_btc.index[50]
    df_funding_late = df_funding.loc[df_funding.index >= cutoff_ts].copy()
    assert not df_funding_late.empty
    first_funding_ts = df_funding_late.index[0]

    feat = FeatureEngine(enabled_families=("derivatives",)).build_features(df_btc, df_funding_late)

    before_mask = feat.index < first_funding_ts
    after_mask = feat.index >= first_funding_ts

    assert before_mask.sum() >= 50
    # Cero look-ahead: antes del primer funding todo debe ser NaN
    assert feat.loc[before_mask, "deriv_funding_rate"].isna().all()
    assert feat.loc[before_mask, "deriv_funding_zscore"].isna().all()
    assert feat.loc[before_mask, "deriv_funding_change_24h"].isna().all()

    # Desde el primer funding en adelante debe propagarse causalmente con ffill
    assert feat.loc[after_mask, "deriv_funding_rate"].notna().all()


def test_no_bfill_when_eth_starts_later_than_btc(sample_ohlcv_and_funding):
    """
    Si `df_eth` comienza más tarde que `df_btc`, `ctx_eth_btc_rel_24h` DEBE permanecer como NaN
    antes del primer dato de ETH (sin `.bfill()`).
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding

    # Recortamos df_eth para que empiece en la vela 80
    df_eth_late = df_eth.iloc[80:].copy()
    first_eth_ts = df_eth_late.index[0]

    ctx = ContextEngine().build_context(df_btc=df_btc, df_eth=df_eth_late)

    before_mask = ctx.index < first_eth_ts
    assert before_mask.sum() == 80
    # Antes de la primera vela de ETH (+ 4 velas de pct_change(4)) no puede haber datos rellenados
    assert ctx.loc[before_mask, "ctx_eth_btc_rel_24h"].isna().all()
    # A partir de vela 80 + 4 (y con ATR listo desde vela 13) debe haber valores válidos
    assert ctx.iloc[85:]["ctx_eth_btc_rel_24h"].notna().all()


def test_strict_causality_future_perturbation_invariance(sample_ohlcv_and_funding):
    """
    Modificar precios, volúmenes y funding a partir del índice T+1 NO debe alterar
    ninguna feature ni columna de contexto en índices <= T.
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    split_idx = 220
    split_ts = df_btc.index[split_idx]

    pipe = QuantFeaturePipeline()
    base = pipe.build_features(df_btc, df_eth, df_funding)

    df_btc_mod = df_btc.copy()
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("close")] *= 1.35
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("high")] *= 1.40
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("low")] *= 0.70
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("volume")] *= 5.0

    df_eth_mod = df_eth.copy()
    df_eth_mod.iloc[split_idx + 1:, df_eth_mod.columns.get_loc("close")] *= 0.50

    df_funding_mod = df_funding.copy()
    df_funding_mod.loc[df_funding_mod.index > split_ts, "funding_rate"] = 0.05

    perturbed = pipe.build_features(df_btc_mod, df_eth_mod, df_funding_mod)

    pd.testing.assert_frame_equal(
        base.iloc[: split_idx + 1],
        perturbed.iloc[: split_idx + 1],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_context_engine_available_at_temporal_control(sample_ohlcv_and_funding):
    """
    Verifica el Control Temporal Cuádruple (`available_at <= bar_timestamp`):
    1. Un evento programado con `available_at` anterior a la ventana se activa en `[event-24h, event+2h]`.
    2. Un evento publicado tarde (`available_at` en mitad de la ventana o tras el evento) NO se activa
       en ninguna vela con `bar_timestamp < available_at`.
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding
    # df_btc cubre 2026-01-01 00:00 UTC .. 2026-01-05 03:45 UTC (400 velas de 15m)

    # Evento A: FOMC programado el 2026-01-02 18:00 UTC, disponible desde 30 días antes
    ev_scheduled = ExternalEvent(
        name="FOMC_SCHEDULED",
        event_time="2026-01-02 18:00:00",
        publication_time="2025-12-03 18:00:00",
        effective_time="2026-01-02 18:00:00",
        available_at="2025-12-03 18:00:00",
    )

    # Evento B: Evento no programado el 2026-01-04 12:00 UTC, pero solo disponible desde 2026-01-04 11:00 UTC
    ev_late_pub = ExternalEvent(
        name="EMERGENCY_DECISION",
        event_time="2026-01-04 12:00:00",
        publication_time="2026-01-04 11:00:00",
        effective_time="2026-01-04 12:00:00",
        available_at="2026-01-04 11:00:00",
    )

    # Evento C: Evento publicado después del cierre de la ventana -> nunca debe activarse
    ev_retroactive = ExternalEvent(
        name="RETROACTIVE_LEAK",
        event_time="2026-01-03 12:00:00",
        publication_time="2026-01-03 16:00:00",
        effective_time="2026-01-03 12:00:00",
        available_at="2026-01-03 16:00:00",
    )

    engine = ContextEngine(events=[ev_scheduled, ev_late_pub, ev_retroactive])
    ctx = engine.build_context(df_btc=df_btc, df_eth=df_eth)
    flag = ctx["ctx_fomc_event_24h"]

    # Evento A: activo en 2026-01-01 18:00..2026-01-02 20:00
    assert flag.loc[pd.Timestamp("2026-01-01 17:45:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-01 18:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 18:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 20:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 20:15:00", tz="UTC")] == 0

    # Evento C (retroactivo): nunca se activa en su ventana
    assert flag.loc[pd.Timestamp("2026-01-03 12:00:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-03 14:00:00", tz="UTC")] == 0

    # Evento B (publicación tardía a las 11:00): 0 antes de las 11:00, 1 desde las 11:00 hasta las 14:00
    assert flag.loc[pd.Timestamp("2026-01-04 10:45:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-04 11:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 12:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 14:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 14:15:00", tz="UTC")] == 0
