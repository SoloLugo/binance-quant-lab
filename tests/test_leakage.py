import numpy as np
import pandas as pd
import pytest

from context_engine import ContextEngine, ExternalEvent
from contracts import LabelSpec
from feature_engine import FeatureEngine
from label_engine import LabelEngine
from pipeline import QuantFeaturePipeline
from splits import TemporalDatasetPartitioner, verify_no_temporal_overlap


def _perturb_future_data(
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    df_funding: pd.DataFrame,
    cutoff_idx: int,
):
    """Perturba masivamente TODOS los datos estrictamente futuros (t > T)."""
    cutoff_ts = df_btc.index[cutoff_idx]

    df_btc_pert = df_btc.copy()
    future_btc_mask = df_btc_pert.index > cutoff_ts
    for col in ("open", "high", "low", "close"):
        df_btc_pert.loc[future_btc_mask, col] = df_btc_pert.loc[future_btc_mask, col] * 3.5
    for col in ("volume", "quote_volume", "taker_buy_volume", "count"):
        df_btc_pert.loc[future_btc_mask, col] = (
            df_btc_pert.loc[future_btc_mask, col].to_numpy()[::-1] * 2.1
        )

    df_eth_pert = df_eth.copy()
    future_eth_mask = df_eth_pert.index > cutoff_ts
    for col in ("open", "high", "low", "close"):
        df_eth_pert.loc[future_eth_mask, col] = df_eth_pert.loc[future_eth_mask, col] * 3.5
    for col in ("volume", "quote_volume", "taker_buy_volume", "count"):
        df_eth_pert.loc[future_eth_mask, col] = (
            df_eth_pert.loc[future_eth_mask, col].to_numpy()[::-1] * 2.1
        )

    df_funding_pert = df_funding.copy()
    future_fund_mask = df_funding_pert.index > cutoff_ts
    df_funding_pert.loc[future_fund_mask, "funding_rate"] = 0.05

    return df_btc_pert, df_eth_pert, df_funding_pert


def test_causal_invariance_bit_for_bit_pipeline_and_engines(sample_ohlcv_and_funding):
    """
    Invarianza causal bit-a-bit de QuantFeaturePipeline, FeatureEngine y ContextEngine:
    Perturbar masivamente t > T no debe alterar ningún valor ni NaN en t <= T.
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    cutoff_t = 250

    df_btc_pert, df_eth_pert, df_funding_pert = _perturb_future_data(
        df_btc, df_eth, df_funding, cutoff_idx=cutoff_t
    )

    # 1. QuantFeaturePipeline completo
    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    feat_orig = pipe.build_features(df_btc, df_eth, df_funding)
    feat_pert = pipe.build_features(df_btc_pert, df_eth_pert, df_funding_pert)

    pd.testing.assert_frame_equal(
        feat_orig.iloc[: cutoff_t + 1],
        feat_pert.iloc[: cutoff_t + 1],
        check_exact=False,
        rtol=0.0,
        atol=0.0,
    )
    # Sanity check: la perturbación futura sí produjo cambios en t > T
    assert not feat_orig.iloc[cutoff_t + 1 :].equals(feat_pert.iloc[cutoff_t + 1 :])

    # 2. FeatureEngine aislado
    fe = FeatureEngine()
    fe_orig = fe.build_features(df_btc=df_btc, df_funding=df_funding)
    fe_pert = fe.build_features(df_btc=df_btc_pert, df_funding=df_funding_pert)
    pd.testing.assert_frame_equal(
        fe_orig.iloc[: cutoff_t + 1],
        fe_pert.iloc[: cutoff_t + 1],
        check_exact=False,
        rtol=0.0,
        atol=0.0,
    )

    # 3. ContextEngine aislado
    ce = ContextEngine()
    ce_orig = ce.build_context(df_btc=df_btc, df_eth=df_eth, atr_pct=fe_orig["atr_pct"])
    ce_pert = ce.build_context(df_btc=df_btc_pert, df_eth=df_eth_pert, atr_pct=fe_pert["atr_pct"])
    pd.testing.assert_frame_equal(
        ce_orig.iloc[: cutoff_t + 1],
        ce_pert.iloc[: cutoff_t + 1],
        check_exact=False,
        rtol=0.0,
        atol=0.0,
    )


def test_no_bfill_lookahead_when_eth_or_funding_start_late(sample_ohlcv_and_funding):
    """
    Ausencia total de .bfill() / look-ahead cuando df_eth o df_funding inician tarde (en t = 50):
    Para t < 50, deriv_funding_rate y ctx_eth_btc_rel_24h deben ser NaN (nunca valores de t >= 50).
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    late_start_idx = 50
    late_start_ts = df_btc.index[late_start_idx]

    df_eth_late = df_eth.iloc[late_start_idx:].copy()
    df_funding_late = df_funding.loc[df_funding.index >= late_start_ts].copy()
    assert not df_eth_late.empty
    assert not df_funding_late.empty

    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    feats = pipe.build_features(df_btc, df_eth_late, df_funding_late)

    # Antes de t=50 NO puede existir ningún valor rellenado hacia atrás desde el futuro
    assert feats["deriv_funding_rate"].iloc[:late_start_idx].isna().all()
    assert feats["ctx_eth_btc_rel_24h"].iloc[:late_start_idx].isna().all()

    # Una vez disponibles los datos (y tras el lag de 4 velas para pct_change(4)), sí hay valores válidos
    first_funding_ts = df_funding_late.index[0]
    assert feats.loc[first_funding_ts:, "deriv_funding_rate"].notna().all()
    assert feats["ctx_eth_btc_rel_24h"].iloc[late_start_idx + 4 :].notna().all()


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_label_engine_barrier_entry_causal_invariance_and_incomplete_tail(
    sample_ohlcv_and_funding, direction
):
    """
    Verifica en LabelEngine (entry_mode='next_open'):
      1. entry_price, tp_price y sl_price en la barra i solo dependen de atr_pct[i] y open[i+1],
         nunca de high/low/close de i+1 ni de ninguna vela posterior i+2..end.
      2. Las últimas horizon_bars velas tienen outcome_state == 'INCOMPLETE' y todos los targets NaN.
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    horizon = 24
    tp_mult = 2.0
    sl_mult = 1.0

    fe = FeatureEngine(enabled_families=("ta",))
    atr_pct = fe.build_features(df_btc)["atr_pct"]

    spec = LabelSpec(
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        horizon_bars=horizon,
        entry_mode="next_open",
        exclude_ambiguous=True,
    )
    engine = LabelEngine(spec=spec)
    outcomes_orig = engine.compute_outcomes(df_btc, atr_pct, direction=direction)

    # Elegimos una barra válida i donde atr_pct[i] es finito
    bar_i = 100
    assert np.isfinite(atr_pct.iloc[bar_i])

    # Perturbamos high/low/close en i+1 y TODO OHLC en i+2..end, preservando open[i+1] y atr_pct[i]
    df_btc_pert = df_btc.copy()
    df_btc_pert.iloc[bar_i + 1, df_btc_pert.columns.get_loc("high")] *= 2.5
    df_btc_pert.iloc[bar_i + 1, df_btc_pert.columns.get_loc("low")] *= 0.4
    df_btc_pert.iloc[bar_i + 1, df_btc_pert.columns.get_loc("close")] *= 1.8
    for col in ("open", "high", "low", "close"):
        df_btc_pert.iloc[bar_i + 2 :, df_btc_pert.columns.get_loc(col)] *= 4.0

    outcomes_pert = engine.compute_outcomes(df_btc_pert, atr_pct, direction=direction)

    # entry_price, tp_price y sl_price en bar_i deben ser idénticos bit-a-bit
    assert outcomes_pert["entry_price"].iloc[bar_i] == outcomes_orig["entry_price"].iloc[bar_i]
    assert outcomes_pert["tp_price"].iloc[bar_i] == outcomes_orig["tp_price"].iloc[bar_i]
    assert outcomes_pert["sl_price"].iloc[bar_i] == outcomes_orig["sl_price"].iloc[bar_i]

    # Comprobación exacta contra la fórmula causal: open[i+1] y atr_pct[i]
    expected_entry = float(df_btc["open"].iloc[bar_i + 1])
    expected_atr_val = float(atr_pct.iloc[bar_i]) * expected_entry
    expected_tp = (
        expected_entry + tp_mult * expected_atr_val
        if direction == "LONG"
        else expected_entry - tp_mult * expected_atr_val
    )
    expected_sl = (
        expected_entry - sl_mult * expected_atr_val
        if direction == "LONG"
        else expected_entry + sl_mult * expected_atr_val
    )
    assert outcomes_orig["entry_price"].iloc[bar_i] == pytest.approx(expected_entry, rel=0.0, abs=0.0)
    assert outcomes_orig["tp_price"].iloc[bar_i] == pytest.approx(expected_tp, rel=0.0, abs=0.0)
    assert outcomes_orig["sl_price"].iloc[bar_i] == pytest.approx(expected_sl, rel=0.0, abs=0.0)

    # Últimas horizon_bars velas: INCOMPLETE y todos los targets NaN
    tail = outcomes_orig.iloc[-horizon:]
    assert (tail["outcome_state"] == "INCOMPLETE").all()
    assert tail["target_tp_before_sl"].isna().all()
    assert tail["target_tp_before_horizon"].isna().all()
    assert tail["target_sl_before_tp"].isna().all()
    assert tail["realized_r_gross"].isna().all()


def test_context_engine_respects_available_at_for_late_publication(sample_ohlcv_and_funding):
    """
    Control temporal available_at en ContextEngine:
    Un ExternalEvent publicado después de event_time (publicación tardía) NO debe activarse
    en ninguna vela con timestamp < available_at, incluso si está dentro de la ventana del evento.
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding

    event_ts = df_btc.index[200]
    pub_ts = event_ts + pd.Timedelta(hours=1)  # Publicación tardía (4 velas de 15m después)

    late_event = ExternalEvent(
        name="SURPRISE_REGULATORY_RELEASE",
        event_time=event_ts,
        publication_time=pub_ts,
        effective_time=event_ts,
        available_at=pub_ts,
    )

    ce = ContextEngine(events=[late_event], pre_event_hours=24.0, post_event_hours=2.0)
    ctx = ce.build_context(df_btc=df_btc, df_eth=df_eth)
    flag = ctx["ctx_fomc_event_24h"]

    # Estrictamente 0 en TODAS las velas anteriores a available_at (incluido event_time)
    before_avail = flag.loc[flag.index < pub_ts]
    assert (before_avail == 0).all()
    assert flag.loc[event_ts] == 0

    # Activo únicamente en [available_at, event_time + 2h]
    window_end = event_ts + pd.Timedelta(hours=2)
    active_window = flag.loc[(flag.index >= pub_ts) & (flag.index <= window_end)]
    assert len(active_window) > 0
    assert (active_window == 1).all()

    # Inactivo después de la ventana post-evento
    after_window = flag.loc[flag.index > window_end]
    assert (after_window == 0).all()


def test_temporal_dataset_partitioner_sealed_isolation_and_walk_forward_gaps():
    """
    Aislamiento de TemporalDatasetPartitioner (src/splits.py):
      - Bloquea el Test Final Sellado sin desbloqueo explícito + reason.
      - Ningún fold walk-forward toca jamás el Test Final Sellado.
      - Todos los splits y folds respetan purga + embargo >= horizon_bars.
    """
    horizon_bars = 24
    purge_bars = 24
    embargo_bars = 24
    bar_delta = pd.Timedelta(minutes=15)

    idx = pd.date_range("2026-01-01 00:00:00", periods=2000, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(70000.0, 75000.0, len(idx))}, index=idx)

    # Rechaza purge_bars o embargo_bars menores a horizon_bars
    with pytest.raises(ValueError):
        TemporalDatasetPartitioner(
            df,
            purge_bars=horizon_bars - 1,
            embargo_bars=embargo_bars,
            horizon_bars=horizon_bars,
        )
    with pytest.raises(ValueError):
        TemporalDatasetPartitioner(
            df,
            purge_bars=purge_bars,
            embargo_bars=horizon_bars - 1,
            horizon_bars=horizon_bars,
        )

    partitioner = TemporalDatasetPartitioner(
        df,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        sealed_test_ratio=0.15,
        bar_delta=bar_delta,
        horizon_bars=horizon_bars,
    )

    # Acceso al Test Final Sellado bloqueado por defecto
    with pytest.raises(PermissionError):
        partitioner.get_sealed_test()
    with pytest.raises(PermissionError):
        partitioner.get_sealed_test(unlock_sealed_test=True, reason="   ")

    sealed_df = partitioner.get_sealed_test(
        unlock_sealed_test=True,
        reason="QA anti-leakage verification",
    )
    min_required_gap = (purge_bars + embargo_bars) * bar_delta
    assert min_required_gap >= horizon_bars * bar_delta

    # Split de desarrollo
    df_train, df_calib, df_val = partitioner.get_dev_split()
    for part in (df_train, df_calib, df_val):
        assert part.index.intersection(sealed_df.index).empty
        assert part.index.max() < sealed_df.index.min()
    verify_no_temporal_overlap(df_train.index, df_calib.index, purge_bars, embargo_bars, bar_delta)
    verify_no_temporal_overlap(df_calib.index, df_val.index, purge_bars, embargo_bars, bar_delta)
    verify_no_temporal_overlap(df_val.index, sealed_df.index, purge_bars, embargo_bars, bar_delta)

    # Folds Walk-Forward
    folds = partitioner.get_walk_forward_folds(n_splits=4)
    assert len(folds) == 4
    for fold_train, fold_calib, fold_val in folds:
        for part in (fold_train, fold_calib, fold_val):
            assert part.index.intersection(sealed_df.index).empty
            assert part.index.max() < sealed_df.index.min()

        assert (fold_calib.index.min() - fold_train.index.max()) >= min_required_gap
        assert (fold_val.index.min() - fold_calib.index.max()) >= min_required_gap
        assert (sealed_df.index.min() - fold_val.index.max()) >= min_required_gap

        verify_no_temporal_overlap(
            fold_train.index, fold_calib.index, purge_bars, embargo_bars, bar_delta
        )
        verify_no_temporal_overlap(
            fold_calib.index, fold_val.index, purge_bars, embargo_bars, bar_delta
        )
        verify_no_temporal_overlap(
            fold_val.index, sealed_df.index, purge_bars, embargo_bars, bar_delta
        )
