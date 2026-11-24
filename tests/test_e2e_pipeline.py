"""
Sprint 6 — Pruebas Reales End-to-End (E2E) del Sistema Completo de Binance Quant Lab.

Reglas estrictas:
- CERO mocks, CERO MagicMock, CERO unittest.mock.patch.
- Ejecuta el pipeline real completo desde CSVs en formato Binance Vision en disco
  hasta BacktestEngine, Paper Trading (RealTimeFuturesSimulator) y TradingMonitor.
- Mantiene el Test Final Sellado estrictamente bloqueado durante todo el flujo de
  investigación y simulación (`len(sealed_test_access_log) == 0`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from backtest_engine import BacktestConfig, BacktestEngine
from calibration_engine import compare_calibrators
from contracts import CostModel, DataManifest, LabelSpec
from data_loader import HistoricalDataLoader
from data_quality import DataQualityGate
from ev_engine import EVEngine, estimate_timeout_economics
from label_engine import LabelEngine
from monitoring import TradingMonitor
from pipeline import QuantFeaturePipeline
from probability_engine import ProbabilityEngine
from risk_engine import RiskEngine, RiskLimits
from simulator import RealTimeFuturesSimulator
from splits import TemporalDatasetPartitioner


def _generate_deterministic_market_data(
    n_bars: int = 900,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera de forma determinista 900 velas 15m UTC de BTCUSDT, ETHUSDT y Funding Rate (8h)
    con estructura causal real (drift moderado + persistencia de flujo de órdenes) para que
    los modelos probabilísticos aprendan señal real sin mocks ni trucos.
    """
    rng = np.random.default_rng(seed)
    idx_15m = pd.date_range("2026-01-01 00:00:00", periods=n_bars, freq="15min", tz="UTC")

    # Señal latente causal observable vía OFI (taker_buy_volume) y momentum corto
    # 12 semiciclos en 900 velas garantizan fases alcistas y bajistas en train, calib y val
    regime_wave = np.sin(np.linspace(0.0, 12.0 * np.pi, n_bars))
    ofi_signal = np.clip(0.5 + 0.30 * regime_wave + rng.normal(0.0, 0.05, size=n_bars), 0.12, 0.88)

    # Retornos futuros influidos causalmente por el desequilibrio previo (ofi_signal[t-1])
    btc_log_rets = np.empty(n_bars, dtype=float)
    btc_log_rets[0] = 0.0
    for t in range(1, n_bars):
        causal_edge = 0.0085 * (ofi_signal[t - 1] - 0.5)
        btc_log_rets[t] = 0.0003 + causal_edge + rng.normal(0.0, 0.0030)

    btc_closes = 70000.0 * np.exp(np.cumsum(btc_log_rets))
    btc_opens = np.empty(n_bars, dtype=float)
    btc_opens[0] = 70000.0
    btc_opens[1:] = btc_closes[:-1]

    btc_wick_up = np.abs(rng.normal(0.0014, 0.0007, size=n_bars)) * btc_closes
    btc_wick_dn = np.abs(rng.normal(0.0014, 0.0007, size=n_bars)) * btc_closes
    btc_highs = np.maximum(btc_opens, btc_closes) + btc_wick_up
    btc_lows = np.maximum(1.0, np.minimum(btc_opens, btc_closes) - btc_wick_dn)

    btc_volume = rng.uniform(80.0, 450.0, size=n_bars)
    btc_taker_buy = btc_volume * ofi_signal
    btc_quote_vol = btc_volume * btc_closes
    btc_taker_buy_quote = btc_taker_buy * btc_closes
    btc_count = rng.integers(800, 4500, size=n_bars)

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    open_time_ms = ((idx_15m - epoch) // pd.Timedelta("1ms")).to_numpy(dtype=np.int64)
    close_time_ms = open_time_ms + (15 * 60 * 1000) - 1

    df_btc_raw = pd.DataFrame(
        {
            "open_time": open_time_ms,
            "open": btc_opens,
            "high": btc_highs,
            "low": btc_lows,
            "close": btc_closes,
            "volume": btc_volume,
            "close_time": close_time_ms,
            "quote_volume": btc_quote_vol,
            "count": btc_count,
            "taker_buy_volume": btc_taker_buy,
            "taker_buy_quote_volume": btc_taker_buy_quote,
            "ignore": np.zeros(n_bars, dtype=int),
        }
    )

    # ETHUSDT correlacionado con BTCUSDT + ruido idiosincrático
    eth_log_rets = 0.85 * btc_log_rets + rng.normal(0.0, 0.0025, size=n_bars)
    eth_closes = 3500.0 * np.exp(np.cumsum(eth_log_rets))
    eth_opens = np.empty(n_bars, dtype=float)
    eth_opens[0] = 3500.0
    eth_opens[1:] = eth_closes[:-1]

    eth_wick_up = np.abs(rng.normal(0.0016, 0.0008, size=n_bars)) * eth_closes
    eth_wick_dn = np.abs(rng.normal(0.0016, 0.0008, size=n_bars)) * eth_closes
    eth_highs = np.maximum(eth_opens, eth_closes) + eth_wick_up
    eth_lows = np.maximum(1.0, np.minimum(eth_opens, eth_closes) - eth_wick_dn)

    eth_volume = rng.uniform(500.0, 3000.0, size=n_bars)
    eth_buy_ratio = rng.uniform(0.35, 0.65, size=n_bars)
    eth_taker_buy = eth_volume * eth_buy_ratio
    eth_quote_vol = eth_volume * eth_closes
    eth_taker_buy_quote = eth_taker_buy * eth_closes
    eth_count = rng.integers(600, 3500, size=n_bars)

    df_eth_raw = pd.DataFrame(
        {
            "open_time": open_time_ms,
            "open": eth_opens,
            "high": eth_highs,
            "low": eth_lows,
            "close": eth_closes,
            "volume": eth_volume,
            "close_time": close_time_ms,
            "quote_volume": eth_quote_vol,
            "count": eth_count,
            "taker_buy_volume": eth_taker_buy,
            "taker_buy_quote_volume": eth_taker_buy_quote,
            "ignore": np.zeros(n_bars, dtype=int),
        }
    )

    # Funding Rate cada 8h cubriendo todo el rango temporal
    idx_funding = pd.date_range(start=idx_15m[0], end=idx_15m[-1], freq="8h", tz="UTC")
    calc_time_ms = ((idx_funding - epoch) // pd.Timedelta("1ms")).to_numpy(dtype=np.int64)
    funding_rates = rng.normal(loc=0.0001, scale=0.00004, size=len(idx_funding))
    df_funding_raw = pd.DataFrame(
        {
            "calc_time": calc_time_ms,
            "funding_interval_hours": np.full(len(idx_funding), 8, dtype=int),
            "last_funding_rate": funding_rates,
        }
    )

    return df_btc_raw, df_eth_raw, df_funding_raw


def _write_binance_vision_csvs(
    raw_dir: Path,
    df_btc_raw: pd.DataFrame,
    df_eth_raw: pd.DataFrame,
    df_funding_raw: pd.DataFrame,
) -> None:
    """Escribe CSVs en la estructura exacta de Binance Vision bajo `raw_dir/symbol=.../`."""
    btc_dir = raw_dir / "symbol=BTCUSDT"
    eth_dir = raw_dir / "symbol=ETHUSDT"
    btc_dir.mkdir(parents=True, exist_ok=True)
    eth_dir.mkdir(parents=True, exist_ok=True)

    df_btc_raw.to_csv(btc_dir / "BTCUSDT-15m-2026-01.csv", index=False)
    df_eth_raw.to_csv(eth_dir / "ETHUSDT-15m-2026-01.csv", index=False)
    df_funding_raw.to_csv(btc_dir / "BTCUSDT-fundingRate-2026-01.csv", index=False)


def test_e2e_full_quant_pipeline_without_mocks(tmp_path: Path) -> None:
    """
    Flujo real End-to-End sin mocks desde CSVs en formato Binance Vision hasta
    BacktestEngine, Paper Trading (RealTimeFuturesSimulator) y TradingMonitor.
    """
    raw_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "manifests"
    paper_dir = tmp_path / "paper"

    # 1. Escritura de CSVs reales en formato Binance Vision (900 velas 15m UTC)
    df_btc_raw, df_eth_raw, df_funding_raw = _generate_deterministic_market_data(
        n_bars=900, seed=2026
    )
    _write_binance_vision_csvs(raw_dir, df_btc_raw, df_eth_raw, df_funding_raw)

    # 2. Carga, validación de calidad y manifiesto SHA256 determinista
    loader = HistoricalDataLoader(raw_dir=str(raw_dir), manifests_dir=str(manifests_dir))
    df_btc, btc_manifest = loader.load_and_validate_klines(
        symbol="BTCUSDT",
        interval="15m",
        dataset_version="BTCUSDT_15m_e2e_v1",
        save_manifest=True,
    )
    df_eth, eth_manifest = loader.load_and_validate_klines(
        symbol="ETHUSDT",
        interval="15m",
        dataset_version="ETHUSDT_15m_e2e_v1",
        save_manifest=True,
    )
    df_funding = loader.load_funding_rate(symbol="BTCUSDT")

    assert isinstance(btc_manifest, DataManifest)
    assert btc_manifest.quality_passed is True
    assert eth_manifest.quality_passed is True
    assert len(df_btc) == 900
    assert len(df_eth) == 900
    assert len(df_funding) > 0
    assert (manifests_dir / "BTCUSDT_15m_e2e_v1.json").exists()
    assert (manifests_dir / "ETHUSDT_15m_e2e_v1.json").exists()

    # Verificar determinismo del hash SHA256 en segunda lectura idéntica
    _, btc_manifest_repeat = loader.load_and_validate_klines(
        symbol="BTCUSDT",
        interval="15m",
        dataset_version="BTCUSDT_15m_e2e_v1",
        save_manifest=True,
    )
    assert btc_manifest_repeat.dataset_hash == btc_manifest.dataset_hash
    assert btc_manifest_repeat.file_hashes == btc_manifest.file_hashes

    alignment = DataQualityGate().validate_alignment(df_btc, df_eth, df_funding)
    assert alignment["passed"] is True
    assert alignment["overlap_ratio"] == pytest.approx(1.0)

    # 3. Features + Contexto (incluyendo market_structure) y Labels (next_open)
    quant_pipe = QuantFeaturePipeline(
        tp_mult=2.0,
        sl_mult=1.0,
        horizon_bars=24,
        entry_mode="next_open",
        exclude_ambiguous=True,
        include_market_structure=True,
    )
    features_df = quant_pipe.build_features(df_btc=df_btc, df_eth=df_eth, df_funding=df_funding)
    assert any(c.startswith("ta_") for c in features_df.columns)
    assert any(c.startswith("sm_") for c in features_df.columns)
    assert any(c.startswith("deriv_") for c in features_df.columns)
    assert any(c.startswith("ms_") for c in features_df.columns)
    assert any(c.startswith("ctx_") for c in features_df.columns)
    assert "atr_pct" in features_df.columns

    label_spec = LabelSpec(
        tp_mult=2.0,
        sl_mult=1.0,
        horizon_bars=24,
        entry_mode="next_open",
        exclude_ambiguous=True,
    )
    label_engine = LabelEngine(spec=label_spec)
    outcomes_df = label_engine.compute_outcomes(
        df_btc=df_btc,
        atr_pct=features_df["atr_pct"],
        direction="LONG",
    )
    assert "outcome_state" in outcomes_df.columns
    assert "target_tp_before_sl" in outcomes_df.columns

    # Seleccionamos features numéricas con cobertura completa tras warmup
    candidate_feature_cols = [
        c
        for c in features_df.columns
        if c.startswith(("ta_", "sm_", "deriv_", "ms_", "ctx_"))
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    # Filtramos columnas que mantengan >=85% de cobertura sobre las velas válidas
    valid_target_mask = outcomes_df["target_tp_before_sl"].notna()
    feature_cols = [
        c
        for c in candidate_feature_cols
        if features_df.loc[valid_target_mask, c].notna().mean() >= 0.85
    ]
    assert len(feature_cols) >= 15
    assert any(c.startswith("ms_") for c in feature_cols)

    dataset = features_df[feature_cols + ["atr_pct"]].copy()
    dataset["target"] = outcomes_df["target_tp_before_sl"]
    dataset["outcome_state"] = outcomes_df["outcome_state"]
    dataset["realized_r_gross"] = outcomes_df["realized_r_gross"]
    dataset = dataset.dropna(subset=feature_cols + ["atr_pct", "target"])
    assert len(dataset) >= 550

    # 4. Partición temporal estricta con purging/embargo y Test Final Sellado bloqueado
    partitioner = TemporalDatasetPartitioner(
        dataset,
        purge_bars=24,
        embargo_bars=24,
        sealed_test_ratio=0.15,
    )
    df_train, df_calib, df_val = partitioner.get_dev_split()

    assert len(partitioner.sealed_test_access_log) == 0
    with pytest.raises(PermissionError):
        partitioner.get_sealed_test(unlock_sealed_test=False, reason="")
    assert len(partitioner.sealed_test_access_log) == 0

    assert df_train.index.max() < df_calib.index.min()
    assert df_calib.index.max() < df_val.index.min()

    y_train = df_train["target"].astype(int)
    y_calib = df_calib["target"].astype(int)
    y_val = df_val["target"].astype(int)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_calib.unique()) == {0, 1}
    assert set(y_val.unique()) == {0, 1}

    # 5. Entrenamiento y selección sobre val con ProbabilityEngine + Calibración en calib -> val
    prob_engine = ProbabilityEngine(direction="LONG", random_state=42)
    train_summary = prob_engine.train_and_compare(
        df_train=df_train,
        y_train=y_train,
        df_val=df_val,
        y_val=y_val,
        feature_cols=feature_cols,
    )
    assert train_summary["best_model_name"] in train_summary["models"]

    raw_calib_probs = prob_engine.predict_proba(df_calib)
    raw_val_probs = prob_engine.predict_proba(df_val)
    assert np.all(np.isfinite(raw_calib_probs))
    assert np.all(np.isfinite(raw_val_probs))

    calib_summary = compare_calibrators(
        raw_calib_probs=raw_calib_probs,
        y_calib=y_calib,
        raw_val_probs=raw_val_probs,
        y_val=y_val,
        methods=("none", "platt", "isotonic"),
        random_state=42,
    )
    best_calibrator = calib_summary["best_calibrator"]
    p_val_cal = pd.Series(
        best_calibrator.predict(raw_val_probs),
        index=df_val.index,
        name="p_calibrated",
    )
    assert np.all((p_val_cal >= 1e-6) & (p_val_cal <= 1.0 - 1e-6))

    # 6. Evaluación económica EV con timeout_economics estimado SOLO sobre train+calib
    train_calib_idx = df_train.index.union(df_calib.index)
    timeout_econ = estimate_timeout_economics(outcomes_df.loc[train_calib_idx])
    assert 0.0 <= timeout_econ.timeout_share_given_not_tp <= 1.0

    cost_model = CostModel(
        taker_fee_rate=0.0005,
        maker_fee_rate=0.0002,
        half_spread_bps=0.5,
        slippage_bps=1.5,
        expected_funding_rate_8h=0.0001,
        horizon_hours=6.0,
    )
    ev_engine = EVEngine(
        tp_mult=2.0,
        sl_mult=1.0,
        cost_model=cost_model,
        ev_min_threshold=0.02,
        direction="LONG",
    )
    ev_val_df = ev_engine.evaluate(
        p_calibrated=p_val_cal,
        atr_pct=df_val["atr_pct"],
        p_raw=raw_val_probs,
        timeout_economics=timeout_econ,
    )
    assert list(ev_val_df.columns) == list(EVEngine.OUTPUT_COLUMNS)
    assert np.all(np.isfinite(ev_val_df["ev_net_r"]))
    assert np.all(ev_val_df["total_cost_in_r"] > 0.0)

    # 7. Evaluación con RiskEngine independiente + verificación de Kill Switch al 100%
    risk_limits = RiskLimits(
        max_risk_per_trade=0.02,
        kelly_fraction=0.25,
        probability_uncertainty_margin=0.01,
        max_leverage=5.0,
        max_open_positions=1,
        max_daily_loss_pct=0.06,
        max_drawdown_pct=0.15,
    )
    risk_engine = RiskEngine(limits=risk_limits)

    initial_equity = 10000.0
    risk_decisions = {}
    for ts in df_val.index:
        row_ev = ev_val_df.loc[ts]
        entry_px = float(df_btc.loc[ts, "close"])
        sl_dist_usdt = entry_px * float(df_val.loc[ts, "atr_pct"]) * 1.0
        b_net_odds = max(0.1, float(row_ev["gross_win_r"] - row_ev["total_cost_in_r"]))
        l_net_odds = 1.0 + float(row_ev["total_cost_in_r"])

        decision = risk_engine.evaluate_order(
            symbol="BTCUSDT",
            side="LONG",
            equity_usdt=initial_equity,
            available_balance_usdt=initial_equity,
            peak_equity_usdt=initial_equity,
            daily_start_equity_usdt=initial_equity,
            entry_price=entry_px,
            sl_distance_usdt=sl_dist_usdt,
            ev_net_r=float(row_ev["ev_net_r"]),
            p_calibrated=float(row_ev["p_calibrated"]),
            b_net_odds=b_net_odds,
            l_net_odds=l_net_odds,
        )
        risk_decisions[ts] = decision

    # Verificar que si se activa kill switch rechaza el 100% de las señales (incluyendo p=0.95)
    risk_engine.activate_kill_switch("E2E_CIRCUIT_TEST")
    assert risk_engine.kill_switch_active is True
    for ts in df_val.index[:10]:
        ks_decision = risk_engine.evaluate_order(
            symbol="BTCUSDT",
            side="LONG",
            equity_usdt=initial_equity,
            available_balance_usdt=initial_equity,
            peak_equity_usdt=initial_equity,
            daily_start_equity_usdt=initial_equity,
            entry_price=float(df_btc.loc[ts, "close"]),
            sl_distance_usdt=float(df_btc.loc[ts, "close"]) * 0.005,
            ev_net_r=0.80,
            p_calibrated=0.95,
            b_net_odds=1.90,
            l_net_odds=1.05,
        )
        assert ks_decision.approved is False
        assert ks_decision.reason == "KILL_SWITCH_ACTIVE"
    risk_engine.reset_kill_switch()
    assert risk_engine.kill_switch_active is False

    # 8. BacktestEngine sobre val: contabilidad exacta y estado aislado
    val_ohlcv = df_btc.loc[df_val.index[0] : df_val.index[-1]].copy()
    signals_df = pd.DataFrame(index=val_ohlcv.index)
    signals_df["side"] = None
    signals_df["atr_pct"] = features_df.reindex(val_ohlcv.index)["atr_pct"]
    signals_df["risk_fraction"] = np.nan
    signals_df["p_calibrated"] = np.nan
    signals_df["ev_net_r"] = np.nan

    approved_timestamps = [
        ts
        for ts in df_val.index
        if bool(ev_val_df.loc[ts, "accepted"]) and risk_decisions[ts].approved
    ]
    assert len(approved_timestamps) >= 1, (
        "El pipeline E2E sobre datos deterministas con señal causal debe aprobar >= 1 trade en val."
    )

    for ts in approved_timestamps:
        signals_df.loc[ts, "side"] = "LONG"
        signals_df.loc[ts, "risk_fraction"] = risk_decisions[ts].risk_fraction
        signals_df.loc[ts, "p_calibrated"] = float(ev_val_df.loc[ts, "p_calibrated"])
        signals_df.loc[ts, "ev_net_r"] = float(ev_val_df.loc[ts, "ev_net_r"])

    bt_config = BacktestConfig(
        initial_balance=initial_equity,
        tp_mult=2.0,
        sl_mult=1.0,
        horizon_bars=24,
        risk_per_trade_pct=0.01,
        max_leverage=5.0,
        cost_model=cost_model,
    )
    bt_engine = BacktestEngine(config=bt_config)
    bt_res_1 = bt_engine.run(df_ohlcv=val_ohlcv, signals=signals_df, df_funding=df_funding)
    bt_res_2 = bt_engine.run(df_ohlcv=val_ohlcv, signals=signals_df, df_funding=df_funding)

    assert bt_res_1.metrics["total_trades"] >= 1
    trades_bt = bt_res_1.trades

    # Contabilidad exacta: final_equity == initial + gross - fees - funding
    expected_final_equity = (
        initial_equity
        + float(trades_bt["gross_pnl_usdt"].sum())
        - float(trades_bt["total_fees_usdt"].sum())
        - float(trades_bt["funding_paid_usdt"].sum())
    )
    assert bt_res_1.metrics["final_equity_usdt"] == pytest.approx(expected_final_equity, abs=1e-6)

    # Estado aislado y determinismo entre corridas consecutivas
    assert bt_res_1.metrics["final_equity_usdt"] == pytest.approx(
        bt_res_2.metrics["final_equity_usdt"], abs=1e-12
    )
    assert bt_res_1.metrics["total_trades"] == bt_res_2.metrics["total_trades"]
    pd.testing.assert_frame_equal(bt_res_1.trades, bt_res_2.trades)

    # 9. Ciclo de Paper Trading con RealTimeFuturesSimulator(..., fresh_state=True) + TradingMonitor
    sim_state_path = paper_dir / "sim_state.json"
    trades_csv_path = paper_dir / "trades.csv"
    signals_jsonl_path = paper_dir / "signals.jsonl"
    executions_jsonl_path = paper_dir / "executions.jsonl"

    simulator = RealTimeFuturesSimulator(
        state_path=str(sim_state_path),
        trade_log_path=str(trades_csv_path),
        initial_balance=initial_equity,
        fresh_state=True,
        timeframe="15m",
    )
    monitor = TradingMonitor(
        signals_log_path=str(signals_jsonl_path),
        execution_log_path=str(executions_jsonl_path),
    )

    funding_aligned = (
        df_funding["funding_rate"]
        .reindex(val_ohlcv.index, method="ffill")
        .fillna(0.0001)
    )
    active_signal_meta = None

    for ts in val_ohlcv.index:
        bar = val_ohlcv.loc[ts]
        now_dt = ts.to_pydatetime()
        close_px = float(bar["close"])
        bid_px = close_px * (1.0 - cost_model.half_spread_bps / 10000.0)
        ask_px = close_px * (1.0 + cost_model.half_spread_bps / 10000.0)
        f_rate = float(funding_aligned.loc[ts])

        # Si hay posición abierta, actualizamos tick real de la vela
        if simulator.has_open_position():
            closed_trade = simulator.update_on_market_tick(
                mark_price=close_px,
                bid_price=bid_px,
                ask_price=ask_px,
                funding_rate=f_rate,
                candle_high=float(bar["high"]),
                candle_low=float(bar["low"]),
                now_utc=now_dt,
            )
            if closed_trade is not None and active_signal_meta is not None:
                actual_cost_r = (
                    float(closed_trade["total_fees_usdt"])
                    + float(closed_trade["funding_paid_usdt"])
                ) / max(active_signal_meta["risk_1r_usdt"], 1e-6)
                monitor.record_execution_comparison(
                    trade_id=closed_trade["trade_id"],
                    side=closed_trade["side"],
                    expected_entry_price=active_signal_meta["expected_entry_price"],
                    actual_entry_price=float(closed_trade["entry_price"]),
                    expected_ev_r=active_signal_meta["expected_ev_r"],
                    realized_net_r=float(closed_trade["realized_net_r"]),
                    expected_cost_r=active_signal_meta["expected_cost_r"],
                    actual_cost_r=actual_cost_r,
                    exit_reason=str(closed_trade["exit_reason"]),
                )
                active_signal_meta = None
            continue

        # Evaluamos nueva señal si la vela pertenece a df_val
        if ts not in ev_val_df.index:
            continue

        row_ev = ev_val_df.loc[ts]
        sl_dist_usdt = close_px * float(df_val.loc[ts, "atr_pct"]) * 1.0
        b_net_odds = max(0.1, float(row_ev["gross_win_r"] - row_ev["total_cost_in_r"]))
        l_net_odds = 1.0 + float(row_ev["total_cost_in_r"])

        live_decision = risk_engine.evaluate_order(
            symbol="BTCUSDT",
            side="LONG",
            equity_usdt=float(simulator.state["equity"]),
            available_balance_usdt=float(simulator.state["wallet_balance"]),
            peak_equity_usdt=float(simulator.state["peak_equity"]),
            daily_start_equity_usdt=float(simulator.state["daily_start_equity"]),
            entry_price=ask_px,
            sl_distance_usdt=sl_dist_usdt,
            ev_net_r=float(row_ev["ev_net_r"]),
            p_calibrated=float(row_ev["p_calibrated"]),
            b_net_odds=b_net_odds,
            l_net_odds=l_net_odds,
        )

        monitor.record_signal(
            timestamp=ts,
            symbol="BTCUSDT",
            side="LONG",
            p_raw=float(row_ev["p_raw"]),
            p_calibrated=float(row_ev["p_calibrated"]),
            ev_net_r=float(row_ev["ev_net_r"]),
            estimated_cost_r=float(row_ev["total_cost_in_r"]),
            risk_decision=live_decision,
            data_status="OK",
        )

        if bool(row_ev["accepted"]) and live_decision.approved and not simulator.check_daily_circuit_breaker(now_dt):
            opened = simulator.open_simulated_position(
                side="LONG",
                qty_btc=live_decision.qty_btc,
                bid_price=bid_px,
                ask_price=ask_px,
                mark_price=close_px,
                atr_pct=float(df_val.loc[ts, "atr_pct"]),
                sl_mult=1.0,
                tp_mult=2.0,
                horizon_bars=24,
                p_win=float(row_ev["p_calibrated"]),
                ev_in_r=float(row_ev["ev_net_r"]),
                b_net_odds=b_net_odds,
                kelly_risk_fraction=live_decision.risk_fraction,
                bar_timestamp=ts,
                now_utc=now_dt,
            )
            active_signal_meta = {
                "expected_entry_price": ask_px,
                "expected_ev_r": float(row_ev["ev_net_r"]),
                "expected_cost_r": float(row_ev["total_cost_in_r"]),
                "risk_1r_usdt": float(opened["risk_1r_usdt"]),
            }

    # Si al finalizar la ventana val quedase una posición abierta, cerramos a precio final de sesión
    if simulator.has_open_position() and active_signal_meta is not None:
        last_ts = val_ohlcv.index[-1]
        last_close = float(val_ohlcv.iloc[-1]["close"])
        closed_trade = simulator.close_simulated_position(
            exit_price=last_close,
            exit_reason="END_OF_SESSION",
            now_utc=last_ts.to_pydatetime(),
        )
        actual_cost_r = (
            float(closed_trade["total_fees_usdt"]) + float(closed_trade["funding_paid_usdt"])
        ) / max(active_signal_meta["risk_1r_usdt"], 1e-6)
        monitor.record_execution_comparison(
            trade_id=closed_trade["trade_id"],
            side=closed_trade["side"],
            expected_entry_price=active_signal_meta["expected_entry_price"],
            actual_entry_price=float(closed_trade["entry_price"]),
            expected_ev_r=active_signal_meta["expected_ev_r"],
            realized_net_r=float(closed_trade["realized_net_r"]),
            expected_cost_r=active_signal_meta["expected_cost_r"],
            actual_cost_r=actual_cost_r,
            exit_reason=str(closed_trade["exit_reason"]),
        )

    # Verificar persistencia de todos los artefactos de Paper Trading y Monitoring
    assert signals_jsonl_path.exists() and signals_jsonl_path.stat().st_size > 0
    assert executions_jsonl_path.exists() and executions_jsonl_path.stat().st_size > 0
    assert sim_state_path.exists() and sim_state_path.stat().st_size > 0
    assert trades_csv_path.exists() and trades_csv_path.stat().st_size > 0

    with open(sim_state_path, "r", encoding="utf-8") as f:
        persisted_state = json.load(f)
    assert persisted_state["total_trades"] >= 1
    assert persisted_state["open_position"] is None

    trades_csv_df = pd.read_csv(trades_csv_path)
    assert len(trades_csv_df) == persisted_state["total_trades"]
    assert {"trade_id", "side", "entry_price", "exit_price", "net_pnl_usdt", "realized_net_r"}.issubset(
        trades_csv_df.columns
    )

    monitor_summary = monitor.summary()
    assert monitor_summary["total_signals"] >= 1
    assert monitor_summary["approved_signals"] >= 1
    assert monitor_summary["total_executions"] == persisted_state["total_trades"]
    assert monitor_summary["error_count"] == 0

    # Garantía final: el Test Final Sellado jamás fue tocado durante todo el flujo E2E
    assert len(partitioner.sealed_test_access_log) == 0


def test_e2e_corrupted_csv_blocked_by_quality_gate(tmp_path: Path) -> None:
    """
    Verifica que DataQualityGate detiene el pipeline E2E de inmediato si el CSV en disco
    contiene datos corruptos (OHLC inconsistente, precios <= 0 y volumen negativo),
    sin persistir manifiestos inválidos.
    """
    raw_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "manifests"

    df_btc_raw, df_eth_raw, df_funding_raw = _generate_deterministic_market_data(
        n_bars=900, seed=2026
    )

    # Inyectamos corrupciones reales en el CSV de BTCUSDT antes de escribir a disco
    df_btc_corrupted = df_btc_raw.copy()
    # 1. OHLC inconsistente: low > high
    df_btc_corrupted.loc[10, "low"] = df_btc_corrupted.loc[10, "high"] + 500.0
    # 2. Precio inválido (<= 0)
    df_btc_corrupted.loc[20, "close"] = -100.0
    # 3. Volumen negativo
    df_btc_corrupted.loc[30, "volume"] = -25.0

    _write_binance_vision_csvs(raw_dir, df_btc_corrupted, df_eth_raw, df_funding_raw)

    loader = HistoricalDataLoader(raw_dir=str(raw_dir), manifests_dir=str(manifests_dir))
    with pytest.raises(ValueError, match="DataQualityGate BLOQUEADO"):
        loader.load_and_validate_klines(
            symbol="BTCUSDT",
            interval="15m",
            dataset_version="BTCUSDT_15m_corrupted_v1",
            save_manifest=True,
        )

    # Confirmamos que no se generó ningún manifiesto corrupto en disco
    assert not (manifests_dir / "BTCUSDT_15m_corrupted_v1.json").exists()
