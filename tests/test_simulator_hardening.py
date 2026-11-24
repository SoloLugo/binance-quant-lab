from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import pytest

from contracts import RiskDecision
from monitoring import TradingMonitor
from simulator import RealTimeFuturesSimulator


def test_trading_monitor_signals_executions_errors_and_summary(tmp_path):
    signals_log = str(tmp_path / "logs" / "signals.jsonl")
    exec_log = str(tmp_path / "logs" / "execution.jsonl")
    monitor = TradingMonitor(signals_log_path=signals_log, execution_log_path=exec_log)

    # Resumen vacío determinista antes de escribir eventos
    empty_summary = monitor.summary()
    assert empty_summary["total_signals"] == 0
    assert empty_summary["approval_rate"] == 0.0
    assert empty_summary["error_count"] == 0
    assert empty_summary["data_degraded_count"] == 0

    t0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    # 1 señal aprobada con datos OK
    monitor.record_signal(
        timestamp=t0,
        symbol="BTCUSDT",
        side="LONG",
        p_raw=0.58,
        p_calibrated=0.55,
        ev_net_r=0.35,
        estimated_cost_r=0.12,
        risk_decision=RiskDecision(
            approved=True,
            qty_btc=0.15,
            risk_fraction=0.02,
            reason="APPROVED",
            effective_leverage=1.05,
        ),
        data_status="OK",
        extra={"regime": "TREND"},
    )
    # 1 señal rechazada por kill switch con datos degradados
    monitor.record_signal(
        timestamp=t0 + timedelta(minutes=15),
        symbol="BTCUSDT",
        side="SHORT",
        p_raw=0.61,
        p_calibrated=0.59,
        ev_net_r=0.40,
        estimated_cost_r=0.12,
        risk_decision=RiskDecision(
            approved=False,
            qty_btc=0.0,
            risk_fraction=0.0,
            reason="KILL_SWITCH_ACTIVE",
            effective_leverage=0.0,
        ),
        data_status="STALE_FEED",
    )
    # 1 señal rechazada por correlación
    monitor.record_signal(
        timestamp=t0 + timedelta(minutes=30),
        symbol="ETHUSDT",
        side="LONG",
        p_raw=0.60,
        p_calibrated=0.57,
        ev_net_r=0.30,
        estimated_cost_r=0.11,
        risk_decision=RiskDecision(
            approved=False,
            qty_btc=0.0,
            risk_fraction=0.0,
            reason="CORRELATION_LIMIT_BREACHED",
            effective_leverage=0.0,
        ),
        data_status="OK",
    )

    # Comparación de ejecución LONG: precio esperado 70,000, real 70,014 -> +2.0 bps adversos
    rec_long = monitor.record_execution_comparison(
        trade_id=1,
        side="LONG",
        expected_entry_price=70000.0,
        actual_entry_price=70014.0,
        expected_ev_r=0.35,
        realized_net_r=1.85,
        expected_cost_r=0.12,
        actual_cost_r=0.14,
        exit_reason="TAKE_PROFIT",
    )
    assert rec_long["entry_slippage_deviation_bps"] == pytest.approx(2.0)
    assert rec_long["ev_vs_realized_gap_r"] == pytest.approx(1.50)

    # Comparación de ejecución SHORT: precio esperado 70,000, real 69,972 -> +4.0 bps adversos
    rec_short = monitor.record_execution_comparison(
        trade_id=2,
        side="SHORT",
        expected_entry_price=70000.0,
        actual_entry_price=69972.0,
        expected_ev_r=0.25,
        realized_net_r=-1.05,
        expected_cost_r=0.12,
        actual_cost_r=0.15,
        exit_reason="STOP_LOSS",
    )
    assert rec_short["entry_slippage_deviation_bps"] == pytest.approx(4.0)
    assert rec_short["ev_vs_realized_gap_r"] == pytest.approx(-1.30)

    # Registro de errores operativos
    monitor.record_error(context="fetch_orderbook", error_msg="Timeout 5s")
    monitor.record_error(context="funding_sync", error_msg="HTTP 429 rate limit")

    # Verificamos que los archivos JSONL son válidos línea por línea
    with open(signals_log, "r", encoding="utf-8") as f:
        signal_lines = [json.loads(line) for line in f if line.strip()]
    assert len(signal_lines) == 3
    assert signal_lines[0]["extra"] == {"regime": "TREND"}

    summary = monitor.summary()
    assert summary["total_signals"] == 3
    assert summary["approved_signals"] == 1
    assert summary["rejected_signals"] == 2
    assert summary["approval_rate"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert summary["rejection_rate"] == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert summary["rejection_reasons"] == {
        "KILL_SWITCH_ACTIVE": 1,
        "CORRELATION_LIMIT_BREACHED": 1,
    }
    assert summary["total_executions"] == 2
    assert summary["mean_entry_slippage_deviation_bps"] == pytest.approx(3.0)
    assert summary["mean_ev_vs_realized_gap_r"] == pytest.approx(0.10)
    assert summary["error_count"] == 2
    assert summary["data_degraded_count"] == 1


def test_simulator_dynamic_timeframe_horizon_1h(tmp_path):
    """Verifica que timeframe='1h' calcule el horizonte exacto como horizon_bars * 3600s y no 15m."""
    sim = RealTimeFuturesSimulator(
        state_path=str(tmp_path / "sim_1h.json"),
        trade_log_path=str(tmp_path / "trades_1h.csv"),
        initial_balance=10000.0,
        fresh_state=True,
        timeframe="1h",
    )

    t0 = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    horizon_bars = 6
    pos = sim.open_simulated_position(
        side="LONG",
        qty_btc=0.1,
        bid_price=69999.0,
        ask_price=70000.0,
        mark_price=70000.0,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        horizon_bars=horizon_bars,
        p_win=0.55,
        ev_in_r=0.30,
        b_net_odds=1.90,
        kelly_risk_fraction=0.02,
        bar_timestamp=pd.Timestamp(t0),
        now_utc=t0,
    )

    expected_horizon_sec = horizon_bars * 3600
    assert pos["horizon_expiry_epoch"] - pos["entry_epoch"] == pytest.approx(expected_horizon_sec)

    # Tras 6 * 15m = 90 minutos NO debe expirar (estamos en timeframe 1h, no 15m)
    t_90m = t0 + timedelta(minutes=90)
    res_90m = sim.update_on_market_tick(
        mark_price=70050.0,
        bid_price=70049.0,
        ask_price=70051.0,
        now_utc=t_90m,
    )
    assert res_90m is None
    assert sim.has_open_position()

    # Al cumplirse exactamente 6 * 3600s (6 horas) debe cerrar por TIME_BARRIER
    t_6h = t0 + timedelta(hours=6)
    res_6h = sim.update_on_market_tick(
        mark_price=70050.0,
        bid_price=70049.0,
        ask_price=70051.0,
        now_utc=t_6h,
    )
    assert res_6h is not None
    assert res_6h["exit_reason"] == "TIME_BARRIER"


def test_simulator_take_profit_applies_adverse_slippage_long_and_short(tmp_path):
    """Verifica que TAKE_PROFIT aplique slippage adverso en LONG (menor precio) y SHORT (mayor precio)."""
    t0 = datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc)

    # Caso LONG
    sim_long = RealTimeFuturesSimulator(
        state_path=str(tmp_path / "sim_long.json"),
        trade_log_path=str(tmp_path / "trades_long.csv"),
        initial_balance=10000.0,
        fresh_state=True,
        timeframe="15m",
    )
    pos_long = sim_long.open_simulated_position(
        side="LONG",
        qty_btc=0.1,
        bid_price=69999.0,
        ask_price=70000.0,
        mark_price=70000.0,
        atr_pct=0.005,
        sl_mult=1.0,
        tp_mult=2.0,
        horizon_bars=24,
        p_win=0.55,
        ev_in_r=0.30,
        b_net_odds=1.90,
        kelly_risk_fraction=0.02,
        bar_timestamp=pd.Timestamp(t0),
        now_utc=t0,
    )
    closed_long = sim_long.update_on_market_tick(
        mark_price=pos_long["tp_price"] + 5.0,
        bid_price=pos_long["tp_price"] + 4.0,
        ask_price=pos_long["tp_price"] + 6.0,
        now_utc=t0 + timedelta(minutes=30),
    )
    assert closed_long is not None
    assert closed_long["exit_reason"] == "TAKE_PROFIT"
    expected_long_exit = round(pos_long["tp_price"] * (1.0 - sim_long.slippage_rate), 2)
    assert closed_long["exit_price"] == pytest.approx(expected_long_exit)
    assert closed_long["exit_price"] < pos_long["tp_price"]

    # Caso SHORT
    sim_short = RealTimeFuturesSimulator(
        state_path=str(tmp_path / "sim_short.json"),
        trade_log_path=str(tmp_path / "trades_short.csv"),
        initial_balance=10000.0,
        fresh_state=True,
        timeframe="15m",
    )
    pos_short = sim_short.open_simulated_position(
        side="SHORT",
        qty_btc=0.1,
        bid_price=70000.0,
        ask_price=70001.0,
        mark_price=70000.0,
        atr_pct=0.005,
        sl_mult=1.0,
        tp_mult=2.0,
        horizon_bars=24,
        p_win=0.55,
        ev_in_r=0.30,
        b_net_odds=1.90,
        kelly_risk_fraction=0.02,
        bar_timestamp=pd.Timestamp(t0),
        now_utc=t0,
    )
    closed_short = sim_short.update_on_market_tick(
        mark_price=pos_short["tp_price"] - 5.0,
        bid_price=pos_short["tp_price"] - 6.0,
        ask_price=pos_short["tp_price"] - 4.0,
        now_utc=t0 + timedelta(minutes=30),
    )
    assert closed_short is not None
    assert closed_short["exit_reason"] == "TAKE_PROFIT"
    expected_short_exit = round(pos_short["tp_price"] * (1.0 + sim_short.slippage_rate), 2)
    assert closed_short["exit_price"] == pytest.approx(expected_short_exit)
    assert closed_short["exit_price"] > pos_short["tp_price"]
