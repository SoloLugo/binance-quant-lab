import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from brain import BenterQuantBrain
from config import settings
from data_loader import HistoricalDataLoader
from live_testnet_bot import BinanceFuturesTestnetBot
from pipeline import QuantFeaturePipeline
from simulator import RealTimeFuturesSimulator


def test_timestamp_normalization_mixed_ms_and_us():
    """Verifica que HistoricalDataLoader normalice correctamente filas mezcladas en ms y us."""
    s = pd.Series([1704067200000, 1735689600000000])  # 2024-01-01 en ms y 2025-01-01 en us
    norm = HistoricalDataLoader._normalize_epoch_ms(s)
    assert int(norm.iloc[0]) == 1704067200000
    assert int(norm.iloc[1]) == 1735689600000


def test_triple_barrier_outcomes_tp_sl_and_time():
    """Verifica que compute_triple_barrier_outcomes identifique TP, SL, TIME y marque NaN al final."""
    idx = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")
    closes = np.full(60, 70000.0)
    highs = np.full(60, 70050.0)
    lows = np.full(60, 69950.0)

    # En la vela i=0: entrada=70000, ATR%=0.005 (350 USDT), TP=70700, SL=69650
    # Hacemos que la vela i=3 toque el TP (70750)
    highs[3] = 70750.0

    # En la vela i=10: hacemos que la vela i=12 toque el SL (69600)
    lows[12] = 69600.0

    df_btc = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes}, index=idx)
    atr_pct = pd.Series(0.005, index=idx)

    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    outcomes = pipe.compute_triple_barrier_outcomes(df_btc, atr_pct, direction="LONG")

    assert outcomes["target_tp_before_sl"].iloc[0] == 1.0
    assert outcomes["exit_type"].iloc[0] == "TP"
    assert outcomes["exit_bars"].iloc[0] == 3
    assert outcomes["realized_r_gross"].iloc[0] == pytest.approx(2.0)

    assert outcomes["target_tp_before_sl"].iloc[10] == 0.0
    assert outcomes["exit_type"].iloc[10] == "SL"
    assert outcomes["exit_bars"].iloc[10] == 2
    assert outcomes["realized_r_gross"].iloc[10] == pytest.approx(-1.0)

    # En la vela i=15 no se toca ni TP ni SL en 24 velas -> TIME barrier
    assert outcomes["exit_type"].iloc[15] == "TIME"
    assert outcomes["exit_bars"].iloc[15] == 24
    assert outcomes["realized_r_gross"].iloc[15] == pytest.approx(0.0)

    # Las últimas 24 filas deben tener target NaN para evitar falsos negativos al final del dataset
    assert outcomes["target_tp_before_sl"].iloc[-24:].isna().all()


def test_simulator_full_lifecycle_tp_sl_time_and_funding(tmp_path):
    """Verifica el ciclo completo del simulador: spread, slippage, fees, funding 8h, TP, SL y Time-Stop."""
    state_file = str(tmp_path / "sim_state.json")
    csv_file = str(tmp_path / "trades.csv")
    sim = RealTimeFuturesSimulator(state_path=state_file, trade_log_path=csv_file, initial_balance=10000.0)

    t0 = datetime(2026, 10, 1, 7, 45, tzinfo=timezone.utc)
    bar_ts = pd.Timestamp(t0)

    # 1. Abrimos posición LONG de 0.1 BTC con precio ~70,000 y ATR%=0.005 (SL=350, TP=700)
    pos = sim.open_simulated_position(
        side="LONG",
        qty_btc=0.1,
        bid_price=69999.0,
        ask_price=70000.0,
        mark_price=70000.0,
        atr_pct=0.005,
        sl_mult=1.0,
        tp_mult=2.0,
        horizon_bars=24,
        p_win=0.52,
        ev_in_r=0.25,
        b_net_odds=1.80,
        kelly_risk_fraction=0.015,
        bar_timestamp=bar_ts,
        now_utc=t0,
    )
    assert sim.has_open_position()
    assert pos["entry_price"] > 70000.0  # Incluye slippage sobre ask

    # 2. Avanzamos a las 08:01 UTC (cruza ventana de funding de las 08:00 UTC con funding=0.01%)
    t1 = datetime(2026, 10, 1, 8, 1, tzinfo=timezone.utc)
    closed = sim.update_on_market_tick(
        mark_price=70100.0,
        bid_price=70099.0,
        ask_price=70101.0,
        funding_rate=0.0001,
        now_utc=t1,
    )
    assert closed is None
    assert sim.state["open_position"]["funding_paid_usdt"] > 0.0

    # 3. Alcanzamos el Take Profit
    t2 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    closed = sim.update_on_market_tick(
        mark_price=pos["tp_price"] + 10.0,
        bid_price=pos["tp_price"] + 9.0,
        ask_price=pos["tp_price"] + 11.0,
        funding_rate=0.0001,
        now_utc=t2,
    )
    assert closed is not None
    assert closed["exit_reason"] == "TAKE_PROFIT"
    assert closed["net_pnl_usdt"] > 0.0
    assert not sim.has_open_position()
    assert os.path.exists(csv_file)


def test_daily_circuit_breaker_halts_trading(tmp_path):
    """Verifica que el Circuit Breaker bloquee el trading si el drawdown diario supera el límite."""
    sim = RealTimeFuturesSimulator(
        state_path=str(tmp_path / "state.json"),
        trade_log_path=str(tmp_path / "trades.csv"),
        initial_balance=10000.0,
    )
    # Simulamos una caída del 7% en el equity del día (> 6% max_daily_drawdown_pct)
    sim.state["equity"] = 9250.0
    assert sim.check_daily_circuit_breaker() is True
    assert sim.state["daily_halted"] is True


def test_safe_position_sizing_and_emergency_sl_rollback(monkeypatch):
    """
    Verifica que:
      1. compute_safe_position_size respete el límite de riesgo y apalancamiento.
      2. Si falla STOP_MARKET tras abrir una orden a mercado en el exchange, se ejecute rollback inmediato.
    """
    bot = BinanceFuturesTestnetBot(execution_mode="paper")

    # Con balance pequeño (100 USDT), un lote mínimo de 0.002 BTC (~140 USDT) con SL de 500 USDT
    # arriesgaría 1.0 USDT (1% de 100 USDT -> permitido), pero con balance de 20 USDT excedería el 3% de riesgo
    qty_rejected = bot.compute_safe_position_size(
        available_balance_usdt=20.0,
        kelly_risk_fraction=0.02,
        entry_price=70000.0,
        sl_distance_usdt=500.0,
    )
    assert qty_rejected == 0.0

    qty_valid = bot.compute_safe_position_size(
        available_balance_usdt=10000.0,
        kelly_risk_fraction=0.01,
        entry_price=70000.0,
        sl_distance_usdt=350.0,
    )
    assert qty_valid == pytest.approx(0.285, abs=0.002)

    # Simulamos fallo en STOP_MARKET para verificar el rollback automático de seguridad
    mock_client = MagicMock()
    mock_client.create_order.side_effect = [
        {"id": "1", "average": 70000.0},  # Orden de entrada market exitosa
         RuntimeError("Binance -2021 Order would immediately trigger"),  # Fallo en STOP_MARKET
        {"id": "2", "status": "closed"},  # Orden de rollback reduceOnly market exitosa
    ]
    bot.execution_client = mock_client
    bot.execution_mode = "testnet"

    with pytest.raises(RuntimeError, match="Posición cerrada por seguridad"):
        bot._place_exchange_bracket_orders(side="LONG", qty_btc=0.01, atr_pct=0.005)

    # Verificamos que la tercera llamada a create_order fue el cierre de emergencia con reduceOnly=True
    assert mock_client.create_order.call_count == 3
    rollback_call = mock_client.create_order.call_args_list[2]
    assert rollback_call[0][1] == "market"
    assert rollback_call[0][2] == "sell"
    assert rollback_call[0][5] == {"reduceOnly": True}
