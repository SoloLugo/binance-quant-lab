"""
Suite de tests de Fase 14: SafeExecutionEngine y endurecimiento de live_testnet_bot.py.

Utiliza un adaptador de exchange en memoria realista (`InMemoryTestnetExchange`)
sin `MagicMock`, con libro real de posiciones, órdenes abiertas, `clientOrderId`,
llenados parciales e inyección de fallos en `STOP_MARKET`.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from execution_engine import ExecutionConfig, SafeExecutionEngine
from live_testnet_bot import BinanceFuturesTestnetBot


class InMemoryTestnetExchange:
    """
    Simulador determinista en memoria de Binance USD-M Futures Testnet (interfaz CCXT).
    Mantiene libro real de posiciones, libro de órdenes abiertas e historial completo.
    """

    def __init__(
        self,
        mark_price: float = 100000.0,
        partial_fill_qty: Optional[float] = None,
        fail_stop_market: bool = False,
    ) -> None:
        self.mark_price = float(mark_price)
        self.partial_fill_qty = partial_fill_qty
        self.fail_stop_market = fail_stop_market
        self._order_seq = 0
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.cancel_all_called: bool = False
        self.canceled_order_ids: List[str] = []

    def seed_open_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        client_order_id: str,
        stop_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._order_seq += 1
        oid = f"ORD_{self._order_seq}"
        order = {
            "id": oid,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": float(amount),
            "status": "open",
            "clientOrderId": client_order_id,
            "info": {"clientOrderId": client_order_id},
            "stopPrice": stop_price,
        }
        self.open_orders[oid] = order
        self.order_history.append(order)
        return order

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "last": self.mark_price, "close": self.mark_price}

    def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [o for o in self.open_orders.values() if o["symbol"] == symbol]

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for sym, pos in self.positions.items():
            if symbols is None or sym in symbols:
                result.append(dict(pos))
        return result

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        if order_id not in self.open_orders:
            raise KeyError(f"Orden no encontrada: {order_id}")
        order = self.open_orders.pop(order_id)
        order["status"] = "canceled"
        self.canceled_order_ids.append(order_id)
        return order

    def cancel_all_orders(self, symbol: str) -> None:
        self.cancel_all_called = True
        raise AssertionError(
            "VIOLACIÓN DE SEGURIDAD: cancel_all_orders() jamás debe ser invocado por SafeExecutionEngine."
        )

    def _apply_market_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool,
    ) -> None:
        signed_delta = qty if side == "buy" else -qty
        existing = self.positions.get(symbol)
        current_signed = 0.0
        if existing is not None and existing["contracts"] > 0:
            current_signed = (
                existing["contracts"] if existing["side"] == "long" else -existing["contracts"]
            )

        new_signed = current_signed + signed_delta
        if reduce_only:
            # No permite invertir dirección en reduceOnly
            if current_signed > 0 and new_signed < 0:
                new_signed = 0.0
            elif current_signed < 0 and new_signed > 0:
                new_signed = 0.0

        if abs(new_signed) < 1e-12:
            self.positions[symbol] = {
                "symbol": symbol,
                "contracts": 0.0,
                "side": "long",
                "entryPrice": 0.0,
                "markPrice": self.mark_price,
                "timestamp": int(time.time() * 1000),
            }
        else:
            self.positions[symbol] = {
                "symbol": symbol,
                "contracts": round(abs(new_signed), 8),
                "side": "long" if new_signed > 0 else "short",
                "entryPrice": self.mark_price,
                "markPrice": self.mark_price,
                "timestamp": int(time.time() * 1000),
            }

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = dict(params or {})
        cid = params.get("newClientOrderId", "")

        if order_type == "STOP_MARKET" and self.fail_stop_market:
            raise RuntimeError("Simulated STOP_MARKET rejection (-2021)")

        self._order_seq += 1
        oid = f"ORD_{self._order_seq}"

        if order_type == "market":
            reduce_only = bool(params.get("reduceOnly", False))
            if not reduce_only and self.partial_fill_qty is not None:
                filled = float(self.partial_fill_qty)
            else:
                filled = float(amount)

            if filled > 0:
                self._apply_market_fill(symbol, side, filled, reduce_only=reduce_only)

            order = {
                "id": oid,
                "symbol": symbol,
                "type": "market",
                "side": side,
                "amount": float(amount),
                "filled": filled,
                "average": self.mark_price if filled > 0 else 0.0,
                "price": self.mark_price if filled > 0 else 0.0,
                "status": "closed" if filled > 0 else "expired",
                "clientOrderId": cid,
                "info": {"clientOrderId": cid},
                "params": params,
            }
            self.order_history.append(order)
            return order

        # Órdenes condicionales (STOP_MARKET / TAKE_PROFIT_MARKET)
        order = {
            "id": oid,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": float(amount),
            "filled": 0.0,
            "stopPrice": params.get("stopPrice"),
            "workingType": params.get("workingType"),
            "reduceOnly": bool(params.get("reduceOnly", False)),
            "status": "open",
            "clientOrderId": cid,
            "info": {"clientOrderId": cid},
            "params": params,
        }
        self.open_orders[oid] = order
        self.order_history.append(order)
        return order


def _make_config(
    tmp_path,
    execution_mode: str = "testnet",
    dry_run: bool = False,
    horizon_bars: int = 24,
    timeframe: str = "15m",
) -> ExecutionConfig:
    return ExecutionConfig(
        execution_mode=execution_mode,
        dry_run=dry_run,
        order_id_prefix="BQL_",
        state_path=str(tmp_path / "execution_state.json"),
        audit_log_path=str(tmp_path / "execution_audit.jsonl"),
        horizon_bars=horizon_bars,
        timeframe=timeframe,
    )


def _read_audit_events(audit_path: str) -> List[Dict[str, Any]]:
    with open(audit_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------
# 1. Bloqueo de Mainnet
# ----------------------------------------------------------------------
def test_mainnet_is_strictly_blocked_with_permission_error(tmp_path):
    with pytest.raises(PermissionError):
        ExecutionConfig(
            execution_mode="mainnet",
            dry_run=True,
            state_path=str(tmp_path / "state.json"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
        )


# ----------------------------------------------------------------------
# 2. DRY_RUN=True por defecto no envía órdenes ni cancela nada
# ----------------------------------------------------------------------
def test_dry_run_default_never_sends_orders_or_cancels(tmp_path):
    cfg = ExecutionConfig(
        state_path=str(tmp_path / "state.json"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    assert cfg.execution_mode == "paper"
    assert cfg.dry_run is True

    exchange = InMemoryTestnetExchange(mark_price=100000.0)
    exchange.seed_open_order("BTC/USDT:USDT", "STOP_MARKET", "sell", 0.05, "BQL_SL_OLD")
    engine = SafeExecutionEngine(cfg, exchange_client=exchange)

    res = engine.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="LONG",
        qty_btc=0.05,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_DRY_1",
    )
    assert res["status"] == "DRY_RUN"
    assert len(exchange.order_history) == 1  # solo la orden sembrada inicialmente
    assert len(exchange.open_orders) == 1

    canceled = engine.cancel_own_open_orders("BTC/USDT:USDT")
    assert canceled == []
    assert len(exchange.open_orders) == 1
    assert exchange.cancel_all_called is False

    events = [e["event"] for e in _read_audit_events(cfg.audit_log_path)]
    assert "DRY_RUN_ENTRY" in events
    assert "DRY_RUN_CANCEL" in events


# ----------------------------------------------------------------------
# 3. Cancelación selectiva: órdenes manuales ajenas permanecen intactas
# ----------------------------------------------------------------------
def test_selective_cancellation_preserves_foreign_manual_orders(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False)
    exchange = InMemoryTestnetExchange(mark_price=100000.0)
    manual_order = exchange.seed_open_order(
        "BTC/USDT:USDT", "LIMIT", "buy", 0.10, "MANUAL_USER_1"
    )
    own_sl = exchange.seed_open_order(
        "BTC/USDT:USDT", "STOP_MARKET", "sell", 0.05, "BQL_SL_PREV"
    )
    own_tp = exchange.seed_open_order(
        "BTC/USDT:USDT", "TAKE_PROFIT_MARKET", "sell", 0.05, "BQL_TP_PREV"
    )

    engine = SafeExecutionEngine(cfg, exchange_client=exchange)
    assert engine.is_own_order(manual_order) is False
    assert engine.is_own_order(own_sl) is True

    canceled = engine.cancel_own_open_orders("BTC/USDT:USDT")
    assert set(canceled) == {own_sl["id"], own_tp["id"]}
    assert exchange.cancel_all_called is False
    assert manual_order["id"] in exchange.open_orders
    assert len(exchange.open_orders) == 1


# ----------------------------------------------------------------------
# 4. Persistencia de entry_time entre reinicios (no extiende Time-Stop)
# ----------------------------------------------------------------------
def test_entry_epoch_persists_across_restart_for_time_stop(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False, horizon_bars=24, timeframe="15m")
    exchange = InMemoryTestnetExchange(mark_price=95000.0)
    manual_order = exchange.seed_open_order(
        "BTC/USDT:USDT", "LIMIT", "buy", 0.25, "MANUAL_USER_1"
    )

    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    engine_v1 = SafeExecutionEngine(cfg, exchange_client=exchange)
    entry_res = engine_v1.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="LONG",
        qty_btc=0.05,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_PERSIST_1",
        now_utc=t0,
    )
    assert entry_res["status"] == "FILLED"
    assert os.path.exists(cfg.state_path)

    # Simulamos reinicio del proceso: nueva instancia cargando el mismo state_path
    engine_v2 = SafeExecutionEngine(cfg, exchange_client=exchange)
    assert engine_v2.state["active_position"]["entry_epoch"] == pytest.approx(t0.timestamp())

    # 1 minuto antes de cumplir las 6 horas (24 * 15m = 360m) -> sigue OPEN
    t_before = t0 + timedelta(minutes=359)
    rec_before = engine_v2.reconcile("BTC/USDT:USDT", now_utc=t_before)
    assert rec_before["status"] == "OPEN"
    assert rec_before["open_contracts"] == pytest.approx(0.05)

    # Exactamente al cumplirse las 6 horas desde t0 -> Time-Stop cierra posición
    t_horizon = t0 + timedelta(minutes=360)
    rec_close = engine_v2.reconcile("BTC/USDT:USDT", now_utc=t_horizon)
    assert rec_close["status"] == "TIME_STOP_CLOSED"
    assert rec_close["closed_contracts"] == pytest.approx(0.05)
    assert rec_close["close_order"]["clientOrderId"].startswith("BQL_TS_")
    assert rec_close["close_order"]["params"]["reduceOnly"] is True

    # Posición en 0 contratos, órdenes BQL canceladas, orden manual ajena intacta
    pos = exchange.fetch_positions(["BTC/USDT:USDT"])[0]
    assert pos["contracts"] == 0.0
    assert manual_order["id"] in exchange.open_orders
    assert len(exchange.open_orders) == 1
    assert exchange.cancel_all_called is False


# ----------------------------------------------------------------------
# 5. Idempotencia: repetir signal_id devuelve DUPLICATE_IGNORED
# ----------------------------------------------------------------------
def test_idempotency_ignores_duplicate_signal_id(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False)
    exchange = InMemoryTestnetExchange(mark_price=90000.0)
    engine = SafeExecutionEngine(cfg, exchange_client=exchange)

    t0 = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)
    first = engine.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="SHORT",
        qty_btc=0.03,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_DUP_42",
        now_utc=t0,
    )
    assert first["status"] == "FILLED"
    orders_after_first = len(exchange.order_history)

    # Repetir en la misma instancia
    dup1 = engine.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="SHORT",
        qty_btc=0.03,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_DUP_42",
        now_utc=t0 + timedelta(seconds=5),
    )
    assert dup1 == {"status": "DUPLICATE_IGNORED", "signal_id": "SIG_DUP_42"}
    assert len(exchange.order_history) == orders_after_first

    # Repetir tras reinicio (persistido en disco)
    engine_restarted = SafeExecutionEngine(cfg, exchange_client=exchange)
    dup2 = engine_restarted.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="SHORT",
        qty_btc=0.03,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_DUP_42",
        now_utc=t0 + timedelta(seconds=10),
    )
    assert dup2 == {"status": "DUPLICATE_IGNORED", "signal_id": "SIG_DUP_42"}
    assert len(exchange.order_history) == orders_after_first


# ----------------------------------------------------------------------
# 6. Llenado parcial: SL y TP se colocan por `filled` (0.04 de 0.10)
# ----------------------------------------------------------------------
def test_partial_fill_anchors_sl_and_tp_to_actual_filled_qty(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False)
    exchange = InMemoryTestnetExchange(mark_price=100000.0, partial_fill_qty=0.04)
    engine = SafeExecutionEngine(cfg, exchange_client=exchange)

    res = engine.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="LONG",
        qty_btc=0.10,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_PARTIAL_1",
    )
    assert res["status"] == "FILLED"
    assert res["requested_qty_btc"] == pytest.approx(0.10)
    assert res["filled_qty_btc"] == pytest.approx(0.04)
    assert res["sl_order"]["amount"] == pytest.approx(0.04)
    assert res["sl_order"]["reduceOnly"] is True
    assert res["sl_order"]["workingType"] == "MARK_PRICE"
    assert res["sl_order"]["clientOrderId"] == "BQL_SL_SIG_PARTIAL_1"
    assert res["tp_order"]["amount"] == pytest.approx(0.04)
    assert res["tp_order"]["reduceOnly"] is True
    assert res["tp_order"]["workingType"] == "MARK_PRICE"
    assert res["tp_order"]["clientOrderId"] == "BQL_TP_SIG_PARTIAL_1"
    assert engine.state["active_position"]["qty_btc"] == pytest.approx(0.04)


def test_unfilled_entry_returns_unfilled_without_placing_sl_or_tp(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False)
    exchange = InMemoryTestnetExchange(mark_price=100000.0, partial_fill_qty=0.0)
    engine = SafeExecutionEngine(cfg, exchange_client=exchange)

    res = engine.place_bracket_order(
        symbol="BTC/USDT:USDT",
        side="LONG",
        qty_btc=0.10,
        atr_pct=0.01,
        sl_mult=1.0,
        tp_mult=2.0,
        signal_id="SIG_UNFILLED_1",
    )
    assert res["status"] == "UNFILLED"
    assert len(exchange.open_orders) == 0
    assert engine.state["active_position"] is None


# ----------------------------------------------------------------------
# 7. Fallo de STOP_MARKET dispara rollback exacto por `filled`
# ----------------------------------------------------------------------
def test_stop_market_failure_triggers_exact_rollback_leaving_zero_contracts(tmp_path):
    cfg = _make_config(tmp_path, dry_run=False)
    exchange = InMemoryTestnetExchange(
        mark_price=100000.0,
        partial_fill_qty=0.04,
        fail_stop_market=True,
    )
    engine = SafeExecutionEngine(cfg, exchange_client=exchange)

    with pytest.raises(RuntimeError, match="rollback"):
        engine.place_bracket_order(
            symbol="BTC/USDT:USDT",
            side="LONG",
            qty_btc=0.10,
            atr_pct=0.01,
            sl_mult=1.0,
            tp_mult=2.0,
            signal_id="SIG_RB_1",
        )

    # Posición neta en el exchange debe quedar exactamente en 0 contratos
    pos = exchange.fetch_positions(["BTC/USDT:USDT"])[0]
    assert pos["contracts"] == 0.0
    assert engine.state["active_position"] is None

    # Verificamos la orden de rollback en el historial
    rollback_orders = [
        o for o in exchange.order_history if o["clientOrderId"] == "BQL_RB_SIG_RB_1"
    ]
    assert len(rollback_orders) == 1
    rb = rollback_orders[0]
    assert rb["type"] == "market"
    assert rb["side"] == "sell"
    assert rb["amount"] == pytest.approx(0.04)
    assert rb["params"]["reduceOnly"] is True

    # Auditoría JSONL debe registrar EMERGENCY_ROLLBACK
    events = [e["event"] for e in _read_audit_events(cfg.audit_log_path)]
    assert "EMERGENCY_ROLLBACK" in events


# ----------------------------------------------------------------------
# 8. Verificación de endurecimiento en live_testnet_bot.py (AUD-08)
# ----------------------------------------------------------------------
def test_live_bot_reconcile_respects_dry_run():
    bot = object.__new__(BinanceFuturesTestnetBot)
    bot.dry_run = True
    bot.active_exchange_entry_ts = time.time() - (24 * 15 * 60 + 60)

    exchange = InMemoryTestnetExchange(mark_price=100000.0)
    exchange.create_order(
        "BTC/USDT:USDT", "market", "buy", 0.05, None, {"newClientOrderId": "BQL_ENT_1"}
    )
    bot.execution_client = exchange

    # Con dry_run=True no debe cancelar órdenes ni cerrar la posición por time-stop
    open_count = bot.reconcile_exchange_state()
    assert open_count == 1
    assert exchange.cancel_all_called is False
    assert exchange.fetch_positions(["BTC/USDT:USDT"])[0]["contracts"] == pytest.approx(0.05)


def test_live_bot_pauses_when_data_source_degraded():
    bot = object.__new__(BinanceFuturesTestnetBot)
    bot.execution_mode = "paper"
    bot.model_name = "TestModel"
    bot.strict_futures_data = True
    bot.last_data_source = None
    bot.monitor_realtime_positions = lambda: None

    def _degraded_fetch(symbol: str, limit: int = 1500, include_forming_bar: bool = False):
        bot.last_data_source = "vision_spot_degraded"
        idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=idx,
        )

    bot.fetch_live_dataframe = _degraded_fetch
    summary = bot.evaluate_and_execute()
    assert summary["action"] == "DATA_SOURCE_DEGRADED"
    assert summary["data_source"] == "vision_spot_degraded"
