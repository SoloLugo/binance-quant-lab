import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from config import settings


class RealTimeFuturesSimulator:
    """
    Motor de Simulación en Tiempo Real (Paper Trading Institucional) para Binance USD-M Futures.
    Replica con fidelidad la microestructura real del mercado:
      1. Ejecución sobre el spread real Bid/Ask del libro de órdenes + Slippage en bps.
      2. Comisiones Taker de apertura y cierre descontadas en tiempo real.
      3. Liquidación de Funding Rate cada 8 horas (00:00, 08:00, 16:00 UTC).
      4. Monitoreo continuo de las 3 barreras: Stop Loss (Mark Price), Take Profit y Barrera Vertical (Time-Stop 6h).
      5. Circuit Breaker por Drawdown Diario Máximo y persistencia atómica en JSON + CSV.
    """
    def __init__(
        self,
        state_path: str = settings.sim_state_path,
        trade_log_path: str = settings.trade_log_path,
        initial_balance: float = settings.initial_paper_balance,
    ):
        self.state_path = state_path
        self.trade_log_path = trade_log_path
        self.initial_balance = initial_balance
        self.taker_fee_rate = settings.roundtrip_fee / 2.0  # 0.05% por lado (0.0005)
        self.slippage_rate = settings.slippage_bps / 10000.0  # 1.5 bps = 0.00015
        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(self.state_path) or "data", exist_ok=True)
        os.makedirs(os.path.dirname(self.trade_log_path) or "data", exist_ok=True)

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[WARN] No se pudo leer {self.state_path}, reiniciando estado: {exc}")

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        default_state = {
            "initial_balance": self.initial_balance,
            "wallet_balance": self.initial_balance,
            "equity": self.initial_balance,
            "peak_equity": self.initial_balance,
            "max_drawdown_pct": 0.0,
            "daily_date": today_str,
            "daily_start_equity": self.initial_balance,
            "daily_halted": False,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_fees_usdt": 0.0,
            "total_funding_usdt": 0.0,
            "cumulative_net_r": 0.0,
            "last_evaluated_bar": None,
            "open_position": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state(default_state)
        return default_state

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state is None:
            state = self.state
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def _append_trade_csv(self, record: Dict[str, Any]) -> None:
        file_exists = os.path.exists(self.trade_log_path) and os.path.getsize(self.trade_log_path) > 0
        fieldnames = list(record.keys())
        with open(self.trade_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def check_daily_circuit_breaker(self, now_utc: Optional[datetime] = None) -> bool:
        """
        Verifica y actualiza el Circuit Breaker diario.
        Si la cuenta cae más de `max_daily_drawdown_pct` en el día UTC actual, bloquea nuevas entradas.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")

        if self.state.get("daily_date") != today_str:
            self.state["daily_date"] = today_str
            self.state["daily_start_equity"] = float(self.state["equity"])
            self.state["daily_halted"] = False

        start_eq = max(float(self.state.get("daily_start_equity", self.initial_balance)), 1e-6)
        current_eq = float(self.state.get("equity", self.initial_balance))
        daily_dd = (start_eq - current_eq) / start_eq

        if daily_dd >= settings.max_daily_drawdown_pct:
            self.state["daily_halted"] = True
            self._save_state()
            return True

        return bool(self.state.get("daily_halted", False))

    def has_open_position(self) -> bool:
        return self.state.get("open_position") is not None

    def open_simulated_position(
        self,
        side: str,
        qty_btc: float,
        bid_price: float,
        ask_price: float,
        mark_price: float,
        atr_pct: float,
        sl_mult: float,
        tp_mult: float,
        horizon_bars: int,
        p_win: float,
        ev_in_r: float,
        b_net_odds: float,
        kelly_risk_fraction: float,
        bar_timestamp: pd.Timestamp,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta una orden simulada a mercado aplicando el spread real Bid/Ask + Slippage + Taker Fee.
        Ancla las barreras SL y TP al precio real de ejecución (`entry_price`).
        """
        if self.has_open_position():
            raise RuntimeError("Ya existe una posición simulada abierta.")

        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        side_upper = side.upper()
        if side_upper == "LONG":
            entry_price = round(ask_price * (1.0 + self.slippage_rate), 2)
            sl_distance = entry_price * atr_pct * sl_mult
            tp_distance = entry_price * atr_pct * tp_mult
            sl_price = round(entry_price - sl_distance, 1)
            tp_price = round(entry_price + tp_distance, 1)
        else:
            entry_price = round(bid_price * (1.0 - self.slippage_rate), 2)
            sl_distance = entry_price * atr_pct * sl_mult
            tp_distance = entry_price * atr_pct * tp_mult
            sl_price = round(entry_price + sl_distance, 1)
            tp_price = round(entry_price - tp_distance, 1)

        notional_usdt = qty_btc * entry_price
        entry_fee_usdt = notional_usdt * self.taker_fee_rate
        risk_1r_usdt = max(qty_btc * sl_distance, 1e-6)

        # Descontamos la comisión Taker de apertura inmediatamente del wallet_balance
        self.state["wallet_balance"] = round(float(self.state["wallet_balance"]) - entry_fee_usdt, 4)
        self.state["total_fees_usdt"] = round(float(self.state["total_fees_usdt"]) + entry_fee_usdt, 4)

        horizon_seconds = horizon_bars * 15 * 60
        expiry_dt = datetime.fromtimestamp(now_utc.timestamp() + horizon_seconds, tz=timezone.utc)

        pos = {
            "trade_id": int(self.state["total_trades"]) + 1,
            "symbol": settings.primary_symbol,
            "side": side_upper,
            "qty_btc": float(qty_btc),
            "entry_price": float(entry_price),
            "mark_price": float(mark_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "sl_distance_usdt": round(float(sl_distance), 2),
            "risk_1r_usdt": round(float(risk_1r_usdt), 4),
            "notional_usdt": round(float(notional_usdt), 2),
            "effective_leverage": round(float(notional_usdt / max(float(self.state["wallet_balance"]), 1.0)), 2),
            "entry_fee_usdt": round(float(entry_fee_usdt), 4),
            "funding_paid_usdt": 0.0,
            "last_funding_ts": int(now_utc.timestamp() // (8 * 3600)) * (8 * 3600),
            "p_win": round(float(p_win), 4),
            "ev_in_r": round(float(ev_in_r), 4),
            "b_net_odds": round(float(b_net_odds), 4),
            "kelly_risk_fraction": round(float(kelly_risk_fraction), 4),
            "bar_timestamp": str(bar_timestamp),
            "entry_time": now_utc.isoformat(),
            "entry_epoch": float(now_utc.timestamp()),
            "horizon_expiry_time": expiry_dt.isoformat(),
            "horizon_expiry_epoch": float(expiry_dt.timestamp()),
            "unrealized_pnl_usdt": 0.0,
            "unrealized_r": 0.0,
        }

        self.state["open_position"] = pos
        self.state["equity"] = round(float(self.state["wallet_balance"]), 4)
        self._save_state()
        return pos

    def update_on_market_tick(
        self,
        mark_price: float,
        bid_price: float,
        ask_price: float,
        funding_rate: float = 0.0001,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Actualiza el PnL no realizado, liquida el Funding Rate si se cruzó una ventana de 8h
        (00:00, 08:00, 16:00 UTC) y verifica si se activó el Stop Loss, Take Profit o Time-Stop (24 velas).
        Retorna el registro del trade cerrado si alguna barrera fue alcanzada, o None si sigue abierta.
        """
        pos = self.state.get("open_position")
        if pos is None:
            return None

        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        now_ts = float(now_utc.timestamp())
        side = pos["side"]
        qty = float(pos["qty_btc"])
        entry_price = float(pos["entry_price"])
        sl_price = float(pos["sl_price"])
        tp_price = float(pos["tp_price"])
        risk_1r = max(float(pos["risk_1r_usdt"]), 1e-6)

        # 1. Verificación de liquidación de Funding Rate cada 8 horas (28800 s)
        current_funding_slot = int(now_ts // 28800) * 28800
        last_funding_slot = int(pos.get("last_funding_ts", current_funding_slot))
        if current_funding_slot > last_funding_slot:
            n_slots = max(1, (current_funding_slot - last_funding_slot) // 28800)
            notional_mark = qty * mark_price
            # En LONG: funding > 0 paga, funding < 0 cobra. En SHORT: inverso.
            direction_sign = 1.0 if side == "LONG" else -1.0
            funding_cost = notional_mark * float(funding_rate) * direction_sign * n_slots
            pos["funding_paid_usdt"] = round(float(pos.get("funding_paid_usdt", 0.0)) + funding_cost, 4)
            pos["last_funding_ts"] = current_funding_slot
            self.state["wallet_balance"] = round(float(self.state["wallet_balance"]) - funding_cost, 4)
            self.state["total_funding_usdt"] = round(float(self.state["total_funding_usdt"]) + funding_cost, 4)

        # 2. Evaluación de barreras de salida (SL, TP y Barrera Vertical de Tiempo)
        high_check = max(mark_price, candle_high if candle_high is not None else mark_price)
        low_check = min(mark_price, candle_low if candle_low is not None else mark_price)

        exit_reason = None
        raw_exit_price = None

        if side == "LONG":
            # Conservador: verificamos SL antes que TP
            if mark_price <= sl_price or low_check <= sl_price:
                exit_reason = "STOP_LOSS"
                raw_exit_price = sl_price * (1.0 - self.slippage_rate)
            elif mark_price >= tp_price or high_check >= tp_price:
                exit_reason = "TAKE_PROFIT"
                raw_exit_price = tp_price
            elif now_ts >= float(pos["horizon_expiry_epoch"]):
                exit_reason = "TIME_BARRIER"
                raw_exit_price = bid_price * (1.0 - self.slippage_rate)
        else:
            if mark_price >= sl_price or high_check >= sl_price:
                exit_reason = "STOP_LOSS"
                raw_exit_price = sl_price * (1.0 + self.slippage_rate)
            elif mark_price <= tp_price or low_check <= tp_price:
                exit_reason = "TAKE_PROFIT"
                raw_exit_price = tp_price
            elif now_ts >= float(pos["horizon_expiry_epoch"]):
                exit_reason = "TIME_BARRIER"
                raw_exit_price = ask_price * (1.0 + self.slippage_rate)

        if exit_reason is not None and raw_exit_price is not None:
            return self.close_simulated_position(
                exit_price=round(float(raw_exit_price), 2),
                exit_reason=exit_reason,
                now_utc=now_utc
            )

        # 3. Si sigue abierta, actualizamos PnL flotante (Mark-to-Market)
        if side == "LONG":
            gross_unrealized = (mark_price - entry_price) * qty
        else:
            gross_unrealized = (entry_price - mark_price) * qty

        est_exit_fee = (qty * mark_price) * self.taker_fee_rate
        net_unrealized = gross_unrealized - float(pos["entry_fee_usdt"]) - est_exit_fee - float(pos["funding_paid_usdt"])

        pos["mark_price"] = round(float(mark_price), 2)
        pos["unrealized_pnl_usdt"] = round(float(net_unrealized), 4)
        pos["unrealized_r"] = round(float(net_unrealized / risk_1r), 3)

        equity = float(self.state["wallet_balance"]) + gross_unrealized - est_exit_fee
        self.state["equity"] = round(equity, 4)
        self.state["peak_equity"] = round(max(float(self.state["peak_equity"]), equity), 4)
        dd_pct = (float(self.state["peak_equity"]) - equity) / max(float(self.state["peak_equity"]), 1e-6)
        self.state["max_drawdown_pct"] = round(max(float(self.state["max_drawdown_pct"]), dd_pct), 4)

        self.check_daily_circuit_breaker(now_utc)
        self._save_state()
        return None

    def close_simulated_position(
        self,
        exit_price: float,
        exit_reason: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Cierra la posición simulada activa, descuenta la comisión Taker de salida,
        actualiza las métricas de cuenta y escribe el registro en el CSV de auditoría.
        """
        pos = self.state.get("open_position")
        if pos is None:
            raise RuntimeError("No hay posición abierta para cerrar.")

        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        side = pos["side"]
        qty = float(pos["qty_btc"])
        entry_price = float(pos["entry_price"])
        risk_1r = max(float(pos["risk_1r_usdt"]), 1e-6)

        if side == "LONG":
            gross_pnl_usdt = (exit_price - entry_price) * qty
        else:
            gross_pnl_usdt = (entry_price - exit_price) * qty

        exit_fee_usdt = (qty * exit_price) * self.taker_fee_rate
        entry_fee_usdt = float(pos["entry_fee_usdt"])
        funding_paid_usdt = float(pos.get("funding_paid_usdt", 0.0))

        # El entry_fee y el funding ya fueron descontados del wallet_balance al ocurrir
        self.state["wallet_balance"] = round(
            float(self.state["wallet_balance"]) + gross_pnl_usdt - exit_fee_usdt, 4
        )
        self.state["total_fees_usdt"] = round(float(self.state["total_fees_usdt"]) + exit_fee_usdt, 4)

        net_pnl_usdt = gross_pnl_usdt - entry_fee_usdt - exit_fee_usdt - funding_paid_usdt
        realized_net_r = net_pnl_usdt / risk_1r

        self.state["equity"] = self.state["wallet_balance"]
        self.state["peak_equity"] = round(
            max(float(self.state["peak_equity"]), float(self.state["equity"])), 4
        )
        dd_pct = (float(self.state["peak_equity"]) - float(self.state["equity"])) / max(
            float(self.state["peak_equity"]), 1e-6
        )
        self.state["max_drawdown_pct"] = round(max(float(self.state["max_drawdown_pct"]), dd_pct), 4)

        self.state["total_trades"] = int(self.state["total_trades"]) + 1
        if net_pnl_usdt > 0:
            self.state["winning_trades"] = int(self.state["winning_trades"]) + 1
        else:
            self.state["losing_trades"] = int(self.state["losing_trades"]) + 1

        self.state["cumulative_net_r"] = round(float(self.state["cumulative_net_r"]) + realized_net_r, 4)
        self.state["open_position"] = None

        trade_record = {
            "trade_id": pos["trade_id"],
            "symbol": pos["symbol"],
            "side": side,
            "qty_btc": qty,
            "entry_time": pos["entry_time"],
            "exit_time": now_utc.isoformat(),
            "exit_reason": exit_reason,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl_price": pos["sl_price"],
            "tp_price": pos["tp_price"],
            "p_win_pred": pos["p_win"],
            "ev_pred_r": pos["ev_in_r"],
            "gross_pnl_usdt": round(gross_pnl_usdt, 4),
            "total_fees_usdt": round(entry_fee_usdt + exit_fee_usdt, 4),
            "funding_paid_usdt": round(funding_paid_usdt, 4),
            "net_pnl_usdt": round(net_pnl_usdt, 4),
            "realized_net_r": round(realized_net_r, 4),
            "balance_after_usdt": self.state["wallet_balance"],
        }

        self._append_trade_csv(trade_record)
        self.check_daily_circuit_breaker(now_utc)
        self._save_state()
        return trade_record

    def summary_dict(self) -> Dict[str, Any]:
        total = int(self.state.get("total_trades", 0))
        wins = int(self.state.get("winning_trades", 0))
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        pnl_pct = (
            (float(self.state["equity"]) - float(self.state["initial_balance"]))
            / max(float(self.state["initial_balance"]), 1e-6)
            * 100.0
        )
        return {
            "equity_usdt": round(float(self.state["equity"]), 2),
            "wallet_balance_usdt": round(float(self.state["wallet_balance"]), 2),
            "net_return_pct": round(pnl_pct, 2),
            "cumulative_net_r": round(float(self.state["cumulative_net_r"]), 3),
            "total_trades": total,
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(float(self.state["max_drawdown_pct"]) * 100.0, 2),
            "total_fees_usdt": round(float(self.state["total_fees_usdt"]), 2),
            "total_funding_usdt": round(float(self.state["total_funding_usdt"]), 2),
            "daily_halted": bool(self.state.get("daily_halted", False)),
            "has_open_position": self.has_open_position(),
        }
