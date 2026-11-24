"""
Motor de Ejecución Segura en Testnet para Binance Quant Lab (Fase 14).

Implementa:
- Bloqueo permanente de Mainnet (via validate_execution_mode).
- DRY_RUN=True por defecto.
- Idempotencia de señales mediante persistencia atómica en disco.
- Persistencia de `entry_epoch` entre reinicios para respetar el horizonte Time-Stop.
- Cancelación selectiva de órdenes propias (`order_id_prefix`, nunca `cancel_all_orders`).
- Manejo seguro de llenados parciales (`filled` real) para SL/TP.
- Rollback de emergencia exacto si falla la colocación de `STOP_MARKET`.
- Registro de auditoría estructurado en JSONL.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import validate_execution_mode
from data_quality import timeframe_to_timedelta


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuración inmutable y validada del motor de ejecución segura."""
    execution_mode: str = "paper"
    dry_run: bool = True
    order_id_prefix: str = "BQL_"
    state_path: str = "data/execution_state.json"
    audit_log_path: str = "data/execution_audit.jsonl"
    horizon_bars: int = 24
    timeframe: str = "15m"
    max_notional_usdt: float = 50000.0
    max_leverage: int = 5

    def __post_init__(self) -> None:
        validate_execution_mode(self.execution_mode)
        timeframe_to_timedelta(self.timeframe)
        if not self.order_id_prefix or not self.order_id_prefix.strip():
            raise ValueError("order_id_prefix no puede estar vacío.")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars debe ser mayor que 0.")
        if self.max_notional_usdt <= 0:
            raise ValueError("max_notional_usdt debe ser mayor que 0.")
        if self.max_leverage < 1:
            raise ValueError("max_leverage debe ser >= 1.")


def _ensure_utc(now_utc: Optional[datetime]) -> datetime:
    if now_utc is None:
        return datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)


class SafeExecutionEngine:
    """
    Motor de ejecución segura con persistencia atómica, idempotencia,
    cancelación selectiva de órdenes propias, soporte de llenados parciales
    y rollback de emergencia ante fallos de Stop Loss.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        exchange_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.exchange_client = exchange_client
        self.state: Dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------
    # Persistencia atómica y auditoría JSONL
    # ------------------------------------------------------------------
    def _default_state(self) -> Dict[str, Any]:
        return {
            "active_position": None,
            "executed_signal_ids": [],
        }

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.config.state_path):
            with open(self.config.state_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                loaded.setdefault("active_position", None)
                loaded.setdefault("executed_signal_ids", [])
                return loaded
        return self._default_state()

    def _save_state(self) -> None:
        state_dir = os.path.dirname(os.path.abspath(self.config.state_path))
        os.makedirs(state_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".exec_state_", suffix=".tmp", dir=state_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.config.state_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _append_audit(
        self,
        event: str,
        details: Dict[str, Any],
        now_utc: Optional[datetime] = None,
    ) -> None:
        audit_dir = os.path.dirname(os.path.abspath(self.config.audit_log_path))
        os.makedirs(audit_dir, exist_ok=True)
        record = {
            "timestamp_utc": _ensure_utc(now_utc).isoformat(),
            "event": event,
            "execution_mode": self.config.execution_mode,
            "dry_run": self.config.dry_run,
            "details": details,
        }
        with open(self.config.audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            fh.flush()

    def _require_exchange_client(self) -> Any:
        if self.exchange_client is None:
            raise RuntimeError(
                "exchange_client es requerido para operar cuando dry_run=False."
            )
        return self.exchange_client

    # ------------------------------------------------------------------
    # Identificación y cancelación selectiva de órdenes propias
    # ------------------------------------------------------------------
    def is_own_order(self, order: Dict[str, Any]) -> bool:
        """
        Retorna True únicamente si `clientOrderId` (o `info.clientOrderId`)
        comienza con `config.order_id_prefix`.
        """
        if not isinstance(order, dict):
            return False
        cid = order.get("clientOrderId")
        if not cid:
            info = order.get("info")
            if isinstance(info, dict):
                cid = info.get("clientOrderId")
        return isinstance(cid, str) and cid.startswith(self.config.order_id_prefix)

    def cancel_own_open_orders(
        self,
        symbol: str,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        """
        Obtiene las órdenes abiertas para `symbol` y cancela ÚNICAMENTE aquellas
        que pertenecen a este bot (`is_own_order`). JAMÁS llama a `cancel_all_orders`.
        """
        if self.config.dry_run:
            self._append_audit(
                "DRY_RUN_CANCEL",
                {"symbol": symbol, "canceled_order_ids": []},
                now_utc=now_utc,
            )
            return []

        client = self._require_exchange_client()
        open_orders = client.fetch_open_orders(symbol) or []
        canceled_ids: List[str] = []
        skipped_foreign = 0

        for order in open_orders:
            if not self.is_own_order(order):
                skipped_foreign += 1
                continue
            order_id = str(order.get("id") or order.get("orderId") or "")
            if not order_id:
                continue
            client.cancel_order(order_id, symbol)
            canceled_ids.append(order_id)

        self._append_audit(
            "CANCEL_OWN_ORDERS",
            {
                "symbol": symbol,
                "canceled_order_ids": canceled_ids,
                "skipped_foreign_orders": skipped_foreign,
            },
            now_utc=now_utc,
        )
        return canceled_ids

    # ------------------------------------------------------------------
    # Colocación de orden bracket con idempotencia, partial fill y rollback
    # ------------------------------------------------------------------
    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty_btc: float,
        atr_pct: float,
        sl_mult: float,
        tp_mult: float,
        signal_id: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Coloca una entrada a mercado con Stop Loss (`STOP_MARKET`) y Take Profit
        (`TAKE_PROFIT_MARKET`) sobre el volumen realmente ejecutado (`filled`).
        """
        now_dt = _ensure_utc(now_utc)
        side_upper = side.upper().strip()
        if side_upper not in ("LONG", "SHORT"):
            raise ValueError(f"side inválido: '{side}'. Debe ser 'LONG' o 'SHORT'.")
        if qty_btc <= 0:
            raise ValueError("qty_btc debe ser mayor que 0.")
        if atr_pct <= 0 or sl_mult <= 0 or tp_mult <= 0:
            raise ValueError("atr_pct, sl_mult y tp_mult deben ser mayores que 0.")
        if not signal_id or not str(signal_id).strip():
            raise ValueError("signal_id no puede estar vacío.")

        # 1. Idempotencia
        if signal_id in self.state["executed_signal_ids"]:
            self._append_audit(
                "DUPLICATE_IGNORED",
                {"symbol": symbol, "side": side_upper, "signal_id": signal_id},
                now_utc=now_dt,
            )
            return {
                "status": "DUPLICATE_IGNORED",
                "signal_id": signal_id,
            }

        # 2. Modo DRY_RUN
        if self.config.dry_run:
            result = {
                "status": "DRY_RUN",
                "symbol": symbol,
                "side": side_upper,
                "qty_btc": float(qty_btc),
                "atr_pct": float(atr_pct),
                "sl_mult": float(sl_mult),
                "tp_mult": float(tp_mult),
                "signal_id": signal_id,
            }
            self._append_audit("DRY_RUN_ENTRY", result, now_utc=now_dt)
            return result

        client = self._require_exchange_client()

        # 3. Cancelar únicamente órdenes propias previas
        self.cancel_own_open_orders(symbol, now_utc=now_dt)

        entry_order_side = "buy" if side_upper == "LONG" else "sell"
        exit_order_side = "sell" if side_upper == "LONG" else "buy"
        prefix = self.config.order_id_prefix

        # 4. Orden de entrada a mercado con clientOrderId propio
        entry_cid = f"{prefix}ENT_{signal_id}"
        entry_order = client.create_order(
            symbol,
            "market",
            entry_order_side,
            float(qty_btc),
            None,
            {"newClientOrderId": entry_cid},
        )

        # 5. Manejo de llenado parcial o nulo
        raw_filled = entry_order.get("filled")
        filled = float(qty_btc) if raw_filled is None else float(raw_filled)
        if filled <= 0.0:
            self._append_audit(
                "ORDER_UNFILLED",
                {
                    "symbol": symbol,
                    "side": side_upper,
                    "requested_qty_btc": float(qty_btc),
                    "filled_qty_btc": filled,
                    "signal_id": signal_id,
                    "client_order_id": entry_cid,
                },
                now_utc=now_dt,
            )
            return {
                "status": "UNFILLED",
                "signal_id": signal_id,
                "filled_qty_btc": filled,
            }

        fill_price = float(entry_order.get("average") or entry_order.get("price") or 0.0)
        if fill_price <= 0.0 and hasattr(client, "fetch_ticker"):
            ticker = client.fetch_ticker(symbol) or {}
            fill_price = float(ticker.get("last") or ticker.get("close") or 0.0)
        if fill_price <= 0.0:
            raise RuntimeError(
                f"No se pudo determinar fill_price válido para la orden {entry_cid}."
            )

        sl_dist = fill_price * atr_pct * sl_mult
        tp_dist = fill_price * atr_pct * tp_mult
        if side_upper == "LONG":
            sl_price = round(fill_price - sl_dist, 1)
            tp_price = round(fill_price + tp_dist, 1)
        else:
            sl_price = round(fill_price + sl_dist, 1)
            tp_price = round(fill_price - tp_dist, 1)

        # 6. Colocar STOP_MARKET sobre `filled` con rollback de emergencia si falla
        sl_cid = f"{prefix}SL_{signal_id}"
        try:
            sl_order = client.create_order(
                symbol,
                "STOP_MARKET",
                exit_order_side,
                filled,
                None,
                {
                    "stopPrice": sl_price,
                    "workingType": "MARK_PRICE",
                    "reduceOnly": True,
                    "newClientOrderId": sl_cid,
                },
            )
        except Exception as sl_exc:
            rb_cid = f"{prefix}RB_{signal_id}"
            rb_order = client.create_order(
                symbol,
                "market",
                exit_order_side,
                filled,
                None,
                {
                    "reduceOnly": True,
                    "newClientOrderId": rb_cid,
                },
            )
            self._append_audit(
                "EMERGENCY_ROLLBACK",
                {
                    "symbol": symbol,
                    "side": side_upper,
                    "signal_id": signal_id,
                    "filled_qty_btc": filled,
                    "rollback_client_order_id": rb_cid,
                    "rollback_order_id": rb_order.get("id") if isinstance(rb_order, dict) else None,
                    "sl_error": str(sl_exc),
                },
                now_utc=now_dt,
            )
            self.state["active_position"] = None
            self._save_state()
            raise RuntimeError(
                f"Posición cerrada por rollback de emergencia tras fallo en STOP_MARKET: {sl_exc}"
            ) from sl_exc

        # 7. Colocar TAKE_PROFIT_MARKET sobre `filled`
        tp_cid = f"{prefix}TP_{signal_id}"
        tp_order: Optional[Dict[str, Any]] = None
        try:
            tp_order = client.create_order(
                symbol,
                "TAKE_PROFIT_MARKET",
                exit_order_side,
                filled,
                None,
                {
                    "stopPrice": tp_price,
                    "workingType": "MARK_PRICE",
                    "reduceOnly": True,
                    "newClientOrderId": tp_cid,
                },
            )
        except Exception as tp_exc:
            self._append_audit(
                "TP_ORDER_WARNING",
                {
                    "symbol": symbol,
                    "side": side_upper,
                    "signal_id": signal_id,
                    "filled_qty_btc": filled,
                    "tp_error": str(tp_exc),
                },
                now_utc=now_dt,
            )

        # 8. Persistencia atómica de posición activa y signal_id
        entry_epoch = now_dt.timestamp()
        active_pos = {
            "symbol": symbol,
            "side": side_upper,
            "qty_btc": filled,
            "entry_price": fill_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "signal_id": signal_id,
            "entry_epoch": entry_epoch,
        }
        self.state["active_position"] = active_pos
        self.state["executed_signal_ids"].append(signal_id)
        self._save_state()

        self._append_audit(
            "ENTRY_FILLED",
            {
                "symbol": symbol,
                "side": side_upper,
                "requested_qty_btc": float(qty_btc),
                "filled_qty_btc": filled,
                "fill_price": fill_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "signal_id": signal_id,
                "entry_epoch": entry_epoch,
                "entry_client_order_id": entry_cid,
                "sl_client_order_id": sl_cid,
                "tp_client_order_id": tp_cid,
            },
            now_utc=now_dt,
        )

        return {
            "status": "FILLED",
            "symbol": symbol,
            "side": side_upper,
            "requested_qty_btc": float(qty_btc),
            "filled_qty_btc": filled,
            "fill_price": fill_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_epoch": entry_epoch,
            "signal_id": signal_id,
            "entry_order": entry_order,
            "sl_order": sl_order,
            "tp_order": tp_order,
        }

    # ------------------------------------------------------------------
    # Reconciliación con persistencia de entry_epoch y Time-Stop
    # ------------------------------------------------------------------
    def reconcile(
        self,
        symbol: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Reconcilia el estado de posiciones y órdenes en el exchange:
        - Si no hay posición abierta: limpia `active_position` en disco y cancela
          únicamente órdenes propias (sin tocar órdenes ajenas).
        - Si hay posición abierta: usa el `entry_epoch` persistido en disco para evaluar
          el Time-Stop (`horizon_bars * timeframe_seconds`), respetando `dry_run`.
        """
        now_dt = _ensure_utc(now_utc)
        now_ts = now_dt.timestamp()

        if self.exchange_client is None:
            if self.config.dry_run:
                self._append_audit(
                    "DRY_RUN_RECONCILE",
                    {"symbol": symbol, "reason": "no_exchange_client"},
                    now_utc=now_dt,
                )
                return {"status": "DRY_RUN_NO_CLIENT", "open_contracts": 0.0}
            raise RuntimeError(
                "exchange_client es requerido para reconciliar cuando dry_run=False."
            )

        positions = self.exchange_client.fetch_positions([symbol]) or []
        active_positions = [
            p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0.0
        ]

        # Caso 1: Sin posición abierta en el exchange
        if not active_positions:
            self.state["active_position"] = None
            self._save_state()
            canceled_ids = self.cancel_own_open_orders(symbol, now_utc=now_dt)
            self._append_audit(
                "RECONCILE_FLAT",
                {"symbol": symbol, "canceled_order_ids": canceled_ids},
                now_utc=now_dt,
            )
            return {
                "status": "FLAT",
                "symbol": symbol,
                "open_contracts": 0.0,
                "canceled_order_ids": canceled_ids,
            }

        # Caso 2: Posición abierta en el exchange
        pos = active_positions[0]
        contracts = abs(float(pos.get("contracts", 0) or 0))
        side_str = str(pos.get("side", "long")).upper()

        persisted = self.state.get("active_position")
        if isinstance(persisted, dict) and persisted.get("entry_epoch") is not None:
            entry_epoch = float(persisted["entry_epoch"])
        else:
            pos_ts = pos.get("timestamp")
            entry_epoch = (float(pos_ts) / 1000.0) if pos_ts else now_ts
            self.state["active_position"] = {
                "symbol": symbol,
                "side": side_str,
                "qty_btc": contracts,
                "entry_price": float(pos.get("entryPrice") or pos.get("markPrice") or 0.0),
                "sl_price": None,
                "tp_price": None,
                "signal_id": None,
                "entry_epoch": entry_epoch,
            }
            self._save_state()

        tf_seconds = timeframe_to_timedelta(self.config.timeframe).total_seconds()
        horizon_seconds = float(self.config.horizon_bars) * tf_seconds
        elapsed_seconds = now_ts - entry_epoch

        if elapsed_seconds >= horizon_seconds:
            if self.config.dry_run:
                self._append_audit(
                    "DRY_RUN_TIME_STOP",
                    {
                        "symbol": symbol,
                        "side": side_str,
                        "open_contracts": contracts,
                        "entry_epoch": entry_epoch,
                        "elapsed_seconds": elapsed_seconds,
                        "horizon_seconds": horizon_seconds,
                    },
                    now_utc=now_dt,
                )
                return {
                    "status": "DRY_RUN_TIME_STOP",
                    "symbol": symbol,
                    "side": side_str,
                    "open_contracts": contracts,
                    "entry_epoch": entry_epoch,
                    "elapsed_seconds": elapsed_seconds,
                    "horizon_seconds": horizon_seconds,
                }

            canceled_ids = self.cancel_own_open_orders(symbol, now_utc=now_dt)
            close_side = "sell" if side_str == "LONG" else "buy"
            ts_cid = f"{self.config.order_id_prefix}TS_{int(now_ts)}"
            close_order = self.exchange_client.create_order(
                symbol,
                "market",
                close_side,
                contracts,
                None,
                {
                    "reduceOnly": True,
                    "newClientOrderId": ts_cid,
                },
            )
            self.state["active_position"] = None
            self._save_state()
            self._append_audit(
                "TIME_STOP_CLOSED",
                {
                    "symbol": symbol,
                    "side": side_str,
                    "closed_contracts": contracts,
                    "entry_epoch": entry_epoch,
                    "elapsed_seconds": elapsed_seconds,
                    "horizon_seconds": horizon_seconds,
                    "canceled_order_ids": canceled_ids,
                    "close_client_order_id": ts_cid,
                },
                now_utc=now_dt,
            )
            return {
                "status": "TIME_STOP_CLOSED",
                "symbol": symbol,
                "side": side_str,
                "closed_contracts": contracts,
                "canceled_order_ids": canceled_ids,
                "close_order": close_order,
                "elapsed_seconds": elapsed_seconds,
                "horizon_seconds": horizon_seconds,
            }

        return {
            "status": "OPEN",
            "symbol": symbol,
            "side": side_str,
            "open_contracts": contracts,
            "entry_epoch": entry_epoch,
            "elapsed_seconds": elapsed_seconds,
            "horizon_seconds": horizon_seconds,
        }
