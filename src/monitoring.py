"""
Monitoring y Journal estructurado para Binance Quant Lab (Fase 13).
Registra señales emitidas, decisiones del Risk Engine, comparativa previsto vs realizado
en ejecución, errores operativos y estado de degradación de datos en formato JSONL.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional

from contracts import RiskDecision


class TradingMonitor:
    """
    Monitor operativo y journal estructurado en JSONL para Paper Trading y Testnet.
    """

    def __init__(self, signals_log_path: str, execution_log_path: str):
        self.signals_log_path = signals_log_path
        self.execution_log_path = execution_log_path
        self._ensure_parent_dir(self.signals_log_path)
        self._ensure_parent_dir(self.execution_log_path)

    @staticmethod
    def _ensure_parent_dir(file_path: str) -> None:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _format_timestamp(ts: Any) -> str:
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    @staticmethod
    def _append_jsonl(file_path: str, payload: Dict[str, Any]) -> None:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            return []
        records: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))
        return records

    def record_signal(
        self,
        timestamp: Any,
        symbol: str,
        side: str,
        p_raw: float,
        p_calibrated: float,
        ev_net_r: float,
        estimated_cost_r: float,
        risk_decision: RiskDecision,
        data_status: str = "OK",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registra una señal del modelo junto con la decisión del Risk Engine."""
        record: Dict[str, Any] = {
            "event_type": "SIGNAL",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": self._format_timestamp(timestamp),
            "symbol": str(symbol),
            "side": str(side).upper(),
            "p_raw": float(p_raw),
            "p_calibrated": float(p_calibrated),
            "ev_net_r": float(ev_net_r),
            "estimated_cost_r": float(estimated_cost_r),
            "approved": bool(risk_decision.approved),
            "risk_reason": str(risk_decision.reason),
            "qty_btc": float(risk_decision.qty_btc),
            "risk_fraction": float(risk_decision.risk_fraction),
            "effective_leverage": float(risk_decision.effective_leverage),
            "data_status": str(data_status),
        }
        if extra is not None:
            record["extra"] = dict(extra)

        self._append_jsonl(self.signals_log_path, record)
        return record

    def record_execution_comparison(
        self,
        trade_id: Any,
        side: str,
        expected_entry_price: float,
        actual_entry_price: float,
        expected_ev_r: float,
        realized_net_r: float,
        expected_cost_r: float,
        actual_cost_r: float,
        exit_reason: str,
    ) -> Dict[str, Any]:
        """
        Registra la comparación entre ejecución esperada y realizada:
        - `entry_slippage_deviation_bps`: desviación adversa en bps (>0 peor que lo esperado).
        - `ev_vs_realized_gap_r`: `realized_net_r - expected_ev_r`.
        """
        side_upper = str(side).upper()
        exp_price = float(expected_entry_price)
        act_price = float(actual_entry_price)
        if exp_price > 0.0:
            if side_upper == "LONG":
                slippage_dev_bps = ((act_price - exp_price) / exp_price) * 10000.0
            else:
                slippage_dev_bps = ((exp_price - act_price) / exp_price) * 10000.0
        else:
            slippage_dev_bps = 0.0

        ev_gap_r = float(realized_net_r) - float(expected_ev_r)
        cost_gap_r = float(actual_cost_r) - float(expected_cost_r)

        record: Dict[str, Any] = {
            "event_type": "EXECUTION_COMPARISON",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade_id,
            "side": side_upper,
            "expected_entry_price": exp_price,
            "actual_entry_price": act_price,
            "entry_slippage_deviation_bps": round(slippage_dev_bps, 4),
            "expected_ev_r": float(expected_ev_r),
            "realized_net_r": float(realized_net_r),
            "ev_vs_realized_gap_r": round(ev_gap_r, 6),
            "expected_cost_r": float(expected_cost_r),
            "actual_cost_r": float(actual_cost_r),
            "cost_gap_r": round(cost_gap_r, 6),
            "exit_reason": str(exit_reason),
        }

        self._append_jsonl(self.execution_log_path, record)
        return record

    def record_error(self, context: str, error_msg: str) -> Dict[str, Any]:
        """Registra un error operativo estructurado en el journal de ejecución."""
        record: Dict[str, Any] = {
            "event_type": "ERROR",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "context": str(context),
            "error_msg": str(error_msg),
        }
        self._append_jsonl(self.execution_log_path, record)
        return record

    def summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen estructurado y determinista de señales, aprobaciones/rechazos,
        desviación media de slippage (bps), gap medio EV predicho vs R realizado,
        errores registrados y eventos de datos degradados.
        """
        signal_records = [
            r for r in self._read_jsonl(self.signals_log_path) if r.get("event_type") == "SIGNAL"
        ]
        exec_file_records = self._read_jsonl(self.execution_log_path)
        execution_records = [
            r for r in exec_file_records if r.get("event_type") == "EXECUTION_COMPARISON"
        ]
        error_records = [
            r for r in exec_file_records if r.get("event_type") == "ERROR"
        ]

        total_signals = len(signal_records)
        approved_signals = sum(1 for r in signal_records if r.get("approved") is True)
        rejected_signals = total_signals - approved_signals

        approval_rate = (approved_signals / total_signals) if total_signals > 0 else 0.0
        rejection_rate = (rejected_signals / total_signals) if total_signals > 0 else 0.0

        rejection_reasons = dict(
            Counter(
                str(r.get("risk_reason", "UNKNOWN"))
                for r in signal_records
                if not r.get("approved", False)
            )
        )

        data_degraded_count = sum(
            1 for r in signal_records if str(r.get("data_status", "OK")).upper() != "OK"
        )

        total_executions = len(execution_records)
        if total_executions > 0:
            mean_slippage_bps = sum(
                float(r.get("entry_slippage_deviation_bps", 0.0)) for r in execution_records
            ) / total_executions
            mean_ev_gap_r = sum(
                float(r.get("ev_vs_realized_gap_r", 0.0)) for r in execution_records
            ) / total_executions
        else:
            mean_slippage_bps = 0.0
            mean_ev_gap_r = 0.0

        return {
            "total_signals": total_signals,
            "approved_signals": approved_signals,
            "rejected_signals": rejected_signals,
            "approval_rate": round(approval_rate, 6),
            "rejection_rate": round(rejection_rate, 6),
            "rejection_reasons": rejection_reasons,
            "total_executions": total_executions,
            "mean_entry_slippage_deviation_bps": round(mean_slippage_bps, 4),
            "mean_ev_vs_realized_gap_r": round(mean_ev_gap_r, 6),
            "error_count": len(error_records),
            "data_degraded_count": data_degraded_count,
        }
