"""
Risk Engine independiente del modelo predictivo para Binance Quant Lab (Fase 12).
Gestiona sizing con Kelly conservador penalizado por incertidumbre, límites de
apalancamiento, exposición total, concentración por símbolo, correlación,
pérdida diaria, drawdown máximo global y kill switch operativo.
"""
from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional

from contracts import RiskDecision


@dataclass(frozen=True)
class RiskLimits:
    """Parámetros inmutables de riesgo independientes del modelo predictivo."""
    max_risk_per_trade: float = 0.03
    kelly_fraction: float = 0.20
    probability_uncertainty_margin: float = 0.02
    max_leverage: float = 5.0
    max_total_exposure_notional_mult: float = 5.0
    max_symbol_concentration_pct: float = 1.0
    max_open_positions: int = 1
    max_correlation_threshold: float = 0.70
    max_daily_loss_pct: float = 0.06
    max_drawdown_pct: float = 0.15
    min_notional_usdt: float = 100.0
    min_qty_btc: float = 0.002

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskEngine:
    """
    Motor de riesgo independiente del modelo (Fase 12).
    Tiene autoridad absoluta para rechazar cualquier operación —incluso si el modelo
    emite una probabilidad calibrada de 0.95— cuando se viola cualquier regla de riesgo.
    """

    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits
        self._kill_switch_active: bool = False
        self._kill_switch_reason: Optional[str] = None

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def kill_switch_reason(self) -> Optional[str]:
        return self._kill_switch_reason

    def activate_kill_switch(self, reason: str) -> None:
        """Activa el kill switch bloqueando inmediatamente cualquier nueva orden."""
        cleaned = str(reason).strip() if reason is not None else ""
        self._kill_switch_active = True
        self._kill_switch_reason = cleaned or "MANUAL_KILL_SWITCH"

    def reset_kill_switch(self) -> None:
        """Desactiva el kill switch y limpia el motivo registrado."""
        self._kill_switch_active = False
        self._kill_switch_reason = None

    def compute_conservative_kelly(
        self,
        p_calibrated: float,
        b_net_odds: float,
        l_net_odds: float = 1.0,
    ) -> float:
        """
        Calcula la fracción de riesgo Kelly conservadora penalizando `p_calibrated`
        por `probability_uncertainty_margin` antes de evaluar la esperanza matemática.
        """
        if not (
            math.isfinite(p_calibrated)
            and math.isfinite(b_net_odds)
            and math.isfinite(l_net_odds)
        ):
            return 0.0

        if b_net_odds <= 0.0 or l_net_odds <= 0.0:
            return 0.0

        p_clamped = min(max(float(p_calibrated), 0.0), 1.0)
        p_cons = max(0.0, p_clamped - self.limits.probability_uncertainty_margin)
        ev_cons = p_cons * b_net_odds - (1.0 - p_cons) * l_net_odds
        if ev_cons <= 0.0:
            return 0.0

        kelly_full = ev_cons / (b_net_odds * l_net_odds)
        kelly_scaled = kelly_full * self.limits.kelly_fraction
        return float(min(kelly_scaled, self.limits.max_risk_per_trade))

    @staticmethod
    def _extract_position_notional(position: Dict[str, Any]) -> float:
        if "notional_usdt" in position and position["notional_usdt"] is not None:
            return abs(float(position["notional_usdt"]))
        qty = abs(float(position.get("qty_btc", 0.0)))
        price = float(position.get("mark_price", position.get("entry_price", 0.0)))
        return qty * max(price, 0.0)

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            qty_btc=0.0,
            risk_fraction=0.0,
            reason=reason,
            effective_leverage=0.0,
        )

    def evaluate_order(
        self,
        symbol: str,
        side: str,
        equity_usdt: float,
        available_balance_usdt: float,
        peak_equity_usdt: float,
        daily_start_equity_usdt: float,
        entry_price: float,
        sl_distance_usdt: float,
        ev_net_r: float,
        p_calibrated: float,
        b_net_odds: float,
        l_net_odds: float = 1.0,
        open_positions: Optional[List[Dict[str, Any]]] = None,
        max_corr_with_open: float = 0.0,
    ) -> RiskDecision:
        """
        Evalúa de forma independiente una orden propuesta y emite una RiskDecision.
        """
        # 1. Kill switch activo
        if self._kill_switch_active:
            return self._reject("KILL_SWITCH_ACTIVE")

        # 2. Validación estricta de entradas
        numeric_inputs = (
            equity_usdt,
            available_balance_usdt,
            peak_equity_usdt,
            daily_start_equity_usdt,
            entry_price,
            sl_distance_usdt,
            ev_net_r,
            p_calibrated,
            b_net_odds,
            l_net_odds,
            max_corr_with_open,
        )
        if not all(math.isfinite(float(x)) for x in numeric_inputs):
            return self._reject("INVALID_INPUTS")

        if (
            equity_usdt <= 0.0
            or available_balance_usdt <= 0.0
            or peak_equity_usdt <= 0.0
            or daily_start_equity_usdt <= 0.0
            or entry_price <= 0.0
            or sl_distance_usdt <= 0.0
            or str(side).upper() not in ("LONG", "SHORT")
        ):
            return self._reject("INVALID_INPUTS")

        # 3. Max Drawdown global
        drawdown_pct = (peak_equity_usdt - equity_usdt) / peak_equity_usdt
        if drawdown_pct >= self.limits.max_drawdown_pct:
            return self._reject("MAX_DRAWDOWN_BREACHED")

        # 4. Pérdida diaria máxima
        daily_loss_pct = (daily_start_equity_usdt - equity_usdt) / daily_start_equity_usdt
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            return self._reject("DAILY_LOSS_LIMIT_BREACHED")

        # 5. Límite de posiciones abiertas simultáneas
        positions = open_positions if open_positions is not None else []
        if len(positions) >= self.limits.max_open_positions:
            return self._reject("MAX_OPEN_POSITIONS_REACHED")

        # 6. Límite de correlación con posiciones abiertas
        if len(positions) > 0 and abs(float(max_corr_with_open)) >= self.limits.max_correlation_threshold:
            return self._reject("CORRELATION_LIMIT_BREACHED")

        # 7. EV neto no positivo
        if ev_net_r <= 0.0:
            return self._reject("NON_POSITIVE_EV")

        # 8. Kelly conservador penalizado por incertidumbre
        conservative_risk_fraction = self.compute_conservative_kelly(
            p_calibrated=p_calibrated,
            b_net_odds=b_net_odds,
            l_net_odds=l_net_odds,
        )
        if conservative_risk_fraction <= 0.0:
            return self._reject("UNCERTAINTY_ADJUSTED_KELLY_ZERO")

        # 9. Sizing respetando riesgo, apalancamiento, exposición total y concentración por símbolo
        capital_base = min(float(equity_usdt), float(available_balance_usdt))
        risk_budget_usdt = capital_base * conservative_risk_fraction
        qty_by_risk = risk_budget_usdt / float(sl_distance_usdt)

        max_notional_by_leverage = capital_base * self.limits.max_leverage

        total_exposure_limit = float(equity_usdt) * self.limits.max_total_exposure_notional_mult
        current_total_notional = sum(self._extract_position_notional(p) for p in positions)
        remaining_total_notional = max(0.0, total_exposure_limit - current_total_notional)

        symbol_upper = str(symbol).upper()
        symbol_exposure_limit = total_exposure_limit * self.limits.max_symbol_concentration_pct
        current_symbol_notional = sum(
            self._extract_position_notional(p)
            for p in positions
            if str(p.get("symbol", "")).upper() == symbol_upper
        )
        remaining_symbol_notional = max(0.0, symbol_exposure_limit - current_symbol_notional)

        max_allowed_notional = min(
            max_notional_by_leverage,
            remaining_total_notional,
            remaining_symbol_notional,
        )
        qty_by_notional = max_allowed_notional / float(entry_price)

        raw_qty = min(qty_by_risk, qty_by_notional)
        qty_btc = math.floor(raw_qty * 1000.0 + 1e-12) / 1000.0

        min_qty_for_notional = math.ceil((self.limits.min_notional_usdt / float(entry_price)) * 1000.0 - 1e-12) / 1000.0
        min_required_qty = max(self.limits.min_qty_btc, min_qty_for_notional)

        if qty_btc < min_required_qty or (qty_btc * float(entry_price)) < self.limits.min_notional_usdt:
            # Verificamos si se puede ajustar al lote mínimo sin exceder riesgo máximo ni límites nocionales
            min_risk_usdt = min_required_qty * float(sl_distance_usdt)
            min_notional_needed = min_required_qty * float(entry_price)
            max_risk_usdt = capital_base * self.limits.max_risk_per_trade
            if (
                min_risk_usdt <= max_risk_usdt + 1e-12
                and min_notional_needed <= max_allowed_notional + 1e-12
            ):
                qty_btc = round(min_required_qty, 3)
            else:
                return self._reject("SIZE_BELOW_SAFE_MINIMUM")

        # 10. Aprobada
        actual_notional = qty_btc * float(entry_price)
        actual_risk_fraction = (qty_btc * float(sl_distance_usdt)) / capital_base
        effective_leverage = actual_notional / float(equity_usdt)

        return RiskDecision(
            approved=True,
            qty_btc=round(qty_btc, 3),
            risk_fraction=round(actual_risk_fraction, 6),
            reason="APPROVED",
            effective_leverage=round(effective_leverage, 4),
        )
