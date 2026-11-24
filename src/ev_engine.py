"""
EV Engine y Cost Model (Fase 7) — Evaluación rigurosa de Valor Esperado Neto en unidades R.

Características principales:
- Desglose explícito de costes de ejecución en unidades de 1R:
  - Fees roundtrip (`fee_in_r`)
  - Spread roundtrip (`spread_in_r`)
  - Slippage roundtrip (`slippage_in_r`)
  - Funding esperado con signo direccional (`funding_in_r`: LONG paga si funding > 0, SHORT cobra)
- Modelado riguroso de `TIMEOUT`: no se penaliza artificialmente como `-1R`; se estima
  `P(TIMEOUT | not WIN)` y `E[realized_r_gross | TIMEOUT]` exclusivamente sobre train/calib
  mediante `estimate_timeout_economics`, o se aceptan probabilidades explícitas `p_sl` y `p_timeout`.
- Reglas de rechazo auditables (`ACCEPTED`, `EV_BELOW_THRESHOLD`, `INVALID_PROBA`,
  `INVALID_ATR`, `NON_POSITIVE_NET_REWARD`).
- Reporte de sensibilidad bajo múltiples escenarios de `CostModel`.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from contracts import CostModel


ArrayLikeOrScalar = Union[pd.Series, np.ndarray, Sequence[float], float, int]


@dataclass(frozen=True)
class TimeoutEconomics:
    """
    Estadísticos económicos de eventos TIMEOUT estimados EXCLUSIVAMENTE sobre train/calib.

    - `timeout_share_given_not_tp`: P(TIMEOUT | not WIN) = n_timeout / (n_loss + n_timeout),
      excluyendo estados `AMBIGUOUS` e `INCOMPLETE`.
    - `mean_timeout_r`: E[realized_r_gross | TIMEOUT] en unidades R brutas.
    - `n_timeout_samples`: número de muestras válidas de `TIMEOUT` empleadas en la estimación.
    """

    timeout_share_given_not_tp: float = 0.0
    mean_timeout_r: float = 0.0
    n_timeout_samples: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.timeout_share_given_not_tp) or not (
            0.0 <= self.timeout_share_given_not_tp <= 1.0
        ):
            raise ValueError(
                "timeout_share_given_not_tp debe ser finito y estar en [0.0, 1.0], "
                f"recibido: {self.timeout_share_given_not_tp}."
            )
        if not np.isfinite(self.mean_timeout_r):
            raise ValueError(
                f"mean_timeout_r debe ser finito, recibido: {self.mean_timeout_r}."
            )
        if self.n_timeout_samples < 0:
            raise ValueError(
                f"n_timeout_samples debe ser >= 0, recibido: {self.n_timeout_samples}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_timeout_economics(outcomes_df: pd.DataFrame) -> TimeoutEconomics:
    """
    Estima la economía empírica de los eventos `TIMEOUT` a partir de un DataFrame de outcomes
    producido por `LabelEngine.compute_outcomes`.

    IMPORTANTE (Cero Look-Ahead Bias):
    Esta función DEBE invocarse EXCLUSIVAMENTE sobre particiones de entrenamiento/calibración
    (train/calib), nunca sobre validación ni sobre el test final sellado.

    Los estados `AMBIGUOUS` e `INCOMPLETE` se ignoran por completo.
    """
    required_cols = {"outcome_state", "realized_r_gross"}
    missing_cols = required_cols.difference(outcomes_df.columns)
    if missing_cols:
        raise ValueError(
            f"outcomes_df no contiene columnas requeridas: {sorted(missing_cols)}"
        )

    if len(outcomes_df) == 0:
        return TimeoutEconomics(
            timeout_share_given_not_tp=0.0,
            mean_timeout_r=0.0,
            n_timeout_samples=0,
        )

    states = outcomes_df["outcome_state"].astype(str)
    loss_mask = states == "LOSS"
    timeout_mask = states == "TIMEOUT"

    n_loss = int(loss_mask.sum())
    n_timeout = int(timeout_mask.sum())
    n_not_tp = n_loss + n_timeout

    if n_not_tp == 0:
        return TimeoutEconomics(
            timeout_share_given_not_tp=0.0,
            mean_timeout_r=0.0,
            n_timeout_samples=0,
        )

    timeout_share = float(n_timeout / n_not_tp)

    timeout_returns = pd.to_numeric(
        outcomes_df.loc[timeout_mask, "realized_r_gross"], errors="coerce"
    ).to_numpy(dtype=float)
    valid_timeout_returns = timeout_returns[np.isfinite(timeout_returns)]
    n_valid_timeout = int(len(valid_timeout_returns))
    mean_timeout_r = (
        float(np.mean(valid_timeout_returns)) if n_valid_timeout > 0 else 0.0
    )

    return TimeoutEconomics(
        timeout_share_given_not_tp=timeout_share,
        mean_timeout_r=mean_timeout_r,
        n_timeout_samples=n_valid_timeout,
    )


def _normalize_array_input(
    value: ArrayLikeOrScalar,
    target_len: int,
    name: str,
) -> np.ndarray:
    """Convierte escalar, Series o array a np.ndarray 1D de float64 con longitud `target_len`."""
    if np.isscalar(value):
        return np.full(target_len, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(target_len, float(arr.item()), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' debe ser escalar o unidimensional, recibido ndim={arr.ndim}.")
    if len(arr) != target_len:
        raise ValueError(
            f"Longitud inconsistente para '{name}': esperado {target_len}, recibido {len(arr)}."
        )
    return arr


def _infer_length_and_index(
    p_calibrated: ArrayLikeOrScalar,
    atr_pct: Optional[ArrayLikeOrScalar],
) -> Tuple[int, pd.Index]:
    """Infiere longitud común e índice preservando `pd.Series.index` cuando esté disponible."""
    if isinstance(p_calibrated, pd.Series):
        return len(p_calibrated), p_calibrated.index
    if isinstance(atr_pct, pd.Series):
        if np.isscalar(p_calibrated):
            return len(atr_pct), atr_pct.index
        p_arr = np.asarray(p_calibrated)
        if len(p_arr) != len(atr_pct):
            raise ValueError(
                f"Longitud inconsistente entre p_calibrated ({len(p_arr)}) y atr_pct ({len(atr_pct)})."
            )
        return len(atr_pct), atr_pct.index

    if not np.isscalar(p_calibrated):
        n = len(np.asarray(p_calibrated))
        return n, pd.RangeIndex(n)
    if atr_pct is not None and not np.isscalar(atr_pct):
        n = len(np.asarray(atr_pct))
        return n, pd.RangeIndex(n)
    return 1, pd.RangeIndex(1)


class EVEngine:
    """
    Motor de Valor Esperado Neto en unidades R con desglose completo de costes
    y tratamiento explícito de `TIMEOUT`.
    """

    VALID_DIRECTIONS = ("LONG", "SHORT")
    MIN_R_DIST_PCT = 1e-6
    PROBA_SUM_TOL = 1e-6

    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "p_raw",
        "p_calibrated",
        "p_sl",
        "p_timeout",
        "expected_timeout_r",
        "gross_win_r",
        "gross_loss_r",
        "fee_in_r",
        "spread_in_r",
        "slippage_in_r",
        "funding_in_r",
        "total_cost_in_r",
        "ev_gross_r",
        "ev_net_r",
        "reward_risk_ratio_net",
        "accepted",
        "rejection_reason",
    )

    def __init__(
        self,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        cost_model: CostModel = CostModel(),
        ev_min_threshold: float = 0.12,
        direction: str = "LONG",
    ) -> None:
        if tp_mult <= 0.0 or not np.isfinite(tp_mult):
            raise ValueError(f"tp_mult debe ser finito y > 0, recibido: {tp_mult}.")
        if sl_mult <= 0.0 or not np.isfinite(sl_mult):
            raise ValueError(f"sl_mult debe ser finito y > 0, recibido: {sl_mult}.")
        if not np.isfinite(ev_min_threshold):
            raise ValueError(
                f"ev_min_threshold debe ser finito, recibido: {ev_min_threshold}."
            )

        direction_upper = (direction or "").upper().strip()
        if direction_upper not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"direction inválido: '{direction}'. Valores permitidos: {self.VALID_DIRECTIONS}."
            )

        self.tp_mult = float(tp_mult)
        self.sl_mult = float(sl_mult)
        self.cost_model = cost_model
        self.ev_min_threshold = float(ev_min_threshold)
        self.direction = direction_upper
        self.direction_sign = 1.0 if self.direction == "LONG" else -1.0
        self.gross_win_r = float(self.tp_mult / self.sl_mult)
        self.gross_loss_r = -1.0

    def evaluate(
        self,
        p_calibrated: ArrayLikeOrScalar,
        atr_pct: Optional[ArrayLikeOrScalar] = None,
        p_raw: Optional[ArrayLikeOrScalar] = None,
        p_sl: Optional[ArrayLikeOrScalar] = None,
        p_timeout: Optional[ArrayLikeOrScalar] = None,
        timeout_economics: Optional[TimeoutEconomics] = None,
    ) -> pd.DataFrame:
        """
        Evalúa el Valor Esperado Bruto y Neto (en unidades R), el desglose de costes
        y la decisión de aceptación/rechazo para una o varias señales.
        """
        n, index = _infer_length_and_index(p_calibrated, atr_pct)

        if n == 0:
            empty_df = pd.DataFrame(columns=list(self.OUTPUT_COLUMNS), index=index)
            empty_df["accepted"] = empty_df["accepted"].astype(bool)
            return empty_df

        p_cal_arr = _normalize_array_input(p_calibrated, n, "p_calibrated")
        p_raw_arr = (
            p_cal_arr.copy()
            if p_raw is None
            else _normalize_array_input(p_raw, n, "p_raw")
        )
        atr_arr = (
            np.full(n, np.nan, dtype=float)
            if atr_pct is None
            else _normalize_array_input(atr_pct, n, "atr_pct")
        )

        te = timeout_economics if timeout_economics is not None else TimeoutEconomics()
        expected_timeout_r = np.full(n, te.mean_timeout_r, dtype=float)

        # Probabilidades de SL y TIMEOUT
        explicit_probs = (p_sl is not None) or (p_timeout is not None)
        if p_sl is not None and p_timeout is not None:
            p_sl_arr = _normalize_array_input(p_sl, n, "p_sl")
            p_to_arr = _normalize_array_input(p_timeout, n, "p_timeout")
        elif p_sl is not None:
            p_sl_arr = _normalize_array_input(p_sl, n, "p_sl")
            p_to_arr = 1.0 - p_cal_arr - p_sl_arr
        elif p_timeout is not None:
            p_to_arr = _normalize_array_input(p_timeout, n, "p_timeout")
            p_sl_arr = 1.0 - p_cal_arr - p_to_arr
        else:
            p_to_arr = (1.0 - p_cal_arr) * te.timeout_share_given_not_tp
            p_sl_arr = (1.0 - p_cal_arr) * (1.0 - te.timeout_share_given_not_tp)

        # Validación de probabilidades
        valid_proba = (
            np.isfinite(p_cal_arr)
            & (p_cal_arr >= 0.0)
            & (p_cal_arr <= 1.0)
            & np.isfinite(p_sl_arr)
            & (p_sl_arr >= 0.0)
            & (p_sl_arr <= 1.0)
            & np.isfinite(p_to_arr)
            & (p_to_arr >= 0.0)
            & (p_to_arr <= 1.0)
        )
        if explicit_probs:
            prob_sum = p_cal_arr + p_sl_arr + p_to_arr
            valid_proba = valid_proba & (np.abs(prob_sum - 1.0) <= self.PROBA_SUM_TOL)

        # Validación de ATR
        valid_atr = np.isfinite(atr_arr) & (atr_arr > 0.0)

        # Desglose de costes en unidades 1R
        fee_in_r = np.full(n, np.nan, dtype=float)
        spread_in_r = np.full(n, np.nan, dtype=float)
        slippage_in_r = np.full(n, np.nan, dtype=float)
        funding_in_r = np.full(n, np.nan, dtype=float)
        total_cost_in_r = np.full(n, np.nan, dtype=float)

        if np.any(valid_atr):
            r_dist_pct = np.maximum(
                atr_arr[valid_atr] * self.sl_mult, self.MIN_R_DIST_PCT
            )
            fee_in_r[valid_atr] = self.cost_model.roundtrip_fee_rate / r_dist_pct
            spread_in_r[valid_atr] = self.cost_model.roundtrip_spread_rate / r_dist_pct
            slippage_in_r[valid_atr] = (
                self.cost_model.roundtrip_slippage_rate / r_dist_pct
            )
            funding_rate_horizon = (
                self.direction_sign
                * self.cost_model.expected_funding_rate_8h
                * (self.cost_model.horizon_hours / 8.0)
            )
            funding_in_r[valid_atr] = funding_rate_horizon / r_dist_pct
            total_cost_in_r[valid_atr] = (
                fee_in_r[valid_atr]
                + spread_in_r[valid_atr]
                + slippage_in_r[valid_atr]
                + funding_in_r[valid_atr]
            )

        gross_win_r = np.full(n, self.gross_win_r, dtype=float)
        gross_loss_r = np.full(n, self.gross_loss_r, dtype=float)

        ev_gross_r = np.full(n, np.nan, dtype=float)
        if np.any(valid_proba):
            ev_gross_r[valid_proba] = (
                p_cal_arr[valid_proba] * gross_win_r[valid_proba]
                + p_sl_arr[valid_proba] * gross_loss_r[valid_proba]
                + p_to_arr[valid_proba] * expected_timeout_r[valid_proba]
            )

        valid_both = valid_proba & valid_atr
        ev_net_r = np.full(n, np.nan, dtype=float)
        if np.any(valid_both):
            ev_net_r[valid_both] = ev_gross_r[valid_both] - total_cost_in_r[valid_both]

        reward_risk_ratio_net = np.full(n, np.nan, dtype=float)
        net_reward_r = np.full(n, np.nan, dtype=float)
        if np.any(valid_atr):
            net_reward_r[valid_atr] = gross_win_r[valid_atr] - total_cost_in_r[valid_atr]
            net_risk_r = 1.0 + total_cost_in_r[valid_atr]
            valid_denom = net_risk_r > 0.0
            rrr_vals = np.full(int(np.sum(valid_atr)), np.nan, dtype=float)
            rrr_vals[valid_denom] = net_reward_r[valid_atr][valid_denom] / net_risk_r[valid_denom]
            reward_risk_ratio_net[valid_atr] = rrr_vals

        # Reglas de aceptación y motivos de rechazo
        accepted = np.zeros(n, dtype=bool)
        rejection_reason = np.full(n, "EV_BELOW_THRESHOLD", dtype=object)

        for i in range(n):
            if not valid_proba[i]:
                rejection_reason[i] = "INVALID_PROBA"
            elif not valid_atr[i]:
                rejection_reason[i] = "INVALID_ATR"
            elif (
                not np.isfinite(net_reward_r[i])
                or net_reward_r[i] <= 0.0
                or not np.isfinite(reward_risk_ratio_net[i])
                or reward_risk_ratio_net[i] <= 0.0
            ):
                rejection_reason[i] = "NON_POSITIVE_NET_REWARD"
            elif not np.isfinite(ev_net_r[i]) or ev_net_r[i] < self.ev_min_threshold:
                rejection_reason[i] = "EV_BELOW_THRESHOLD"
            else:
                accepted[i] = True
                rejection_reason[i] = "ACCEPTED"

        return pd.DataFrame(
            {
                "p_raw": p_raw_arr,
                "p_calibrated": p_cal_arr,
                "p_sl": p_sl_arr,
                "p_timeout": p_to_arr,
                "expected_timeout_r": expected_timeout_r,
                "gross_win_r": gross_win_r,
                "gross_loss_r": gross_loss_r,
                "fee_in_r": fee_in_r,
                "spread_in_r": spread_in_r,
                "slippage_in_r": slippage_in_r,
                "funding_in_r": funding_in_r,
                "total_cost_in_r": total_cost_in_r,
                "ev_gross_r": ev_gross_r,
                "ev_net_r": ev_net_r,
                "reward_risk_ratio_net": reward_risk_ratio_net,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
            },
            index=index,
        )

    def sensitivity_report(
        self,
        p_calibrated: ArrayLikeOrScalar,
        atr_pct: ArrayLikeOrScalar,
        cost_scenarios: Dict[str, CostModel],
        timeout_economics: Optional[TimeoutEconomics] = None,
    ) -> pd.DataFrame:
        """
        Evalúa la sensibilidad del EV neto y la tasa de aceptación frente a múltiples
        escenarios de `CostModel`.
        """
        report_cols = [
            "scenario",
            "n_total",
            "n_accepted",
            "acceptance_rate",
            "mean_fee_in_r",
            "mean_spread_in_r",
            "mean_slippage_in_r",
            "mean_funding_in_r",
            "mean_total_cost_in_r",
            "mean_ev_gross_r",
            "mean_ev_net_r",
            "median_ev_net_r",
            "mean_ev_net_r_accepted",
            "mean_reward_risk_ratio_net",
        ]
        if not cost_scenarios:
            return pd.DataFrame(columns=report_cols)

        rows: List[Dict[str, Any]] = []
        for scenario_name, scenario_cost_model in cost_scenarios.items():
            scenario_engine = EVEngine(
                tp_mult=self.tp_mult,
                sl_mult=self.sl_mult,
                cost_model=scenario_cost_model,
                ev_min_threshold=self.ev_min_threshold,
                direction=self.direction,
            )
            eval_df = scenario_engine.evaluate(
                p_calibrated=p_calibrated,
                atr_pct=atr_pct,
                timeout_economics=timeout_economics,
            )
            n_total = int(len(eval_df))
            n_accepted = int(eval_df["accepted"].sum())
            accepted_ev = eval_df.loc[eval_df["accepted"], "ev_net_r"]

            rows.append(
                {
                    "scenario": scenario_name,
                    "n_total": n_total,
                    "n_accepted": n_accepted,
                    "acceptance_rate": float(n_accepted / n_total) if n_total > 0 else 0.0,
                    "mean_fee_in_r": float(eval_df["fee_in_r"].mean()) if n_total > 0 else np.nan,
                    "mean_spread_in_r": float(eval_df["spread_in_r"].mean()) if n_total > 0 else np.nan,
                    "mean_slippage_in_r": float(eval_df["slippage_in_r"].mean()) if n_total > 0 else np.nan,
                    "mean_funding_in_r": float(eval_df["funding_in_r"].mean()) if n_total > 0 else np.nan,
                    "mean_total_cost_in_r": float(eval_df["total_cost_in_r"].mean()) if n_total > 0 else np.nan,
                    "mean_ev_gross_r": float(eval_df["ev_gross_r"].mean()) if n_total > 0 else np.nan,
                    "mean_ev_net_r": float(eval_df["ev_net_r"].mean()) if n_total > 0 else np.nan,
                    "median_ev_net_r": float(eval_df["ev_net_r"].median()) if n_total > 0 else np.nan,
                    "mean_ev_net_r_accepted": float(accepted_ev.mean()) if n_accepted > 0 else np.nan,
                    "mean_reward_risk_ratio_net": float(eval_df["reward_risk_ratio_net"].mean()) if n_total > 0 else np.nan,
                }
            )

        return pd.DataFrame(rows, columns=report_cols)
