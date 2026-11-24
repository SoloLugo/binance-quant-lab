"""
Label Engine (Fase 2) — Triple-Barrier ejecutable con 4 estados y cero look-ahead bias.

Protocolo de entrada por defecto (`entry_mode="next_open"`):
- La señal se genera al cierre de la vela `i`.
- La entrada simulada ocurre en `open[i + 1]`.
- Las barreras `tp_price` y `sl_price` se anclan a `entry = open[i + 1]` utilizando
  `atr_val = atr_pct[i] * entry` (el ATR relativo conocido al cierre de la vela `i`,
  sin incorporar información futura).
- La trayectoria se evalúa en las velas `j = 1 .. horizon_bars` (índices `i + 1 .. i + horizon_bars`).

Estados conservados en `outcome_state`:
- `"WIN"`        : Se alcanza TP antes que SL dentro del horizonte.
- `"LOSS"`       : Se alcanza SL antes que TP dentro del horizonte.
- `"TIMEOUT"`    : Ni TP ni SL se tocan dentro de `horizon_bars`; se preserva el retorno
                   económico real `(close[i + horizon_bars] - entry) / sl_dist` (invertido en SHORT).
- `"AMBIGUOUS"`  : TP y SL se tocan dentro de la misma vela `i + j`; no se inventa el orden
                   intrabar. Si `spec.exclude_ambiguous=True`, los targets derivados quedan en `NaN`.
- `"INCOMPLETE"` : Últimas `horizon_bars` velas (horizonte incompleto) o ATR/precio inválido.

Targets derivados documentados:
- `target_tp_before_sl`      : 1.0 si `WIN`, 0.0 si `LOSS` o `TIMEOUT`, `NaN` si `INCOMPLETE`
                               (y `NaN` si `AMBIGUOUS` con `exclude_ambiguous=True`).
- `target_tp_before_horizon` : 1.0 si `WIN` dentro del horizonte, 0.0 si `LOSS` o `TIMEOUT`,
                               `NaN` si `INCOMPLETE` (y `NaN` si `AMBIGUOUS` con `exclude_ambiguous=True`).
- `target_sl_before_tp`      : 1.0 si `LOSS`, 0.0 si `WIN` o `TIMEOUT`, `NaN` si `INCOMPLETE`
                               (y `NaN` si `AMBIGUOUS` con `exclude_ambiguous=True`).
"""
from typing import Any, Dict
import numpy as np
import pandas as pd

from contracts import LabelSpec


_OUTCOME_TO_EXIT_TYPE = {
    "WIN": "TP",
    "LOSS": "SL",
    "TIMEOUT": "TIME",
    "AMBIGUOUS": "AMBIGUOUS",
    "INCOMPLETE": "NONE",
}


class LabelEngine:
    """Motor de etiquetado Triple-Barrier ejecutable con trazabilidad completa."""

    VALID_ENTRY_MODES = ("next_open", "close")
    VALID_DIRECTIONS = ("LONG", "SHORT")

    def __init__(self, spec: LabelSpec = LabelSpec()):
        if spec.entry_mode not in self.VALID_ENTRY_MODES:
            raise ValueError(
                f"entry_mode inválido: '{spec.entry_mode}'. "
                f"Valores permitidos: {self.VALID_ENTRY_MODES}."
            )
        if spec.tp_mult <= 0 or spec.sl_mult <= 0:
            raise ValueError("tp_mult y sl_mult deben ser estrictamente positivos.")
        if spec.horizon_bars < 1:
            raise ValueError("horizon_bars debe ser >= 1.")
        self.spec = spec

    def compute_outcomes(
        self,
        df_btc: pd.DataFrame,
        atr_pct: pd.Series,
        direction: str = "LONG",
    ) -> pd.DataFrame:
        """
        Calcula resultados Triple-Barrier para cada vela `i` del DataFrame OHLC.

        Columnas devueltas:
        - `outcome_state`
        - `target_tp_before_sl`
        - `target_tp_before_horizon`
        - `target_sl_before_tp`
        - `realized_r_gross`
        - `exit_bars`
        - `exit_type`
        - `entry_price`
        - `tp_price`
        - `sl_price`
        """
        direction_upper = direction.upper().strip()
        if direction_upper not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"direction inválido: '{direction}'. Valores permitidos: {self.VALID_DIRECTIONS}."
            )

        required_cols = {"open", "high", "low", "close"}
        missing_cols = required_cols.difference(df_btc.columns)
        if missing_cols:
            raise ValueError(f"df_btc no contiene columnas requeridas: {sorted(missing_cols)}")

        opens = df_btc["open"].to_numpy(dtype=float)
        highs = df_btc["high"].to_numpy(dtype=float)
        lows = df_btc["low"].to_numpy(dtype=float)
        closes = df_btc["close"].to_numpy(dtype=float)
        atr_pct_aligned = atr_pct.reindex(df_btc.index).to_numpy(dtype=float)

        n = len(df_btc)
        horizon = self.spec.horizon_bars
        tp_mult = self.spec.tp_mult
        sl_mult = self.spec.sl_mult
        rr_ratio = tp_mult / sl_mult
        use_next_open = self.spec.entry_mode == "next_open"
        exclude_ambiguous = self.spec.exclude_ambiguous

        outcome_state = np.full(n, "INCOMPLETE", dtype=object)
        target_tp_before_sl = np.full(n, np.nan, dtype=float)
        target_tp_before_horizon = np.full(n, np.nan, dtype=float)
        target_sl_before_tp = np.full(n, np.nan, dtype=float)
        realized_r_gross = np.full(n, np.nan, dtype=float)
        exit_bars = np.full(n, horizon, dtype=int)
        exit_type = np.full(n, "NONE", dtype=object)
        entry_price_arr = np.full(n, np.nan, dtype=float)
        tp_price_arr = np.full(n, np.nan, dtype=float)
        sl_price_arr = np.full(n, np.nan, dtype=float)

        last_valid_i = max(0, n - horizon)
        for i in range(last_valid_i):
            atr_rel = atr_pct_aligned[i]
            if np.isnan(atr_rel) or atr_rel <= 0.0:
                continue

            entry = opens[i + 1] if use_next_open else closes[i]
            if np.isnan(entry) or entry <= 0.0:
                continue

            # ATR en unidades de precio usando atr_pct conocido al cierre de la vela i
            atr_val = atr_rel * entry
            sl_dist = sl_mult * atr_val
            tp_dist = tp_mult * atr_val
            if sl_dist <= 0.0 or tp_dist <= 0.0:
                continue

            if direction_upper == "LONG":
                tp_price = entry + tp_dist
                sl_price = entry - sl_dist
            else:
                tp_price = entry - tp_dist
                sl_price = entry + sl_dist

            entry_price_arr[i] = entry
            tp_price_arr[i] = tp_price
            sl_price_arr[i] = sl_price

            resolved = False
            for j in range(1, horizon + 1):
                bar_high = highs[i + j]
                bar_low = lows[i + j]

                if direction_upper == "LONG":
                    hit_tp = bar_high >= tp_price
                    hit_sl = bar_low <= sl_price
                else:
                    hit_tp = bar_low <= tp_price
                    hit_sl = bar_high >= sl_price

                if hit_tp and hit_sl:
                    outcome_state[i] = "AMBIGUOUS"
                    exit_type[i] = _OUTCOME_TO_EXIT_TYPE["AMBIGUOUS"]
                    exit_bars[i] = j
                    if exclude_ambiguous:
                        target_tp_before_sl[i] = np.nan
                        target_tp_before_horizon[i] = np.nan
                        target_sl_before_tp[i] = np.nan
                        realized_r_gross[i] = np.nan
                    else:
                        target_tp_before_sl[i] = 0.0
                        target_tp_before_horizon[i] = 0.0
                        target_sl_before_tp[i] = 1.0
                        realized_r_gross[i] = -1.0
                    resolved = True
                    break

                if hit_sl:
                    outcome_state[i] = "LOSS"
                    exit_type[i] = _OUTCOME_TO_EXIT_TYPE["LOSS"]
                    exit_bars[i] = j
                    target_tp_before_sl[i] = 0.0
                    target_tp_before_horizon[i] = 0.0
                    target_sl_before_tp[i] = 1.0
                    realized_r_gross[i] = -1.0
                    resolved = True
                    break

                if hit_tp:
                    outcome_state[i] = "WIN"
                    exit_type[i] = _OUTCOME_TO_EXIT_TYPE["WIN"]
                    exit_bars[i] = j
                    target_tp_before_sl[i] = 1.0
                    target_tp_before_horizon[i] = 1.0
                    target_sl_before_tp[i] = 0.0
                    realized_r_gross[i] = rr_ratio
                    resolved = True
                    break

            if not resolved:
                outcome_state[i] = "TIMEOUT"
                exit_type[i] = _OUTCOME_TO_EXIT_TYPE["TIMEOUT"]
                exit_bars[i] = horizon
                target_tp_before_sl[i] = 0.0
                target_tp_before_horizon[i] = 0.0
                target_sl_before_tp[i] = 0.0
                final_close = closes[i + horizon]
                if direction_upper == "LONG":
                    realized_r_gross[i] = float((final_close - entry) / sl_dist)
                else:
                    realized_r_gross[i] = float((entry - final_close) / sl_dist)

        return pd.DataFrame(
            {
                "outcome_state": outcome_state,
                "target_tp_before_sl": target_tp_before_sl,
                "target_tp_before_horizon": target_tp_before_horizon,
                "target_sl_before_tp": target_sl_before_tp,
                "realized_r_gross": realized_r_gross,
                "exit_bars": exit_bars,
                "exit_type": exit_type,
                "entry_price": entry_price_arr,
                "tp_price": tp_price_arr,
                "sl_price": sl_price_arr,
            },
            index=df_btc.index,
        )

    @staticmethod
    def summarize_outcomes(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Resume los conteos, porcentajes, frecuencia de AMBIGUOUS y análisis de sensibilidad
        (tasa de WIN si todos los AMBIGUOUS fueran WIN vs si todos fueran LOSS).
        """
        if "outcome_state" not in outcomes_df.columns:
            raise ValueError("outcomes_df debe contener la columna 'outcome_state'.")

        states = outcomes_df["outcome_state"]
        total_rows = int(len(outcomes_df))

        counts = {
            "WIN": int((states == "WIN").sum()),
            "LOSS": int((states == "LOSS").sum()),
            "TIMEOUT": int((states == "TIMEOUT").sum()),
            "AMBIGUOUS": int((states == "AMBIGUOUS").sum()),
            "INCOMPLETE": int((states == "INCOMPLETE").sum()),
        }
        evaluated_rows = counts["WIN"] + counts["LOSS"] + counts["TIMEOUT"] + counts["AMBIGUOUS"]

        if evaluated_rows > 0:
            percentages = {
                k: float(counts[k] / evaluated_rows * 100.0)
                for k in ("WIN", "LOSS", "TIMEOUT", "AMBIGUOUS")
            }
            ambiguous_rate = float(counts["AMBIGUOUS"] / evaluated_rows)
            win_if_amb_win = float((counts["WIN"] + counts["AMBIGUOUS"]) / evaluated_rows)
            win_if_amb_loss = float(counts["WIN"] / evaluated_rows)
        else:
            percentages = {k: 0.0 for k in ("WIN", "LOSS", "TIMEOUT", "AMBIGUOUS")}
            ambiguous_rate = 0.0
            win_if_amb_win = 0.0
            win_if_amb_loss = 0.0

        non_ambiguous = evaluated_rows - counts["AMBIGUOUS"]
        win_rate_excl_amb = float(counts["WIN"] / non_ambiguous) if non_ambiguous > 0 else 0.0

        return {
            "total_rows": total_rows,
            "evaluated_rows": evaluated_rows,
            "counts": counts,
            "percentages": percentages,
            "ambiguous_count": counts["AMBIGUOUS"],
            "ambiguous_rate": ambiguous_rate,
            "win_rate_excluding_ambiguous": win_rate_excl_amb,
            "sensitivity": {
                "win_rate_if_ambiguous_win": win_if_amb_win,
                "win_rate_if_ambiguous_loss": win_if_amb_loss,
                "win_rate_spread": float(win_if_amb_win - win_if_amb_loss),
            },
        }
