"""
Backtest Engine (Fase 8) — Motor de Backtesting de Futuros Perpetuos USD-M
con contabilidad exacta, microestructura realista y estado aislado por corrida.

Convenciones compatibles con LabelEngine (Fase 2) y Paper Trading (simulator.py):
- Señal al cierre de la vela `i` -> entrada ejecutable en `open[i + 1]` con
  half-spread + slippage adverso.
- Tamaño de posición por riesgo `risk_usdt = balance * risk_frac`, acotado por
  `max_leverage` y escalado por `partial_fill_ratio`.
- Comisión taker de apertura y cierre descontada explícitamente.
- Liquidación de Funding Rate cada 8h (00:00, 08:00, 16:00 UTC) con signo exacto
  en LONG y SHORT.
- Manejo explícito de gaps de precio en SL/TP, ambigüedad intrabar y cierre por TIMEOUT.
- Una posición activa a la vez y sin re-entrada en la misma barra de cierre
  (`next_free_idx = exit_idx + 1`).
- Sin lectura ni escritura de archivos de estado de Paper Trading.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math
import numpy as np
import pandas as pd

from contracts import CostModel


VALID_AMBIGUOUS_POLICIES = ("conservative_sl", "skip")

TRADE_COLUMNS = [
    "trade_id",
    "side",
    "signal_idx",
    "entry_idx",
    "exit_idx",
    "signal_time",
    "entry_time",
    "exit_time",
    "duration_bars",
    "qty",
    "entry_price",
    "exit_price",
    "sl_price",
    "tp_price",
    "notional_usdt",
    "risk_1r_usdt",
    "gross_pnl_usdt",
    "entry_fee_usdt",
    "exit_fee_usdt",
    "total_fees_usdt",
    "funding_paid_usdt",
    "slippage_usdt",
    "net_pnl_usdt",
    "realized_r_gross",
    "realized_r_net",
    "exit_reason",
    "intrabar_ambiguous",
    "p_calibrated",
    "ev_net_r",
    "balance_after_usdt",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuración inmutable del BacktestEngine."""
    initial_balance: float = 10000.0
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    horizon_bars: int = 24
    risk_per_trade_pct: float = 0.01
    max_leverage: float = 5.0
    cost_model: CostModel = field(default_factory=CostModel)
    ambiguous_policy: str = "conservative_sl"
    partial_fill_ratio: float = 1.0
    max_entry_gap_pct: Optional[float] = None
    timeframe_minutes: int = 15

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance debe ser estrictamente positivo.")
        if self.tp_mult <= 0 or self.sl_mult <= 0:
            raise ValueError("tp_mult y sl_mult deben ser estrictamente positivos.")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars debe ser >= 1.")
        if not (0.0 < self.risk_per_trade_pct <= 1.0):
            raise ValueError("risk_per_trade_pct debe estar en el rango (0, 1].")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage debe ser estrictamente positivo.")
        if self.ambiguous_policy not in VALID_AMBIGUOUS_POLICIES:
            raise ValueError(
                f"ambiguous_policy inválido: '{self.ambiguous_policy}'. "
                f"Valores permitidos: {VALID_AMBIGUOUS_POLICIES}."
            )
        if not (0.0 < self.partial_fill_ratio <= 1.0):
            raise ValueError("partial_fill_ratio debe estar en el rango (0, 1].")
        if self.max_entry_gap_pct is not None and self.max_entry_gap_pct < 0:
            raise ValueError("max_entry_gap_pct debe ser >= 0 o None.")
        if self.timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes debe ser estrictamente positivo.")


@dataclass
class BacktestResult:
    """Resultado completo e inmutable de una corrida de backtest."""
    metrics: Dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.Series
    config: BacktestConfig


class BacktestEngine:
    """
    Motor de Backtesting de Futuros Perpetuos USD-M con estado limpio y aislado
    por cada invocación de `run(...)`.
    """

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.config = config

    @staticmethod
    def _build_funding_lookup(df_funding: Optional[pd.DataFrame]) -> Dict[int, float]:
        """
        Construye un mapa `{slot_epoch_seconds: funding_rate}` alineado a marcas de 8h UTC.
        Soporta DataFrames con columna `funding_rate` (o `last_funding_rate` / primera numérica)
        o Series indexadas por DatetimeIndex.
        """
        if df_funding is None:
            return {}

        if isinstance(df_funding, pd.Series):
            series = df_funding.dropna()
        else:
            if df_funding.empty:
                return {}
            if "funding_rate" in df_funding.columns:
                series = df_funding["funding_rate"].dropna()
            elif "last_funding_rate" in df_funding.columns:
                series = df_funding["last_funding_rate"].dropna()
            else:
                num_cols = df_funding.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    return {}
                series = df_funding[num_cols[0]].dropna()

        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return {}

        idx_utc = (
            series.index.tz_convert("UTC")
            if series.index.tz is not None
            else series.index.tz_localize("UTC")
        )
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        epoch_sec = ((idx_utc - epoch) // pd.Timedelta("1s")).to_numpy(dtype=np.int64)
        # Normalizamos al slot exacto de 8h (28800s) redondeando al slot más cercano si hubiera desfase de ms/s
        slots = ((epoch_sec + 1800) // 28800) * 28800

        lookup: Dict[int, float] = {}
        for slot_sec, rate in zip(slots, series.to_numpy(dtype=float)):
            if np.isfinite(rate):
                lookup[int(slot_sec)] = float(rate)
        return lookup

    @staticmethod
    def _extract_epoch_seconds(index: pd.Index) -> Optional[np.ndarray]:
        """Extrae timestamps en segundos UTC desde un DatetimeIndex, o None si no aplica."""
        if not isinstance(index, pd.DatetimeIndex):
            return None
        idx_utc = (
            index.tz_convert("UTC")
            if index.tz is not None
            else index.tz_localize("UTC")
        )
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return ((idx_utc - epoch) // pd.Timedelta("1s")).to_numpy(dtype=np.int64)

    @staticmethod
    def _normalize_side(raw_side: Any) -> Optional[str]:
        if raw_side is None:
            return None
        if isinstance(raw_side, float) and math.isnan(raw_side):
            return None
        val = str(raw_side).strip().upper()
        if val in ("LONG", "SHORT"):
            return val
        return None

    def _compute_metrics(
        self,
        df_ohlcv: pd.DataFrame,
        trades_df: pd.DataFrame,
        equity_curve: pd.Series,
        final_equity: float,
        rejected_entries: int,
        bars_in_position: int,
        ambiguous_trades_count: int,
    ) -> Dict[str, Any]:
        cfg = self.config
        initial = float(cfg.initial_balance)
        n_bars = len(df_ohlcv)
        total_trades = int(len(trades_df))

        total_return_pct = ((final_equity - initial) / initial) * 100.0

        # Años transcurridos para CAGR
        years = 0.0
        if n_bars > 1 and isinstance(df_ohlcv.index, pd.DatetimeIndex):
            delta_sec = (df_ohlcv.index[-1] - df_ohlcv.index[0]).total_seconds()
            if delta_sec > 0:
                years = delta_sec / (365.25 * 86400.0)
        if years <= 0.0 and n_bars > 0:
            years = (n_bars * cfg.timeframe_minutes) / (365.25 * 1440.0)

        if years > 0.0 and final_equity > 0.0:
            ratio = final_equity / initial
            if ratio == 1.0:
                cagr_pct = 0.0
            else:
                try:
                    cagr_pct = (math.pow(ratio, 1.0 / years) - 1.0) * 100.0
                except OverflowError:
                    cagr_pct = float("inf") if ratio > 1.0 else -100.0
        elif final_equity <= 0.0:
            cagr_pct = -100.0
        else:
            cagr_pct = 0.0

        # Sharpe y Sortino anualizados sobre retornos barra a barra de equity_curve
        bars_per_year = (365.25 * 24.0 * 60.0) / float(cfg.timeframe_minutes)
        if len(equity_curve) > 1:
            bar_returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        else:
            bar_returns = pd.Series(dtype=float)

        if len(bar_returns) > 1:
            mean_ret = float(bar_returns.mean())
            std_ret = float(bar_returns.std(ddof=1))
            sharpe_ratio = (
                (mean_ret / std_ret) * math.sqrt(bars_per_year)
                if std_ret > 1e-12
                else 0.0
            )
            downside = np.minimum(bar_returns.to_numpy(dtype=float), 0.0)
            downside_rms = float(np.sqrt(np.mean(downside ** 2)))
            sortino_ratio = (
                (mean_ret / downside_rms) * math.sqrt(bars_per_year)
                if downside_rms > 1e-12
                else 0.0
            )
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Max Drawdown (%)
        if len(equity_curve) > 0:
            eq_vals = equity_curve.to_numpy(dtype=float)
            running_peak = np.maximum.accumulate(eq_vals)
            valid_peak = np.where(running_peak > 0.0, running_peak, np.nan)
            dd_series = (running_peak - eq_vals) / valid_peak
            max_dd = float(np.nanmax(dd_series)) if np.any(np.isfinite(dd_series)) else 0.0
            max_drawdown_pct = max(0.0, max_dd * 100.0)
        else:
            max_drawdown_pct = 0.0

        if total_trades > 0:
            net_pnls = trades_df["net_pnl_usdt"].to_numpy(dtype=float)
            r_net = trades_df["realized_r_net"].to_numpy(dtype=float)

            gross_wins = float(np.sum(net_pnls[net_pnls > 0.0]))
            gross_losses = float(np.abs(np.sum(net_pnls[net_pnls < 0.0])))
            if gross_losses > 0.0:
                profit_factor = gross_wins / gross_losses
            elif gross_wins > 0.0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

            expectancy_r = float(np.mean(r_net))
            expectancy_usdt = float(np.mean(net_pnls))
            win_rate_pct = float(np.mean(net_pnls > 0.0) * 100.0)

            r_distribution = {
                "mean": float(np.mean(r_net)),
                "std": float(np.std(r_net, ddof=0)),
                "p10": float(np.percentile(r_net, 10)),
                "p50": float(np.percentile(r_net, 50)),
                "p90": float(np.percentile(r_net, 90)),
                "min": float(np.min(r_net)),
                "max": float(np.max(r_net)),
            }

            avg_duration_bars = float(trades_df["duration_bars"].mean())
            total_fees_usdt = float(trades_df["total_fees_usdt"].sum())
            total_funding_usdt = float(trades_df["funding_paid_usdt"].sum())
            total_slippage_usdt = float(trades_df["slippage_usdt"].sum())

            entry_notional = float(
                np.sum(trades_df["qty"].to_numpy(dtype=float) * trades_df["entry_price"].to_numpy(dtype=float))
            )
            exit_notional = float(
                np.sum(trades_df["qty"].to_numpy(dtype=float) * trades_df["exit_price"].to_numpy(dtype=float))
            )
            turnover = (entry_notional + exit_notional) / initial

            exit_reason_counts = {
                str(k): int(v)
                for k, v in trades_df["exit_reason"].value_counts().items()
            }
        else:
            profit_factor = 0.0
            expectancy_r = 0.0
            expectancy_usdt = 0.0
            win_rate_pct = 0.0
            r_distribution = {
                "mean": 0.0,
                "std": 0.0,
                "p10": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            avg_duration_bars = 0.0
            total_fees_usdt = 0.0
            total_funding_usdt = 0.0
            total_slippage_usdt = 0.0
            turnover = 0.0
            exit_reason_counts = {}

        exposure_pct = (bars_in_position / n_bars * 100.0) if n_bars > 0 else 0.0

        return {
            "total_return_pct": float(total_return_pct),
            "cagr_pct": float(cagr_pct),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown_pct": float(max_drawdown_pct),
            "profit_factor": float(profit_factor),
            "expectancy_r": float(expectancy_r),
            "expectancy_usdt": float(expectancy_usdt),
            "win_rate_pct": float(win_rate_pct),
            "total_trades": total_trades,
            "rejected_entries": int(rejected_entries),
            "exposure_pct": float(exposure_pct),
            "turnover": float(turnover),
            "r_distribution": r_distribution,
            "avg_duration_bars": float(avg_duration_bars),
            "total_fees_usdt": float(total_fees_usdt),
            "total_funding_usdt": float(total_funding_usdt),
            "total_slippage_usdt": float(total_slippage_usdt),
            "exit_reason_counts": exit_reason_counts,
            "ambiguous_trades_count": int(ambiguous_trades_count),
            "final_equity_usdt": float(final_equity),
        }

    def run(
        self,
        df_ohlcv: pd.DataFrame,
        signals: pd.DataFrame,
        df_funding: Optional[pd.DataFrame] = None,
        execution_failures: Optional[Set[Any]] = None,
    ) -> BacktestResult:
        """
        Ejecuta una simulación histórica completamente aislada y determinista.

        Parámetros:
        - `df_ohlcv`: DataFrame con columnas `open`, `high`, `low`, `close`.
        - `signals`: DataFrame alineado al índice de `df_ohlcv` con `side`, `atr_pct`,
          y opcionalmente `risk_fraction`, `p_calibrated`, `ev_net_r`.
        - `df_funding`: DataFrame opcional de funding rates históricos (8h UTC).
        - `execution_failures`: Conjunto opcional de índices enteros `i` o timestamps
          donde la orden de entrada falla / es rechazada.
        """
        required_ohlcv = {"open", "high", "low", "close"}
        missing_ohlcv = required_ohlcv.difference(df_ohlcv.columns)
        if missing_ohlcv:
            raise ValueError(f"df_ohlcv no contiene columnas requeridas: {sorted(missing_ohlcv)}")

        required_signals = {"side", "atr_pct"}
        missing_signals = required_signals.difference(signals.columns)
        if missing_signals:
            raise ValueError(f"signals no contiene columnas requeridas: {sorted(missing_signals)}")

        cfg = self.config
        cost = cfg.cost_model
        n = len(df_ohlcv)

        if n == 0:
            empty_trades = pd.DataFrame(columns=TRADE_COLUMNS)
            empty_eq = pd.Series(dtype=float, index=df_ohlcv.index, name="equity")
            metrics = self._compute_metrics(
                df_ohlcv=df_ohlcv,
                trades_df=empty_trades,
                equity_curve=empty_eq,
                final_equity=float(cfg.initial_balance),
                rejected_entries=0,
                bars_in_position=0,
                ambiguous_trades_count=0,
            )
            return BacktestResult(
                metrics=metrics,
                trades=empty_trades,
                equity_curve=empty_eq,
                config=cfg,
            )

        sig_aligned = signals.reindex(df_ohlcv.index)

        opens = df_ohlcv["open"].to_numpy(dtype=float)
        highs = df_ohlcv["high"].to_numpy(dtype=float)
        lows = df_ohlcv["low"].to_numpy(dtype=float)
        closes = df_ohlcv["close"].to_numpy(dtype=float)
        index_vals = df_ohlcv.index

        sides_raw = sig_aligned["side"].to_numpy(dtype=object)
        atr_pcts = pd.to_numeric(sig_aligned["atr_pct"], errors="coerce").to_numpy(dtype=float)

        has_risk_frac = "risk_fraction" in sig_aligned.columns
        risk_fracs = (
            pd.to_numeric(sig_aligned["risk_fraction"], errors="coerce").to_numpy(dtype=float)
            if has_risk_frac
            else None
        )
        has_p_cal = "p_calibrated" in sig_aligned.columns
        p_cals = (
            pd.to_numeric(sig_aligned["p_calibrated"], errors="coerce").to_numpy(dtype=float)
            if has_p_cal
            else None
        )
        has_ev = "ev_net_r" in sig_aligned.columns
        ev_vals = (
            pd.to_numeric(sig_aligned["ev_net_r"], errors="coerce").to_numpy(dtype=float)
            if has_ev
            else None
        )

        epoch_seconds = self._extract_epoch_seconds(index_vals)
        funding_lookup = self._build_funding_lookup(df_funding)
        failures_set: Set[Any] = set(execution_failures) if execution_failures is not None else set()

        half_spread = float(cost.half_spread_bps) / 10000.0
        slippage = float(cost.slippage_bps) / 10000.0
        taker_fee_rate = float(cost.taker_fee_rate)
        default_funding_rate = float(cost.expected_funding_rate_8h)

        balance = float(cfg.initial_balance)
        equity_arr = np.full(n, balance, dtype=float)

        trades: List[Dict[str, Any]] = []
        rejected_entries = 0
        bars_in_position = 0
        ambiguous_trades_count = 0
        next_free_idx = 0
        trade_counter = 0

        for i in range(n):
            if i < next_free_idx:
                continue

            equity_arr[i] = balance

            side = self._normalize_side(sides_raw[i])
            if side is None:
                continue

            # Se necesita la vela siguiente i+1 para ejecutar en open[i+1]
            if i + 1 >= n:
                continue

            atr_rel = atr_pcts[i]
            raw_open_next = opens[i + 1]
            close_i = closes[i]

            if (
                not np.isfinite(atr_rel)
                or atr_rel <= 0.0
                or not np.isfinite(raw_open_next)
                or raw_open_next <= 0.0
                or not np.isfinite(close_i)
                or close_i <= 0.0
                or balance <= 0.0
            ):
                continue

            # Fallo explícito de ejecución
            if failures_set and (i in failures_set or index_vals[i] in failures_set):
                rejected_entries += 1
                continue

            # Filtro de gap máximo en la entrada
            if cfg.max_entry_gap_pct is not None:
                entry_gap_pct = abs(raw_open_next - close_i) / close_i
                if entry_gap_pct > cfg.max_entry_gap_pct:
                    rejected_entries += 1
                    continue

            # Precio de entrada ejecutable en open[i+1] con half-spread + slippage adverso
            if side == "LONG":
                entry_price = raw_open_next * (1.0 + half_spread + slippage)
            else:
                entry_price = raw_open_next * (1.0 - half_spread - slippage)

            atr_val = atr_rel * entry_price
            sl_dist = cfg.sl_mult * atr_val
            tp_dist = cfg.tp_mult * atr_val
            if sl_dist <= 0.0 or tp_dist <= 0.0:
                continue

            if side == "LONG":
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            # Riesgo por trade y dimensionamiento acotado por apalancamiento + partial_fill_ratio
            if risk_fracs is not None and np.isfinite(risk_fracs[i]) and risk_fracs[i] > 0.0:
                risk_frac = float(risk_fracs[i])
            else:
                risk_frac = float(cfg.risk_per_trade_pct)

            risk_usdt = balance * risk_frac
            max_notional_qty = (balance * cfg.max_leverage) / entry_price
            qty = min(risk_usdt / sl_dist, max_notional_qty) * cfg.partial_fill_ratio
            if not np.isfinite(qty) or qty <= 0.0:
                continue

            risk_1r_usdt = qty * sl_dist
            notional_usdt = qty * entry_price
            entry_fee = notional_usdt * taker_fee_rate
            entry_slippage_usdt = abs(entry_price - raw_open_next) * qty

            balance_before_entry = balance
            balance -= entry_fee

            entry_idx = i + 1
            end_k = min(i + cfg.horizon_bars, n - 1)
            last_funding_slot = (
                int(epoch_seconds[entry_idx] // 28800) * 28800
                if epoch_seconds is not None
                else None
            )

            funding_paid_usdt = 0.0
            exit_idx = end_k
            exit_price = 0.0
            exit_reason = "TIMEOUT"
            ref_exit_price = closes[end_k]
            intrabar_ambiguous = False
            skip_trade = False

            direction_sign = 1.0 if side == "LONG" else -1.0

            for k in range(entry_idx, end_k + 1):
                # 1. Liquidación de Funding Rate cada 8h (00:00, 08:00, 16:00 UTC)
                if epoch_seconds is not None and last_funding_slot is not None:
                    curr_slot = int(epoch_seconds[k] // 28800) * 28800
                    if curr_slot > last_funding_slot:
                        slot_cursor = last_funding_slot + 28800
                        while slot_cursor <= curr_slot:
                            rate = funding_lookup.get(slot_cursor, default_funding_rate)
                            # En LONG: funding > 0 paga (+), funding < 0 cobra (-). En SHORT: inverso.
                            f_cost = (qty * opens[k]) * rate * direction_sign
                            funding_paid_usdt += f_cost
                            balance -= f_cost
                            slot_cursor += 28800
                        last_funding_slot = curr_slot

                bar_open = opens[k]
                bar_high = highs[k]
                bar_low = lows[k]
                bar_close = closes[k]

                # 2. Evaluación de gaps en open[k] y barreras SL / TP
                if side == "LONG":
                    gap_sl = bar_open <= sl_price
                    gap_tp = bar_open >= tp_price
                    hit_sl = bar_low <= sl_price
                    hit_tp = bar_high >= tp_price
                else:
                    gap_sl = bar_open >= sl_price
                    gap_tp = bar_open <= tp_price
                    hit_sl = bar_high >= sl_price
                    hit_tp = bar_low <= tp_price

                closed_on_k = False

                if gap_sl:
                    # Gap adverso: open[k] ya abre más allá de SL -> ejecuta a open[k] + slippage adverso
                    ref_exit_price = bar_open
                    exit_price = (
                        bar_open * (1.0 - slippage)
                        if side == "LONG"
                        else bar_open * (1.0 + slippage)
                    )
                    exit_reason = "STOP_LOSS"
                    exit_idx = k
                    closed_on_k = True
                elif gap_tp:
                    # Gap favorable más allá de TP -> ejecuta a open[k] con slippage de salida
                    ref_exit_price = bar_open
                    exit_price = (
                        bar_open * (1.0 - slippage)
                        if side == "LONG"
                        else bar_open * (1.0 + slippage)
                    )
                    exit_reason = "TAKE_PROFIT"
                    exit_idx = k
                    closed_on_k = True
                elif hit_sl and hit_tp:
                    intrabar_ambiguous = True
                    ambiguous_trades_count += 1
                    exit_idx = k
                    if cfg.ambiguous_policy == "skip":
                        skip_trade = True
                        closed_on_k = True
                    else:
                        # Política conservadora: ejecuta Stop Loss con slippage adverso
                        ref_exit_price = sl_price
                        exit_price = (
                            sl_price * (1.0 - slippage)
                            if side == "LONG"
                            else sl_price * (1.0 + slippage)
                        )
                        exit_reason = "STOP_LOSS"
                        closed_on_k = True
                elif hit_sl:
                    ref_exit_price = sl_price
                    exit_price = (
                        sl_price * (1.0 - slippage)
                        if side == "LONG"
                        else sl_price * (1.0 + slippage)
                    )
                    exit_reason = "STOP_LOSS"
                    exit_idx = k
                    closed_on_k = True
                elif hit_tp:
                    # TAKE_PROFIT_MARKET aplica slippage de salida
                    ref_exit_price = tp_price
                    exit_price = (
                        tp_price * (1.0 - slippage)
                        if side == "LONG"
                        else tp_price * (1.0 + slippage)
                    )
                    exit_reason = "TAKE_PROFIT"
                    exit_idx = k
                    closed_on_k = True
                elif k == end_k:
                    # Cierre por TIMEOUT al close[k] con half-spread + slippage adverso
                    ref_exit_price = bar_close
                    exit_price = (
                        bar_close * (1.0 - half_spread - slippage)
                        if side == "LONG"
                        else bar_close * (1.0 + half_spread + slippage)
                    )
                    exit_reason = "TIMEOUT"
                    exit_idx = k
                    closed_on_k = True

                if closed_on_k:
                    break

                # Mark-to-market mientras la posición sigue abierta al cierre de barra k
                unrealized_gross = (
                    (bar_close - entry_price) * qty
                    if side == "LONG"
                    else (entry_price - bar_close) * qty
                )
                equity_arr[k] = balance + unrealized_gross

            if skip_trade:
                # Revertimos completamente el trade ambiguo omitido para preservar contabilidad exacta
                balance = balance_before_entry
                for k_rev in range(entry_idx, exit_idx + 1):
                    equity_arr[k_rev] = balance
                rejected_entries += 1
                next_free_idx = exit_idx + 1
                continue

            # Cierre efectivo y contabilidad exacta
            gross_pnl_usdt = (
                (exit_price - entry_price) * qty
                if side == "LONG"
                else (entry_price - exit_price) * qty
            )
            exit_fee = (qty * exit_price) * taker_fee_rate
            total_fees_usdt = entry_fee + exit_fee
            exit_slippage_usdt = abs(exit_price - ref_exit_price) * qty
            total_slippage_usdt = entry_slippage_usdt + exit_slippage_usdt

            balance += gross_pnl_usdt - exit_fee
            equity_arr[exit_idx] = balance

            net_pnl_usdt = gross_pnl_usdt - total_fees_usdt - funding_paid_usdt
            realized_r_gross = gross_pnl_usdt / risk_1r_usdt if risk_1r_usdt > 0.0 else 0.0
            realized_r_net = net_pnl_usdt / risk_1r_usdt if risk_1r_usdt > 0.0 else 0.0

            duration_bars = exit_idx - entry_idx + 1
            bars_in_position += duration_bars
            trade_counter += 1

            trades.append(
                {
                    "trade_id": trade_counter,
                    "side": side,
                    "signal_idx": int(i),
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "signal_time": index_vals[i],
                    "entry_time": index_vals[entry_idx],
                    "exit_time": index_vals[exit_idx],
                    "duration_bars": int(duration_bars),
                    "qty": float(qty),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "sl_price": float(sl_price),
                    "tp_price": float(tp_price),
                    "notional_usdt": float(notional_usdt),
                    "risk_1r_usdt": float(risk_1r_usdt),
                    "gross_pnl_usdt": float(gross_pnl_usdt),
                    "entry_fee_usdt": float(entry_fee),
                    "exit_fee_usdt": float(exit_fee),
                    "total_fees_usdt": float(total_fees_usdt),
                    "funding_paid_usdt": float(funding_paid_usdt),
                    "slippage_usdt": float(total_slippage_usdt),
                    "net_pnl_usdt": float(net_pnl_usdt),
                    "realized_r_gross": float(realized_r_gross),
                    "realized_r_net": float(realized_r_net),
                    "exit_reason": exit_reason,
                    "intrabar_ambiguous": bool(intrabar_ambiguous),
                    "p_calibrated": (
                        float(p_cals[i])
                        if p_cals is not None and np.isfinite(p_cals[i])
                        else float("nan")
                    ),
                    "ev_net_r": (
                        float(ev_vals[i])
                        if ev_vals is not None and np.isfinite(ev_vals[i])
                        else float("nan")
                    ),
                    "balance_after_usdt": float(balance),
                }
            )

            # Sin re-entrada en la misma barra de cierre
            next_free_idx = exit_idx + 1

        trades_df = (
            pd.DataFrame(trades, columns=TRADE_COLUMNS)
            if trades
            else pd.DataFrame(columns=TRADE_COLUMNS)
        )
        equity_curve = pd.Series(equity_arr, index=index_vals, name="equity")

        metrics = self._compute_metrics(
            df_ohlcv=df_ohlcv,
            trades_df=trades_df,
            equity_curve=equity_curve,
            final_equity=balance,
            rejected_entries=rejected_entries,
            bars_in_position=bars_in_position,
            ambiguous_trades_count=ambiguous_trades_count,
        )

        return BacktestResult(
            metrics=metrics,
            trades=trades_df,
            equity_curve=equity_curve,
            config=cfg,
        )
