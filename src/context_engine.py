"""
Context Engine (Fase 4 completa) — Los 6 regímenes causales de mercado y eventos externos
con Control Temporal Cuádruple (§6 del Plan Maestro).

Regímenes causales generados (`ctx_*`):
1. Market Regime      : `ctx_market_regime` (`-1` bearish, `0` sideways si `|ema_1d - ema_5d| / close < 0.003`, `+1` bullish).
2. Volatility Regime  : `ctx_vol_regime_pct`, `ctx_vol_regime_state` (`0` low < 0.33, `1` normal, `2` high > 0.67).
3. Liquidity Regime   : `ctx_liquidity_regime_zscore` (z-score causal de volumen rodante 24h vs ventana 15d).
4. BTC Regime         : `ctx_btc_bull_regime`, `ctx_btc_momentum_24h` (retorno 24h normalizado por ATR).
5. Altcoin Regime     : `ctx_eth_btc_rel_24h`, `ctx_alt_strength_regime` (`1` si ETH/BTC > EMA 24h de ETH/BTC, `0` si no; causal sin `.bfill()`).
6. Derivatives Regime : `ctx_deriv_funding_regime` (`-1` si funding < -0.0001, `1` si > 0.0003, `0` neutral; causal sin `.bfill()`).

Control Temporal Cuádruple para eventos externos (`ExternalEvent`):
1. `event_time`       : Momento UTC en que ocurre el evento.
2. `publication_time` : Momento UTC en que se publica oficialmente la noticia/calendario.
3. `effective_time`   : Momento UTC en que el evento entra en vigor.
4. `available_at`     : Momento UTC a partir del cual el dato es consumible por el sistema.

Un evento SOLO puede influir en una vela con timestamp `t` si `available_at <= t`.
CERO LOOK-AHEAD BIAS: Prohibido `.bfill()` en alineaciones cross-asset (`df_eth`) y derivados (`df_funding`).
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd


# Calendario histórico y proyectado de reuniones FOMC (2024 - 2027)
FOMC_DATES = [
    "2024-01-31 19:00:00", "2024-03-20 18:00:00", "2024-05-01 18:00:00", "2024-06-12 18:00:00",
    "2024-07-31 18:00:00", "2024-09-18 18:00:00", "2024-11-07 19:00:00", "2024-12-18 19:00:00",
    "2025-01-29 19:00:00", "2025-03-19 18:00:00", "2025-05-07 18:00:00", "2025-06-18 18:00:00",
    "2025-07-30 18:00:00", "2025-09-17 18:00:00", "2025-10-29 18:00:00", "2025-12-10 19:00:00",
    "2026-01-28 19:00:00", "2026-03-18 18:00:00", "2026-04-29 18:00:00", "2026-06-17 18:00:00",
    "2026-07-29 18:00:00", "2026-09-16 18:00:00", "2026-10-28 18:00:00", "2026-12-09 19:00:00",
    "2027-01-27 19:00:00", "2027-03-17 18:00:00", "2027-04-28 18:00:00", "2027-06-09 18:00:00",
    "2027-07-28 18:00:00", "2027-09-15 18:00:00", "2027-10-27 18:00:00", "2027-12-08 19:00:00",
]


def _to_utc_timestamp(value) -> pd.Timestamp:
    """Convierte cualquier representación temporal a pd.Timestamp con zona horaria UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class ExternalEvent:
    """
    Evento externo macro genérico con control temporal cuádruple
    (todos los timestamps normalizados a UTC).
    """
    name: str
    event_time: pd.Timestamp
    publication_time: pd.Timestamp
    effective_time: pd.Timestamp
    available_at: pd.Timestamp
    category: str = "macro"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", _to_utc_timestamp(self.event_time))
        object.__setattr__(self, "publication_time", _to_utc_timestamp(self.publication_time))
        object.__setattr__(self, "effective_time", _to_utc_timestamp(self.effective_time))
        object.__setattr__(self, "available_at", _to_utc_timestamp(self.available_at))


def build_default_fomc_events(
    dates: Sequence[str] = FOMC_DATES,
    calendar_lead_days: int = 30,
) -> Tuple[ExternalEvent, ...]:
    """
    Construye los eventos del calendario oficial FOMC.
    Al ser un calendario público publicado con meses de antelación por la Reserva Federal,
    `publication_time` y `available_at` preceden a `event_time`.
    """
    lead = pd.Timedelta(days=calendar_lead_days)
    events = []
    for raw_dt in dates:
        ev_ts = _to_utc_timestamp(raw_dt)
        pub_ts = ev_ts - lead
        events.append(
            ExternalEvent(
                name=f"FOMC_{ev_ts.strftime('%Y%m%d')}",
                event_time=ev_ts,
                publication_time=pub_ts,
                effective_time=ev_ts,
                available_at=pub_ts,
                category="FOMC",
            )
        )
    return tuple(events)


class ContextEngine:
    """
    Motor de contexto macro, 6 regímenes causales y eventos externos (`ctx_*`).
    Estrictamente causal: sin `.bfill()` y con filtro `available_at <= bar_timestamp`.
    """

    def __init__(
        self,
        events: Optional[Sequence[ExternalEvent]] = None,
        pre_event_hours: float = 24.0,
        post_event_hours: float = 2.0,
    ):
        self.events: Tuple[ExternalEvent, ...] = (
            build_default_fomc_events() if events is None else tuple(events)
        )
        self.pre_event_hours = float(pre_event_hours)
        self.post_event_hours = float(post_event_hours)

    def build_context(
        self,
        df_btc: pd.DataFrame,
        df_eth: Optional[pd.DataFrame] = None,
        atr_pct: Optional[pd.Series] = None,
        df_funding: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calcula las features de contexto (`ctx_*`) alineadas estrictamente al índice de `df_btc`:
        - Market Regime      : `ctx_market_regime` (-1 bearish, 0 sideways, +1 bullish).
        - Volatility Regime  : `ctx_vol_regime_pct`, `ctx_vol_regime_state` (0 low, 1 normal, 2 high).
        - Liquidity Regime   : `ctx_liquidity_regime_zscore` (z-score causal volumen 24h vs 15d).
        - BTC Regime         : `ctx_btc_bull_regime`, `ctx_btc_momentum_24h`.
        - Altcoin Regime     : `ctx_eth_btc_rel_24h`, `ctx_alt_strength_regime` (solo `ffill`, sin `bfill`).
        - Derivatives Regime : `ctx_deriv_funding_regime` (-1, 0, +1; solo `ffill`, sin `bfill`).
        - Macro Catalyst     : `ctx_fomc_event_24h` respetando `available_at <= bar_timestamp`.
        """
        ctx = pd.DataFrame(index=df_btc.index)
        close = df_btc["close"]
        volume = df_btc["volume"]

        if atr_pct is None:
            atr_rel = self._compute_causal_atr_pct(df_btc)
        else:
            atr_rel = atr_pct.reindex(ctx.index)

        # 1. Market Regime & 4. BTC Regime (1D = 96 velas vs 5D = 480 velas)
        ema_1d = close.ewm(span=96, adjust=False).mean()
        ema_5d = close.ewm(span=96 * 5, adjust=False).mean()
        ctx["ctx_btc_bull_regime"] = (ema_1d > ema_5d).astype(int)

        rel_ema_diff = (ema_1d - ema_5d) / (close + 1e-8)
        ctx["ctx_market_regime"] = np.where(
            rel_ema_diff.abs() < 0.003,
            0,
            np.where(rel_ema_diff > 0.0, 1, -1),
        ).astype(int)

        ctx["ctx_btc_momentum_24h"] = close.pct_change(96) / (atr_rel + 1e-8)

        # 2. Volatility Regime (percentil rodante causal de 15 días + estado discreto 0/1/2)
        vol_24h = close.pct_change().rolling(96, min_periods=24).std()
        vol_pct = vol_24h.rolling(96 * 15, min_periods=96).rank(pct=True)
        ctx["ctx_vol_regime_pct"] = vol_pct

        vol_state = pd.Series(np.nan, index=ctx.index, dtype=float)
        valid_vol = vol_pct.notna()
        vol_state.loc[valid_vol] = np.where(
            vol_pct.loc[valid_vol] < 0.33,
            0.0,
            np.where(vol_pct.loc[valid_vol] > 0.67, 2.0, 1.0),
        )
        ctx["ctx_vol_regime_state"] = vol_state

        # 3. Liquidity Regime (z-score causal de volumen rodante 24h vs ventana 15d)
        vol_roll_24h = volume.rolling(96, min_periods=24).sum()
        liq_mean_15d = vol_roll_24h.rolling(96 * 15, min_periods=96).mean()
        liq_std_15d = vol_roll_24h.rolling(96 * 15, min_periods=96).std()
        ctx["ctx_liquidity_regime_zscore"] = (vol_roll_24h - liq_mean_15d) / (
            liq_std_15d + 1e-8
        )

        # 5. Altcoin Regime (flujo relativo ETH/BTC estrictamente causal, SIN .bfill())
        if df_eth is not None and not df_eth.empty:
            eth_series = df_eth["close"].copy()
            eth_series = eth_series[~eth_series.index.duplicated(keep="last")].sort_index()
            eth_close = eth_series.reindex(ctx.index, method="ffill")

            eth_btc = eth_close / (close + 1e-8)
            ctx["ctx_eth_btc_rel_24h"] = (
                eth_btc.pct_change(4, fill_method=None) / (atr_rel + 1e-8)
            ).clip(-3.0, 3.0)

            eth_btc_ema_24h = eth_btc.ewm(span=96, adjust=False, min_periods=1).mean()
            alt_strength = pd.Series(np.nan, index=ctx.index, dtype=float)
            valid_eth = eth_btc.notna() & eth_btc_ema_24h.notna()
            alt_strength.loc[valid_eth] = (
                eth_btc.loc[valid_eth] > eth_btc_ema_24h.loc[valid_eth]
            ).astype(float)
            ctx["ctx_alt_strength_regime"] = alt_strength
        else:
            ctx["ctx_eth_btc_rel_24h"] = np.nan
            ctx["ctx_alt_strength_regime"] = np.nan

        # 6. Derivatives Regime (funding rate causal sin .bfill())
        if df_funding is not None and not df_funding.empty:
            f_series = df_funding["funding_rate"].copy()
            f_series.index = pd.to_datetime(f_series.index, utc=True).floor("1min")
            f_series = f_series[~f_series.index.duplicated(keep="last")].sort_index()
            f_rate = f_series.reindex(ctx.index, method="ffill")

            deriv_regime = pd.Series(np.nan, index=ctx.index, dtype=float)
            valid_f = f_rate.notna()
            deriv_regime.loc[valid_f] = np.where(
                f_rate.loc[valid_f] < -0.0001,
                -1.0,
                np.where(f_rate.loc[valid_f] > 0.0003, 1.0, 0.0),
            )
            ctx["ctx_deriv_funding_regime"] = deriv_regime
        else:
            ctx["ctx_deriv_funding_regime"] = 0.0

        # 7. Catalizador Macro (ventana de eventos con filtro estricto available_at <= bar_timestamp)
        ctx["ctx_fomc_event_24h"] = self._build_event_flag(ctx.index)

        return ctx

    def _build_event_flag(
        self,
        index: pd.Index,
        categories: Optional[Sequence[str]] = None,
    ) -> pd.Series:
        """
        Marca 1 en las velas dentro de `[event_time - pre_event_hours, event_time + post_event_hours]`
        ÚNICAMENTE cuando `ev.available_at <= bar_timestamp` (previene leakage por publicación tardía).
        Si `categories` es None, evalúa todos los eventos registrados en `self.events`.
        """
        flag = pd.Series(0, index=index, dtype=int)
        if len(index) == 0 or not self.events:
            return flag

        idx_utc = pd.to_datetime(index, utc=True)
        idx_s = pd.Series(idx_utc, index=index)
        pre_delta = pd.Timedelta(hours=self.pre_event_hours)
        post_delta = pd.Timedelta(hours=self.post_event_hours)
        cat_filter = set(categories) if categories is not None else None

        for ev in self.events:
            if cat_filter is not None and ev.category not in cat_filter:
                continue
            window_start = ev.event_time - pre_delta
            window_end = ev.event_time + post_delta
            mask = (
                (idx_s >= ev.available_at)
                & (idx_s >= window_start)
                & (idx_s <= window_end)
            )
            flag.loc[mask] = 1

        return flag

    @staticmethod
    def _compute_causal_atr_pct(df_btc: pd.DataFrame) -> pd.Series:
        high = df_btc["high"]
        low = df_btc["low"]
        close = df_btc["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        return (atr_14 / close).clip(lower=1e-4)

