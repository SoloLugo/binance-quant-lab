"""
Feature Engine (Fase 3 base) — Construcción modular y estrictamente causal de features.

Familias activables independientemente:
- `"ta"`             : Análisis técnico causal (`atr_pct`, `ta_*`).
- `"microstructure"` : Proxies de microestructura / flujo de órdenes (`sm_*`).
- `"derivatives"`    : Derivados y funding rate con alineación causal `ffill` (`deriv_*`).

Regla estricta de integridad temporal:
- CERO LOOK-AHEAD BIAS: prohibido `.bfill()` o cualquier uso de información futura.
- Las barras iniciales anteriores a la primera observación de `df_funding` permanecen
  como `NaN` de forma explícita y auditable.
"""
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd


ALL_FEATURE_FAMILIES: Tuple[str, ...] = ("ta", "microstructure", "derivatives")


class FeatureEngine:
    """Motor causal de features con familias activables independientemente."""

    def __init__(self, enabled_families: Optional[Sequence[str]] = None):
        if enabled_families is None:
            families = ALL_FEATURE_FAMILIES
        else:
            families = tuple(enabled_families)
            unknown = [f for f in families if f not in ALL_FEATURE_FAMILIES]
            if unknown:
                raise ValueError(
                    f"Familias de features desconocidas: {unknown}. "
                    f"Familias válidas: {ALL_FEATURE_FAMILIES}."
                )
        self.enabled_families: Tuple[str, ...] = families

    def build_features(
        self,
        df_btc: pd.DataFrame,
        df_funding: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Construye un DataFrame de features estrictamente causales alineado al índice de `df_btc`.
        Solo se calculan las familias presentes en `self.enabled_families`.
        """
        feat = pd.DataFrame(index=df_btc.index)

        if "ta" in self.enabled_families:
            ta_df = self._build_ta_features(df_btc)
            feat = pd.concat([feat, ta_df], axis=1)

        if "microstructure" in self.enabled_families:
            sm_df = self._build_microstructure_features(df_btc)
            feat = pd.concat([feat, sm_df], axis=1)

        if "derivatives" in self.enabled_families:
            deriv_df = self._build_derivatives_features(df_btc.index, df_funding)
            feat = pd.concat([feat, deriv_df], axis=1)

        return feat

    @staticmethod
    def _build_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        """Familia A: Indicadores técnicos estrictamente causales (`atr_pct`, `ta_*`)."""
        out = pd.DataFrame(index=df.index)
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # ATR de 14 periodos (base para volatilidad y normalización)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_pct = (atr_14 / close).clip(lower=1e-4)
        out["atr_pct"] = atr_pct

        # Retornos normalizados por volatilidad en múltiples horizontes
        out["ta_ret_1"] = close.pct_change(1) / (atr_pct + 1e-8)
        out["ta_ret_4"] = close.pct_change(4) / (atr_pct + 1e-8)
        out["ta_ret_16"] = close.pct_change(16) / (atr_pct + 1e-8)

        # Distancia a EMAs normalizada por ATR
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        out["ta_dist_ema20"] = (close - ema_20) / (atr_14 + 1e-8)
        out["ta_dist_ema50"] = (close - ema_50) / (atr_14 + 1e-8)

        # RSI de 14 periodos
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / (loss + 1e-8)
        out["ta_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Ancho de bandas de Bollinger
        bb_std = close.rolling(20, min_periods=20).std()
        out["ta_bb_width"] = (4.0 * bb_std) / (ema_20 + 1e-8)

        # Momentum de 16 velas (log-retorno causal)
        out["ta_momentum_16"] = np.log((close + 1e-12) / (close.shift(16) + 1e-12))

        # Volatilidad realizada de 24 velas
        out["ta_volatility_24"] = close.pct_change(1).rolling(24, min_periods=24).std()

        # Volumen relativo respecto a su media móvil de 20 velas
        vol_ma_20 = volume.rolling(20, min_periods=20).mean()
        out["ta_rel_volume_20"] = volume / (vol_ma_20 + 1e-8)

        # Distancia al VWAP rodante causal de 24 velas normalizada por ATR
        typical_price = (high + low + close) / 3.0
        vwap_num = (typical_price * volume).rolling(24, min_periods=24).sum()
        vwap_den = volume.rolling(24, min_periods=24).sum()
        vwap_24 = vwap_num / (vwap_den + 1e-8)
        out["ta_vwap_dist_24"] = (close - vwap_24) / (atr_14 + 1e-8)

        # Distancia a máximos y mínimos rodantes de 24 velas normalizada por ATR
        roll_high_24 = high.rolling(24, min_periods=24).max()
        roll_low_24 = low.rolling(24, min_periods=24).min()
        out["ta_dist_high_24"] = (close - roll_high_24) / (atr_14 + 1e-8)
        out["ta_dist_low_24"] = (close - roll_low_24) / (atr_14 + 1e-8)

        # Proporción del cuerpo de la vela respecto a su rango total (con signo)
        out["ta_body_ratio"] = (close - open_) / ((high - low) + 1e-8)

        return out

    @staticmethod
    def _build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Familia B: Proxies de microestructura y orderflow (`sm_*`).

        Nota explícita sobre `sm_avg_trade_size_zscore` / `sm_whale_trade_zscore`:
        Ambos reflejan un proxy estadístico del tamaño medio por transacción
        (`quote_volume / count`) en ventana rodante causal de 96 velas; NO constituyen
        una detección confirmada de ballenas ni de identidad de contrapartes.
        `sm_whale_trade_zscore` se mantiene únicamente como alias de compatibilidad temporal.
        """
        out = pd.DataFrame(index=df.index)
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        vol = df["volume"].replace(0, np.nan)
        taker_buy = df["taker_buy_volume"]
        taker_sell = vol - taker_buy

        # Order Flow Imbalance (OFI) causal
        ofi = (taker_buy - taker_sell) / (vol + 1e-8)
        out["sm_ofi_1"] = ofi
        out["sm_ofi_cum_4"] = ofi.rolling(4, min_periods=1).mean()
        out["sm_ofi_cum_16"] = ofi.rolling(16, min_periods=4).mean()

        # Proxy de tamaño medio por transacción (NO detección confirmada de ballenas)
        avg_trade_size = df["quote_volume"] / (df["count"].replace(0, np.nan) + 1e-8)
        avg_trade_size_z = (
            avg_trade_size - avg_trade_size.rolling(96, min_periods=24).mean()
        ) / (avg_trade_size.rolling(96, min_periods=24).std() + 1e-8)
        out["sm_avg_trade_size_zscore"] = avg_trade_size_z
        # Alias de compatibilidad temporal (proxy de tamaño medio de trade, no ballenas confirmadas)
        out["sm_whale_trade_zscore"] = avg_trade_size_z

        # Ratios de mechas (absorción/rechazo intrabar)
        candle_range = (high - low) + 1e-8
        out["sm_lower_wick_ratio"] = (np.minimum(open_, close) - low) / candle_range
        out["sm_upper_wick_ratio"] = (high - np.maximum(open_, close)) / candle_range

        return out

    @staticmethod
    def _build_derivatives_features(
        target_index: pd.Index,
        df_funding: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Familia C: Derivados y funding rate (`deriv_*`) con alineación estrictamente causal.
        Usa exclusivamente `reindex(target_index, method="ffill")` sin `.bfill()`.
        Las barras anteriores a la primera observación disponible quedan en `NaN`.
        """
        out = pd.DataFrame(index=target_index)

        if df_funding is not None and not df_funding.empty:
            f_series = df_funding["funding_rate"].copy()
            f_series.index = pd.to_datetime(f_series.index, utc=True).floor("1min")
            f_series = f_series[~f_series.index.duplicated(keep="last")].sort_index()
            f_rate = f_series.reindex(target_index, method="ffill")
        else:
            f_rate = pd.Series(np.nan, index=target_index, dtype=float)

        out["deriv_funding_rate"] = f_rate
        f_mean = f_rate.rolling(96 * 7, min_periods=96).mean()
        f_std = f_rate.rolling(96 * 7, min_periods=96).std()
        out["deriv_funding_zscore"] = (f_rate - f_mean) / (f_std + 1e-8)
        out["deriv_funding_change_24h"] = f_rate - f_rate.shift(96)

        return out
