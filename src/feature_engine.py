"""
Feature Engine (Fase 3 completa) — Construcción modular y estrictamente causal de features.

Familias activables independientemente:
- `"ta"`               : Análisis técnico causal (`atr_pct`, `ta_*`).
- `"microstructure"`   : Proxies de microestructura / flujo de órdenes (`sm_*`).
- `"derivatives"`      : Derivados, funding rate y métricas opcionales de derivados con
                         alineación causal `ffill` (`deriv_*`).
- `"market_structure"` : Estructura de mercado / Smart Money 100% algorítmico y causal
                         (`ms_*`: swings confirmados con lag `swing_right`, BOS, CHoCH,
                         Equal Highs/Lows, Liquidity Sweeps, FVG, Order Blocks, Premium/Discount).

Regla estricta de integridad temporal:
- CERO LOOK-AHEAD BIAS: prohibido `.bfill()` o cualquier uso de información futura (`> t`).
- Las barras iniciales anteriores a la primera observación de `df_funding` o `df_derivatives`
  permanecen como `NaN` de forma explícita y auditable.
"""
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd


ALL_FEATURE_FAMILIES: Tuple[str, ...] = (
    "ta",
    "microstructure",
    "derivatives",
    "market_structure",
)


class FeatureEngine:
    """Motor causal de features con familias activables independientemente."""

    def __init__(
        self,
        enabled_families: Optional[Sequence[str]] = None,
        swing_left: int = 3,
        swing_right: int = 3,
        equal_tol_atr: float = 0.15,
    ):
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
        if swing_left < 1 or swing_right < 1:
            raise ValueError("swing_left y swing_right deben ser enteros >= 1.")
        if equal_tol_atr <= 0:
            raise ValueError("equal_tol_atr debe ser estrictamente positivo.")

        self.enabled_families: Tuple[str, ...] = families
        self.swing_left: int = int(swing_left)
        self.swing_right: int = int(swing_right)
        self.equal_tol_atr: float = float(equal_tol_atr)

    def build_features(
        self,
        df_btc: pd.DataFrame,
        df_funding: Optional[pd.DataFrame] = None,
        df_derivatives: Optional[pd.DataFrame] = None,
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
            deriv_df = self._build_derivatives_features(
                df_btc.index,
                df_funding=df_funding,
                df_derivatives=df_derivatives,
            )
            feat = pd.concat([feat, deriv_df], axis=1)

        if "market_structure" in self.enabled_families:
            ms_df = self._build_market_structure_features(
                df_btc,
                swing_left=self.swing_left,
                swing_right=self.swing_right,
                equal_tol_atr=self.equal_tol_atr,
            )
            feat = pd.concat([feat, ms_df], axis=1)

        return feat

    @staticmethod
    def _compute_causal_atr_14(df: pd.DataFrame) -> pd.Series:
        """ATR de 14 periodos estrictamente causal."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(14, min_periods=14).mean()

    @classmethod
    def _build_ta_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Familia A: Indicadores técnicos estrictamente causales (`atr_pct`, `ta_*`)."""
        out = pd.DataFrame(index=df.index)
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # ATR de 14 periodos (base para volatilidad y normalización)
        atr_14 = cls._compute_causal_atr_14(df)
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
        Familia B (microestructura): Proxies de microestructura y orderflow (`sm_*`).

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

    @classmethod
    def _build_market_structure_features(
        cls,
        df: pd.DataFrame,
        swing_left: int = 3,
        swing_right: int = 3,
        equal_tol_atr: float = 0.15,
    ) -> pd.DataFrame:
        """
        Familia B algorítmica: Market Structure / Smart Money (`ms_*`) 100% causal.

        Cada barra `t` utiliza exclusivamente información de barras `<= t`:
        - Un pivot ocurrido en `p = t - swing_right` solo se confirma en la barra `t`
          cuando es extremo en la ventana `[p - swing_left, p + swing_right]` (= `[p - swing_left, t]`).
        - BOS, CHoCH, Equal Highs/Lows, Liquidity Sweeps, FVG, Order Blocks y
          Premium/Discount se computan únicamente sobre pivots y velas ya cerrados en `<= t`.
        """
        n = len(df)
        open_arr = df["open"].to_numpy(dtype=float)
        high_arr = df["high"].to_numpy(dtype=float)
        low_arr = df["low"].to_numpy(dtype=float)
        close_arr = df["close"].to_numpy(dtype=float)
        atr_arr = cls._compute_causal_atr_14(df).to_numpy(dtype=float)

        dist_last_swing_high = np.full(n, np.nan, dtype=float)
        dist_last_swing_low = np.full(n, np.nan, dtype=float)
        bullish_bos = np.zeros(n, dtype=int)
        bearish_bos = np.zeros(n, dtype=int)
        bullish_choch = np.zeros(n, dtype=int)
        bearish_choch = np.zeros(n, dtype=int)
        equal_highs = np.zeros(n, dtype=int)
        equal_lows = np.zeros(n, dtype=int)
        bullish_sweep_strength = np.zeros(n, dtype=float)
        bearish_sweep_strength = np.zeros(n, dtype=float)
        bullish_fvg = np.zeros(n, dtype=int)
        bearish_fvg = np.zeros(n, dtype=int)
        fvg_size_atr = np.zeros(n, dtype=float)
        dist_bull_ob = np.full(n, np.nan, dtype=float)
        bull_ob_age = np.full(n, np.nan, dtype=float)
        dist_bear_ob = np.full(n, np.nan, dtype=float)
        bear_ob_age = np.full(n, np.nan, dtype=float)
        premium_discount_pos = np.full(n, np.nan, dtype=float)

        last_sh: Optional[float] = None
        prev_sh: Optional[float] = None
        sh_broken = False

        last_sl: Optional[float] = None
        prev_sl: Optional[float] = None
        sl_broken = False

        # Estado de estructura vigente: 0 = indefinida, +1 = alcista, -1 = bajista
        trend_state = 0

        bull_ob_price: Optional[float] = None
        bull_ob_idx: Optional[int] = None
        bear_ob_price: Optional[float] = None
        bear_ob_idx: Optional[int] = None

        min_confirm_t = swing_left + swing_right

        for t in range(n):
            atr_t = atr_arr[t]
            atr_valid = np.isfinite(atr_t)
            atr_denom = atr_t + 1e-8 if atr_valid else np.nan

            # 1. Confirmación estrictamente causal de Swing High / Swing Low en p = t - swing_right
            if t >= min_confirm_t:
                p = t - swing_right
                win_start = p - swing_left
                win_end = t + 1  # slice exclusivo -> incluye hasta t (NUNCA > t)

                if high_arr[p] == np.max(high_arr[win_start:win_end]):
                    prev_sh = last_sh
                    last_sh = float(high_arr[p])
                    sh_broken = False

                if low_arr[p] == np.min(low_arr[win_start:win_end]):
                    prev_sl = last_sl
                    last_sl = float(low_arr[p])
                    sl_broken = False

            # 2. Distancias a últimos swings confirmados (en unidades de ATR)
            if last_sh is not None and atr_valid:
                dist_last_swing_high[t] = (close_arr[t] - last_sh) / atr_denom
            if last_sl is not None and atr_valid:
                dist_last_swing_low[t] = (close_arr[t] - last_sl) / atr_denom

            # 3. Equal Highs / Equal Lows
            if prev_sh is not None and last_sh is not None and atr_valid:
                if abs(last_sh - prev_sh) < equal_tol_atr * atr_t:
                    equal_highs[t] = 1
            if prev_sl is not None and last_sl is not None and atr_valid:
                if abs(last_sl - prev_sl) < equal_tol_atr * atr_t:
                    equal_lows[t] = 1

            # 4. Liquidity Sweeps (mecha perfora swing confirmado y cierra dentro del rango)
            if last_sl is not None and low_arr[t] < last_sl and close_arr[t] > last_sl:
                if atr_valid:
                    bullish_sweep_strength[t] = (last_sl - low_arr[t]) / atr_denom
            if last_sh is not None and high_arr[t] > last_sh and close_arr[t] < last_sh:
                if atr_valid:
                    bearish_sweep_strength[t] = (high_arr[t] - last_sh) / atr_denom

            # 5. BOS y CHoCH estrictamente causales
            bull_break = last_sh is not None and (not sh_broken) and close_arr[t] > last_sh
            bear_break = last_sl is not None and (not sl_broken) and close_arr[t] < last_sl

            if bull_break:
                if trend_state == -1:
                    bullish_choch[t] = 1
                else:
                    bullish_bos[t] = 1
                trend_state = 1
                sh_broken = True

                # Bullish Order Block: última vela bajista (close < open) previa a la ruptura (k < t)
                for k in range(t - 1, -1, -1):
                    if close_arr[k] < open_arr[k]:
                        bull_ob_price = 0.5 * (open_arr[k] + close_arr[k])
                        bull_ob_idx = k
                        break

            if bear_break:
                if trend_state == 1:
                    bearish_choch[t] = 1
                else:
                    bearish_bos[t] = 1
                trend_state = -1
                sl_broken = True

                # Bearish Order Block: última vela alcista (close > open) previa a la ruptura (k < t)
                for k in range(t - 1, -1, -1):
                    if close_arr[k] > open_arr[k]:
                        bear_ob_price = 0.5 * (open_arr[k] + close_arr[k])
                        bear_ob_idx = k
                        break

            # 6. Fair Value Gaps de 3 velas [t-2, t-1, t] (conocido al cierre de t)
            if t >= 2:
                if low_arr[t] > high_arr[t - 2]:
                    bullish_fvg[t] = 1
                    if atr_valid:
                        fvg_size_atr[t] = (low_arr[t] - high_arr[t - 2]) / atr_denom
                elif high_arr[t] < low_arr[t - 2]:
                    bearish_fvg[t] = 1
                    if atr_valid:
                        fvg_size_atr[t] = (low_arr[t - 2] - high_arr[t]) / atr_denom

            # 7. Order Blocks activos (distancia en ATR y antigüedad en velas)
            if bull_ob_price is not None and bull_ob_idx is not None:
                bull_ob_age[t] = float(t - bull_ob_idx)
                if atr_valid:
                    dist_bull_ob[t] = (close_arr[t] - bull_ob_price) / atr_denom

            if bear_ob_price is not None and bear_ob_idx is not None:
                bear_ob_age[t] = float(t - bear_ob_idx)
                if atr_valid:
                    dist_bear_ob[t] = (close_arr[t] - bear_ob_price) / atr_denom

            # 8. Premium / Discount dentro del rango causal [last_swing_low, last_swing_high]
            if last_sh is not None and last_sl is not None and last_sh > last_sl:
                pos = (close_arr[t] - last_sl) / (last_sh - last_sl)
                premium_discount_pos[t] = float(np.clip(pos, -0.5, 1.5))

        return pd.DataFrame(
            {
                "ms_dist_last_swing_high": dist_last_swing_high,
                "ms_dist_last_swing_low": dist_last_swing_low,
                "ms_bullish_bos": bullish_bos,
                "ms_bearish_bos": bearish_bos,
                "ms_bullish_choch": bullish_choch,
                "ms_bearish_choch": bearish_choch,
                "ms_equal_highs": equal_highs,
                "ms_equal_lows": equal_lows,
                "ms_bullish_sweep_strength": bullish_sweep_strength,
                "ms_bearish_sweep_strength": bearish_sweep_strength,
                "ms_bullish_fvg": bullish_fvg,
                "ms_bearish_fvg": bearish_fvg,
                "ms_fvg_size_atr": fvg_size_atr,
                "ms_dist_bull_ob": dist_bull_ob,
                "ms_bull_ob_age": bull_ob_age,
                "ms_dist_bear_ob": dist_bear_ob,
                "ms_bear_ob_age": bear_ob_age,
                "ms_premium_discount_pos": premium_discount_pos,
            },
            index=df.index,
        )

    @staticmethod
    def _align_causal_series(series: pd.Series, target_index: pd.Index) -> pd.Series:
        """Alinea una serie temporal al índice objetivo usando exclusivamente `ffill` (SIN `.bfill()`)."""
        s = series.copy()
        s.index = pd.to_datetime(s.index, utc=True).floor("1min")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(target_index, method="ffill")

    @classmethod
    def _build_derivatives_features(
        cls,
        target_index: pd.Index,
        df_funding: Optional[pd.DataFrame],
        df_derivatives: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Familia C: Derivados y funding rate (`deriv_*`) con alineación estrictamente causal.
        Usa exclusivamente `reindex(target_index, method="ffill")` sin `.bfill()`.
        Las barras anteriores a la primera observación disponible quedan en `NaN`.
        """
        out = pd.DataFrame(index=target_index)

        if df_funding is not None and not df_funding.empty:
            f_rate = cls._align_causal_series(df_funding["funding_rate"], target_index)
        else:
            f_rate = pd.Series(np.nan, index=target_index, dtype=float)

        out["deriv_funding_rate"] = f_rate
        f_mean = f_rate.rolling(96 * 7, min_periods=96).mean()
        f_std = f_rate.rolling(96 * 7, min_periods=96).std()
        out["deriv_funding_zscore"] = (f_rate - f_mean) / (f_std + 1e-8)
        out["deriv_funding_change_24h"] = f_rate - f_rate.shift(96)

        # Derivados ampliados opcionales (Fase 3.4) — estrictamente causal sin .bfill()
        if df_derivatives is not None and not df_derivatives.empty:
            if "open_interest" in df_derivatives.columns:
                oi = cls._align_causal_series(df_derivatives["open_interest"], target_index)
                out["deriv_oi_change_24h"] = oi.pct_change(96, fill_method=None)
                oi_mean = oi.rolling(96 * 7, min_periods=96).mean()
                oi_std = oi.rolling(96 * 7, min_periods=96).std()
                out["deriv_oi_zscore"] = (oi - oi_mean) / (oi_std + 1e-8)

            if "basis" in df_derivatives.columns:
                out["deriv_basis"] = cls._align_causal_series(
                    df_derivatives["basis"], target_index
                )

            if "premium" in df_derivatives.columns:
                out["deriv_premium"] = cls._align_causal_series(
                    df_derivatives["premium"], target_index
                )

            if "liquidations" in df_derivatives.columns:
                out["deriv_liquidations"] = cls._align_causal_series(
                    df_derivatives["liquidations"], target_index
                )

        return out

