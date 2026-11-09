from typing import List, Tuple
import numpy as np
import pandas as pd


# Calendario histórico de reuniones FOMC (2024 - 2026) para el Context Engine Macro
FOMC_DATES = [
    "2024-01-31 19:00:00", "2024-03-20 18:00:00", "2024-05-01 18:00:00", "2024-06-12 18:00:00",
    "2024-07-31 18:00:00", "2024-09-18 18:00:00", "2024-11-07 19:00:00", "2024-12-18 19:00:00",
    "2025-01-29 19:00:00", "2025-03-19 18:00:00", "2025-05-07 18:00:00", "2025-06-18 18:00:00",
    "2025-07-30 18:00:00", "2025-09-17 18:00:00", "2025-10-29 18:00:00", "2025-12-10 19:00:00",
    "2026-01-28 19:00:00", "2026-03-18 18:00:00", "2026-04-29 18:00:00", "2026-06-17 18:00:00",
    "2026-07-29 18:00:00", "2026-09-16 18:00:00", "2026-10-28 18:00:00", "2026-12-09 19:00:00",
]


class QuantFeaturePipeline:
    """
    Construye las 4 capas de información:
    1. Technical Analysis (ta_*)
    2. Smart Money / Microestructura Orderflow (sm_*)
    3. Derivatives / Funding (deriv_*)
    4. Context Engine / Macro / Ciclo BTC-ETH (ctx_*)
    """
    def __init__(self, tp_mult: float = 2.0, sl_mult: float = 1.0, horizon_bars: int = 24):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.horizon_bars = horizon_bars
        self.fomc_timestamps = pd.to_datetime(FOMC_DATES, utc=True)

    def build_features(
        self, 
        df_btc: pd.DataFrame, 
        df_eth: pd.DataFrame, 
        df_funding: pd.DataFrame
    ) -> pd.DataFrame:
        df = df_btc.copy()
        feat = pd.DataFrame(index=df.index)

        # =========================================================
        # CAPA 1: TECHNICAL ANALYSIS (ta_*)
        # =========================================================
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR de 14 periodos (base para volatilidad y barreras TP/SL)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        feat["atr_pct"] = atr_14 / close  # Guardamos para el cálculo de EV y barreras

        # Retornos normalizados por volatilidad en múltiples horizontes
        feat["ta_ret_1"] = close.pct_change(1) / (feat["atr_pct"] + 1e-8)
        feat["ta_ret_4"] = close.pct_change(4) / (feat["atr_pct"] + 1e-8)
        feat["ta_ret_16"] = close.pct_change(16) / (feat["atr_pct"] + 1e-8)

        # Distancia a medias móviles exponenciales normalizada por ATR
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        feat["ta_dist_ema20"] = (close - ema_20) / (atr_14 + 1e-8)
        feat["ta_dist_ema50"] = (close - ema_50) / (atr_14 + 1e-8)

        # RSI de 14 periodos
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        feat["ta_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Compresión de Bollinger (ancho de banda)
        bb_std = close.rolling(20).std()
        feat["ta_bb_width"] = (4.0 * bb_std) / ema_20

        # =========================================================
        # CAPA 2: SMART MONEY / ORDERFLOW (sm_*)
        # =========================================================
        # En los CSVs de Binance, taker_buy_volume nos permite medir la agresión real a mercado
        vol = df["volume"].replace(0, np.nan)
        taker_buy = df["taker_buy_volume"]
        taker_sell = vol - taker_buy
        
        # Delta de volumen por vela (-1 = venta institucional agresiva, +1 = compra agresiva)
        order_flow_imbalance = (taker_buy - taker_sell) / (vol + 1e-8)
        feat["sm_ofi_1"] = order_flow_imbalance
        feat["sm_ofi_cum_4"] = order_flow_imbalance.rolling(4).mean()
        feat["sm_ofi_cum_16"] = order_flow_imbalance.rolling(16).mean()

        # Tamaño promedio por transacción (Detecta si operan ballenas o retail)
        avg_trade_size = df["quote_volume"] / (df["count"].replace(0, np.nan) + 1e-8)
        feat["sm_whale_trade_zscore"] = (
            avg_trade_size - avg_trade_size.rolling(96).mean()
        ) / (avg_trade_size.rolling(96).std() + 1e-8)

        # Absorción / Mechas (Rechazo de liquidez en máximos y mínimos)
        candle_range = (high - low) + 1e-8
        feat["sm_lower_wick_ratio"] = (np.minimum(df["open"], close) - low) / candle_range
        feat["sm_upper_wick_ratio"] = (high - np.maximum(df["open"], close)) / candle_range

        # =========================================================
        # CAPA 3: DERIVATIVES (deriv_*)
        # =========================================================
        if not df_funding.empty:
            f_rate = df_funding["funding_rate"].reindex(feat.index, method="ffill").fillna(0.0)
        else:
            f_rate = pd.Series(0.0, index=feat.index)

        feat["deriv_funding_rate"] = f_rate
        # Z-Score de 7 días (96 velas * 7) para identificar posicionamiento saturado (crowded trades)
        f_mean = f_rate.rolling(96 * 7).mean()
        f_std = f_rate.rolling(96 * 7).std()
        feat["deriv_funding_zscore"] = (f_rate - f_mean) / (f_std + 1e-8)
        feat["deriv_funding_change_24h"] = f_rate - f_rate.shift(96)

        # =========================================================
        # CAPA 4: CONTEXT ENGINE / MACRO / REGIME (ctx_*)
        # =========================================================
        # 1. Régimen Macro de Tendencia en HTF (1D = 96 velas, 1W = 672 velas)
        ema_1d = close.ewm(span=96, adjust=False).mean()
        ema_5d = close.ewm(span=96 * 5, adjust=False).mean()
        feat["ctx_btc_htf_trend"] = (ema_1d - ema_5d) / (ema_5d + 1e-8)
        feat["ctx_btc_bull_regime"] = (ema_1d > ema_5d).astype(int)

        # 2. Régimen de Volatilidad (Expansión vs Compresión en ventana de 15 días)
        vol_24h = close.pct_change().rolling(96).std()
        feat["ctx_vol_regime_pct"] = vol_24h.rolling(96 * 15).rank(pct=True)

        # 3. Flujo BTC -> ETH -> ALTs (Fuerza relativa y Correlación)
        if not df_eth.empty:
            eth_close = df_eth["close"].reindex(feat.index, method="ffill")
            eth_btc = eth_close / close
            feat["ctx_eth_btc_rel_24h"] = eth_btc.pct_change(96)
            feat["ctx_eth_btc_rel_7d"] = eth_btc.pct_change(96 * 7)
            feat["ctx_btc_eth_corr_24h"] = close.pct_change().rolling(96).corr(eth_close.pct_change())
        else:
            feat["ctx_eth_btc_rel_24h"] = 0.0
            feat["ctx_eth_btc_rel_7d"] = 0.0
            feat["ctx_btc_eth_corr_24h"] = 0.0

        # 4. Catalizador Macro (Ventana FOMC 24h antes)
        fomc_flag = pd.Series(0, index=feat.index)
        idx_s = feat.index.to_series()
        for dt in self.fomc_timestamps:
            mask = (idx_s >= (dt - pd.Timedelta(hours=24))) & (idx_s <= (dt + pd.Timedelta(hours=2)))
            fomc_flag.loc[mask] = 1
        feat["ctx_fomc_event_24h"] = fomc_flag

        return feat

    def create_triple_barrier_labels(self, df_btc: pd.DataFrame, atr_pct: pd.Series, direction: str = "LONG") -> pd.Series:
        """
        Calcula la variable objetivo exacta de tu arquitectura: P(TP before SL).
        Retorna 1 si el precio toca el Take Profit antes que el Stop Loss dentro de `horizon_bars`.
        Retorna 0 si toca el Stop Loss primero o si expira el tiempo sin tocar el TP.
        """
        closes = df_btc["close"].values
        highs = df_btc["high"].values
        lows = df_btc["low"].values
        atrs = (atr_pct * df_btc["close"]).values

        n = len(closes)
        labels = np.zeros(n, dtype=int)

        for i in range(n - self.horizon_bars):
            if np.isnan(atrs[i]) or atrs[i] <= 0:
                continue

            entry = closes[i]
            if direction == "LONG":
                tp_price = entry + (self.tp_mult * atrs[i])
                sl_price = entry - (self.sl_mult * atrs[i])

                for j in range(1, self.horizon_bars + 1):
                    # Si en una misma vela toca ambos, asumimos conservadoramente que tocó SL primero
                    if lows[i + j] <= sl_price:
                        labels[i] = 0
                        break
                    if highs[i + j] >= tp_price:
                        labels[i] = 1
                        break
            else:
                tp_price = entry - (self.tp_mult * atrs[i])
                sl_price = entry + (self.sl_mult * atrs[i])

                for j in range(1, self.horizon_bars + 1):
                    if highs[i + j] >= sl_price:
                        labels[i] = 0
                        break
                    if lows[i + j] <= tp_price:
                        labels[i] = 1
                        break

        return pd.Series(labels, index=df_btc.index, name="target_tp_before_sl")