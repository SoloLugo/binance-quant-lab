import numpy as np
import pandas as pd


# Calendario histórico y proyectado de reuniones FOMC (2024 - 2027) para el Context Engine Macro
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


class QuantFeaturePipeline:
    """
    Construye las 4 capas de información estacionarias y sin Look-Ahead Bias:
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
        atr_14 = tr.rolling(14, min_periods=14).mean()
        feat["atr_pct"] = (atr_14 / close).clip(lower=1e-4)

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
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / (loss + 1e-8)
        feat["ta_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Compresión de Bollinger (ancho de banda)
        bb_std = close.rolling(20, min_periods=20).std()
        feat["ta_bb_width"] = (4.0 * bb_std) / ema_20

        # =========================================================
        # CAPA 2: SMART MONEY / ORDERFLOW (sm_*)
        # =========================================================
        vol = df["volume"].replace(0, np.nan)
        taker_buy = df["taker_buy_volume"]
        taker_sell = vol - taker_buy

        # Delta de volumen por vela (-1 = venta institucional agresiva, +1 = compra agresiva)
        order_flow_imbalance = (taker_buy - taker_sell) / (vol + 1e-8)
        feat["sm_ofi_1"] = order_flow_imbalance
        feat["sm_ofi_cum_4"] = order_flow_imbalance.rolling(4, min_periods=1).mean()
        feat["sm_ofi_cum_16"] = order_flow_imbalance.rolling(16, min_periods=4).mean()

        # Tamaño promedio por transacción (Detecta si operan ballenas o retail)
        avg_trade_size = df["quote_volume"] / (df["count"].replace(0, np.nan) + 1e-8)
        feat["sm_whale_trade_zscore"] = (
            avg_trade_size - avg_trade_size.rolling(96, min_periods=24).mean()
        ) / (avg_trade_size.rolling(96, min_periods=24).std() + 1e-8)

        # Absorción / Mechas (Rechazo de liquidez en máximos y mínimos)
        candle_range = (high - low) + 1e-8
        feat["sm_lower_wick_ratio"] = (np.minimum(df["open"], close) - low) / candle_range
        feat["sm_upper_wick_ratio"] = (high - np.maximum(df["open"], close)) / candle_range

        # =========================================================
        # CAPA 3: DERIVATIVES (deriv_*)
        # =========================================================
        if not df_funding.empty:
            f_series = df_funding["funding_rate"].copy()
            f_series.index = pd.to_datetime(f_series.index, utc=True).floor("1min")
            f_series = f_series[~f_series.index.duplicated(keep="last")].sort_index()
            f_rate = f_series.reindex(feat.index, method="ffill").bfill().fillna(0.0)
        else:
            f_rate = pd.Series(0.0, index=feat.index)

        feat["deriv_funding_rate"] = f_rate
        # Z-Score de 7 días con min_periods=96 (1 día) para garantizar disponibilidad en inferencia live
        f_mean = f_rate.rolling(96 * 7, min_periods=96).mean()
        f_std = f_rate.rolling(96 * 7, min_periods=96).std()
        feat["deriv_funding_zscore"] = ((f_rate - f_mean) / (f_std + 1e-8)).fillna(0.0)
        feat["deriv_funding_change_24h"] = (f_rate - f_rate.shift(96)).fillna(0.0)

        # =========================================================
        # CAPA 4: CONTEXT ENGINE / MACRO / REGIME (ctx_*)
        # Diseñada con indicadores estacionarios de horizonte corto/medio para evitar
        # que los árboles de decisión memoricen semanas calendario específicas.
        # =========================================================
        # 1. Régimen Macro de Tendencia en HTF (1D = 96 velas vs 5D = 480 velas)
        ema_1d = close.ewm(span=96, adjust=False).mean()
        ema_5d = close.ewm(span=96 * 5, adjust=False).mean()
        feat["ctx_btc_bull_regime"] = (ema_1d > ema_5d).astype(int)

        # 2. Régimen de Volatilidad (Percentil rodante de 15 días, compatible con 1500 velas en vivo)
        vol_24h = close.pct_change().rolling(96, min_periods=24).std()
        feat["ctx_vol_regime_pct"] = vol_24h.rolling(96 * 15, min_periods=96).rank(pct=True)

        # 3. Flujo BTC -> ETH -> ALTs (Impulso relativo de corto plazo normalizado por ATR)
        if not df_eth.empty:
            eth_close = df_eth["close"].reindex(feat.index, method="ffill").bfill()
            eth_btc = eth_close / (close + 1e-8)
            feat["ctx_eth_btc_rel_24h"] = (eth_btc.pct_change(4) / (feat["atr_pct"] + 1e-8)).clip(-3.0, 3.0)
        else:
            feat["ctx_eth_btc_rel_24h"] = 0.0

        # 4. Catalizador Macro (Ventana FOMC 24h antes hasta 2h después)
        fomc_flag = pd.Series(0, index=feat.index)
        idx_s = feat.index.to_series()
        for dt in self.fomc_timestamps:
            mask = (idx_s >= (dt - pd.Timedelta(hours=24))) & (idx_s <= (dt + pd.Timedelta(hours=2)))
            fomc_flag.loc[mask] = 1
        feat["ctx_fomc_event_24h"] = fomc_flag

        return feat

    def compute_triple_barrier_outcomes(
        self,
        df_btc: pd.DataFrame,
        atr_pct: pd.Series,
        direction: str = "LONG"
    ) -> pd.DataFrame:
        """
        Calcula de forma exacta las 3 barreras (Take Profit, Stop Loss y Barrera Vertical de Tiempo):
          - target_tp_before_sl : 1.0 si toca TP antes de SL dentro de horizon_bars, 0.0 si no (NaN al final).
          - realized_r_gross    : Retorno bruto en unidades de 1R (+tp_mult, -1.0, o mark-to-market al expirar).
          - exit_bars           : Número de velas transcurridas hasta tocar la barrera (1 .. horizon_bars).
          - exit_type           : 'TP', 'SL' o 'TIME'.
        """
        closes = df_btc["close"].values
        highs = df_btc["high"].values
        lows = df_btc["low"].values
        atrs = (atr_pct * df_btc["close"]).values

        n = len(closes)
        labels = np.full(n, np.nan, dtype=float)
        realized_r = np.full(n, np.nan, dtype=float)
        exit_bars = np.full(n, self.horizon_bars, dtype=int)
        exit_types = np.full(n, "NONE", dtype=object)

        direction_upper = direction.upper()

        for i in range(n - self.horizon_bars):
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            entry = closes[i]
            sl_dist = self.sl_mult * atr_val
            tp_dist = self.tp_mult * atr_val

            if direction_upper == "LONG":
                tp_price = entry + tp_dist
                sl_price = entry - sl_dist
                hit = False
                for j in range(1, self.horizon_bars + 1):
                    # Conservador: si una misma vela toca SL y TP, asumimos SL primero
                    if lows[i + j] <= sl_price:
                        labels[i] = 0.0
                        realized_r[i] = -1.0
                        exit_bars[i] = j
                        exit_types[i] = "SL"
                        hit = True
                        break
                    if highs[i + j] >= tp_price:
                        labels[i] = 1.0
                        realized_r[i] = self.tp_mult / self.sl_mult
                        exit_bars[i] = j
                        exit_types[i] = "TP"
                        hit = True
                        break
                if not hit:
                    labels[i] = 0.0
                    time_ret_r = (closes[i + self.horizon_bars] - entry) / sl_dist
                    realized_r[i] = float(np.clip(time_ret_r, -1.0, self.tp_mult / self.sl_mult))
                    exit_bars[i] = self.horizon_bars
                    exit_types[i] = "TIME"
            else:
                tp_price = entry - tp_dist
                sl_price = entry + sl_dist
                hit = False
                for j in range(1, self.horizon_bars + 1):
                    if highs[i + j] >= sl_price:
                        labels[i] = 0.0
                        realized_r[i] = -1.0
                        exit_bars[i] = j
                        exit_types[i] = "SL"
                        hit = True
                        break
                    if lows[i + j] <= tp_price:
                        labels[i] = 1.0
                        realized_r[i] = self.tp_mult / self.sl_mult
                        exit_bars[i] = j
                        exit_types[i] = "TP"
                        hit = True
                        break
                if not hit:
                    labels[i] = 0.0
                    time_ret_r = (entry - closes[i + self.horizon_bars]) / sl_dist
                    realized_r[i] = float(np.clip(time_ret_r, -1.0, self.tp_mult / self.sl_mult))
                    exit_bars[i] = self.horizon_bars
                    exit_types[i] = "TIME"

        return pd.DataFrame({
            "target_tp_before_sl": labels,
            "realized_r_gross": realized_r,
            "exit_bars": exit_bars,
            "exit_type": exit_types
        }, index=df_btc.index)

    def create_triple_barrier_labels(
        self,
        df_btc: pd.DataFrame,
        atr_pct: pd.Series,
        direction: str = "LONG"
    ) -> pd.Series:
        """
        Calcula P(TP before SL) dentro de `horizon_bars`.
        Las últimas `horizon_bars` filas se marcan como NaN para evitar falsos negativos.
        """
        outcomes = self.compute_triple_barrier_outcomes(df_btc, atr_pct, direction=direction)
        return outcomes["target_tp_before_sl"]
