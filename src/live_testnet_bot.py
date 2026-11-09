import os
import time
import ccxt
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class BinanceFuturesTestnetBot:
    def __init__(self, model_path: str = "models/quant_brain.joblib"):
        bundle = joblib.load(model_path)
        self.brain = bundle["brain"]
        self.pipe = bundle["pipeline"]

        # 1. Cliente Público en MAINNET para leer precios, volumen taker y funding REALES
        self.market_data_client = ccxt.binanceusdm({"enableRateLimit": True})

        # 2. Cliente Privado en TESTNET para ejecutar órdenes con dinero de prueba
        self.execution_client = ccxt.binanceusdm({
            "apiKey": os.getenv("BINANCE_TESTNET_API_KEY"),
            "secret": os.getenv("BINANCE_TESTNET_SECRET"),
            "enableRateLimit": True,
        })
        # Activamos el modo Sandbox (Apunta automáticamente a testnet.binancefuture.com)
        self.execution_client.set_sandbox_mode(True)

    def fetch_live_dataframe(self, symbol: str, limit: int = 800) -> pd.DataFrame:
        """
        Descarga las últimas 800 velas de 15m desde el endpoint nativo de Binance Futures
        incluyendo el volumen Taker Buy para el bloque Smart Money.
        """
        raw_klines = self.market_data_client.fapiPublicGetKlines({
            "symbol": symbol.replace("/", ""),
            "interval": "15m",
            "limit": limit
        })
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(raw_klines, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        
        num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        # Descartamos la última vela en formación para usar solo velas cerradas
        return df[num_cols].iloc[:-1]

    def fetch_live_funding(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        raw_funding = self.market_data_client.fapiPublicGetFundingRate({
            "symbol": symbol,
            "limit": 200
        })
        df = pd.DataFrame(raw_funding)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["funding_rate"]]

    def evaluate_and_execute(self, ev_threshold: float = 0.15):
        print("\n--- Evaluando Cierre de Vela 15m en Mercado Real ---")
        df_btc = self.fetch_live_dataframe("BTCUSDT", limit=800)
        df_eth = self.fetch_live_dataframe("ETHUSDT", limit=800)
        df_funding = self.fetch_live_funding("BTCUSDT")

        features = self.pipe.build_features(df_btc, df_eth, df_funding).dropna()
        last_row = features.iloc[[-1]]
        last_close = float(df_btc["close"].iloc[-1])
        last_atr_pct = float(last_row["atr_pct"].iloc[0])

        # Inferencia en el Probability Engine y cálculo de EV
        p_win = self.brain.predict_calibrated_proba(last_row)
        ev_decision = self.brain.compute_ev_and_kelly(p_win, last_row["atr_pct"].values).iloc[0]

        print(f"Timestamp Vela : {last_row.index[0]}")
        print(f"Precio BTC     : {last_close:.2f} USDT | ATR%: {last_atr_pct*100:.3f}%")
        print(f"Contexto HTF   : {'BULLISH' if last_row['ctx_btc_bull_regime'].iloc[0] == 1 else 'BEARISH'}")
        print(f"P(TP antes SL) : {ev_decision['p_tp_before_sl']*100:.2f}% (Cuota Neta b={ev_decision['b_net_odds']:.2f}R)")
        print(f"Expected Value : {ev_decision['ev_in_r']:+.3f} R")

        if ev_decision["ev_in_r"] < ev_threshold:
            print("Decisión       : PASAR (EV insuficiente según criterio de Bill Benter).")
            return

        # Si el EV supera el umbral, verificamos que no haya ya una posición abierta en Testnet
        positions = self.execution_client.fetch_positions(["BTC/USDT:USDT"])
        open_contracts = sum(abs(float(p.get("contracts", 0))) for p in positions)
        if open_contracts > 0:
            print("Decisión       : Posición ya activa en Testnet. Se omite nueva entrada.")
            return

        # Dimensionamiento de riesgo en base al balance disponible en Testnet
        balance = self.execution_client.fetch_balance()
        usdt_free = float(balance["USDT"]["free"])
        risk_usdt = usdt_free * float(ev_decision["risk_fraction_kelly"])
        
        sl_distance_usdt = last_close * last_atr_pct * self.brain.sl_mult
        tp_distance_usdt = last_close * last_atr_pct * self.brain.tp_mult
        
        # Cantidad de BTC a comprar para que si toca el SL solo perdamos `risk_usdt`
        qty_btc = round(max(risk_usdt / sl_distance_usdt, 0.002), 3)
        sl_price = round(last_close - sl_distance_usdt, 1)
        tp_price = round(last_close + tp_distance_usdt, 1)

        print(f"EJECUTANDO LONG EN TESTNET: {qty_btc} BTC | TP: {tp_price} | SL: {sl_price}")
        
        # Orden de entrada a mercado + Barreras nativas en Binance Futures Testnet
        self.execution_client.create_order("BTC/USDT:USDT", "market", "buy", qty_btc)
        self.execution_client.create_order(
            "BTC/USDT:USDT", "STOP_MARKET", "sell", qty_btc, None, 
            {"stopPrice": sl_price, "closePosition": True}
        )
        self.execution_client.create_order(
            "BTC/USDT:USDT", "TAKE_PROFIT_MARKET", "sell", qty_btc, None, 
            {"stopPrice": tp_price, "closePosition": True}
        )
        print("Órdenes enviadas y confirmadas en Binance Futures Testnet.")


if __name__ == "__main__":
    bot = BinanceFuturesTestnetBot()
    while True:
        try:
            bot.evaluate_and_execute(ev_threshold=0.12)
        except Exception as e:
            print(f"[ERROR EN CICLO]: {e}")
        # Espera hasta el próximo bloque de 15 minutos
        time.sleep(900)