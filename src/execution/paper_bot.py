import os
import sys
import time
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Importamos la logica exacta con la que entrenamos para evitar discrepancias
sys.path.append("/app")
from src.features.builder import calculate_features

BASE_DIR = "/app"
MODELS_DIR = os.path.join(BASE_DIR, "models")
SYMBOL = "BTCUSDT"

# Las mismas 11 columnas exactas que vimos en tu log de exportacion
FEATURES_COLS = [
    'volume', 'open_interest', 'funding_rate', 'atr_14', 
    'dist_ema_20_atr', 'rsi_14', 'funding_zscore', 
    'oi_change_pct', 'oi_zscore', 'ema_200_slope', 'bb_width'
]

def get_binance_data():
    # 1. Velas de 15m (Descargamos 300 para calentar la EMA 200 y Bandas de Bollinger)
    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval=15m&limit=300"
    k_data = requests.get(klines_url).json()
    
    df = pd.DataFrame(k_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # 2. Funding Rate Actual
    fund_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={SYMBOL}"
    f_data = requests.get(fund_url).json()
    current_funding = float(f_data.get('lastFundingRate', 0))
    
    # 3. Open Interest Actual
    oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL}"
    oi_data = requests.get(oi_url).json()
    current_oi = float(oi_data.get('openInterest', 0))
    
    # Asignamos el valor actual al bloque de warmup para calcular Z-scores
    df['funding_rate'] = current_funding
    df['open_interest'] = current_oi
    
    return df

def run_paper_bot():
    current_time = datetime.now().strftime('%H:%M:%S')
    print(f"\n[{current_time}] Despertando Motor REST para {SYMBOL}...")
    
    scaler_path = os.path.join(MODELS_DIR, f"{SYMBOL}_scaler_v1.joblib")
    model_path = os.path.join(MODELS_DIR, f"{SYMBOL}_model_v1.joblib")
    
    if not os.path.exists(scaler_path):
        print("[!] No se encontro el modelo. Ejecuta train_final.py primero.")
        return
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    # 1. Obtener datos en vivo de Binance
    df_raw = get_binance_data()
    
    # 2. Reconstruir la realidad matematica original
    # Suprimimos los prints del calculate_features temporalmente
    sys.stdout = open(os.devnull, 'w')
    df_features = calculate_features(df_raw.copy())
    sys.stdout = sys.__stdout__
    
    if df_features.empty:
        print("[!] Error construyendo variables de calentamiento.")
        return
        
    # Extraer la foto matematica del momento exacto actual
    last_row = df_features.iloc[-1:]
    current_context = last_row[['regime_bull', 'regime_bear', 'regime_high_vol']].iloc[0]
    
    # 3. Escalar y Predecir
    X_live = last_row[FEATURES_COLS]
    X_live_scaled = scaler.transform(X_live)
    prob_long = model.predict_proba(X_live_scaled)[0, 1]
    
    # 4. Logica de Ejecucion y Context Engine
    umbral = 0.35
    is_bull = current_context['regime_bull'] == 1
    
    print("-" * 55)
    print(f"PROBABILIDAD PREDICTIVA : {prob_long:.2%}")
    print(f"CONTEXTO MACRO          : {'ALCISTA' if is_bull else 'BAJISTA'}")
    print(f"VOLATILIDAD (BB WIDTH)  : {last_row['bb_width'].iloc[0]:.4f}")
    print("-" * 55)
    
    if prob_long >= umbral and is_bull:
        print("[SENAL DISPARADA] Condiciones optimas. Ejecutando LONG simulado.")
    elif prob_long >= umbral and not is_bull:
        print("[SENAL BLOQUEADA] Probabilidad superada, pero el Context Engine lo rechazo (Mercado Bajista).")
    else:
        print("[STANDBY] Probabilidad insuficiente. Sin Edge detectado.")
    print("-" * 55)

if __name__ == "__main__":
    run_paper_bot()