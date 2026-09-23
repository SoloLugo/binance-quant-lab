import os
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = "/app"
CANONICAL_DIR = os.path.join(BASE_DIR, "data", "canonical")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(df):
    """Construye características macro y micro (V0.2)"""
    X = pd.DataFrame(index=df.index)
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    vol = df['volume']

    # --- 1. CLÁSICOS (Momentum y Retornos) ---
    X['log_ret_15m'] = np.log(close / close.shift(1))
    X['log_ret_1h'] = np.log(close / close.shift(4))
    X['log_ret_4h'] = np.log(close / close.shift(16))
    X['rsi_14'] = calc_rsi(close, 14)

    # --- 2. DISTANCIAS Y ALINEACIÓN DE TENDENCIA (Trend Regime) ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    X['dist_ema_20'] = (close - ema20) / ema20
    X['dist_ema_50'] = (close - ema50) / ema50
    X['dist_ema_200'] = (close - ema200) / ema200
    
    # Cuantificador de tendencia: +1 (Alcista total), -1 (Bajista total), 0 (Rango/Cruce)
    trend_up = (ema20 > ema50) & (ema50 > ema200)
    trend_down = (ema20 < ema50) & (ema50 < ema200)
    X['trend_alignment'] = np.where(trend_up, 1.0, np.where(trend_down, -1.0, 0.0))

    # --- 3. REGÍMENES DE VOLATILIDAD (ATR Expansion & Bollinger) ---
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(14).mean()
    atr_50 = tr.rolling(50).mean()
    
    X['atr_rel_14'] = atr_14 / close
    X['volatility_regime'] = atr_14 / atr_50  # >1 indica expansión de volatilidad
    
    # Bollinger Band Width (Compresión del precio)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    X['bb_width'] = (std_20 * 2) / sma_20

    # --- 4. ESTADÍSTICA DE COLA (Skewness y Kurtosis en ventana de 5 horas) ---
    window = 20 # 20 velas de 15m = 5 horas
    X['ret_skewness'] = X['log_ret_15m'].rolling(window).skew()
    X['ret_kurtosis'] = X['log_ret_15m'].rolling(window).kurt()

    # --- 5. MICRO-ESTRUCTURA Y VOLUMEN ---
    candle_range = (high - low).replace(0, 1e-8)
    X['body_ratio'] = (close - open_p).abs() / candle_range
    X['upper_wick_ratio'] = (high - np.maximum(open_p, close)) / candle_range
    
    vol_sma_50 = vol.rolling(50).mean().replace(0, 1e-8)
    X['vol_rel_50'] = vol / vol_sma_50

    return X

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    print("\n🧠 Iniciando Feature Engineering Avanzado V0.2...")
    
    for symbol in symbols:
        parquet_path = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
        if not os.path.exists(parquet_path): continue
            
        print(f"[*] Construyendo features para {symbol}...")
        df = pd.read_parquet(parquet_path)
        
        X = build_features(df)
        X.dropna(inplace=True) # Elimina los primeros NaNs causados por ventanas largas (200 periodos)
        
        out_dir = os.path.join(FEATURES_DIR, f"symbol={symbol}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "features.parquet")
        
        table = pa.Table.from_pandas(X)
        pq.write_table(table, out_file, compression='snappy')
        print(f"  [OK] Matriz X generada con {len(X.columns)} columnas.")

if __name__ == "__main__":
    main()