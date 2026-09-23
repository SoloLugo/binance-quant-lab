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
    """RSI estricto sin look-ahead bias"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(df):
    """
    Construye las features estacionarias V0.1.
    Toda métrica aquí solo usa datos en el tiempo t o anteriores.
    """
    X = pd.DataFrame(index=df.index)
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    vol = df['volume']

    # 1. Retornos Logarítmicos (Multi-escala)
    # log(Pt / Pt-1) es el estándar matemático para series financieras
    X['log_ret_15m'] = np.log(close / close.shift(1))
    X['log_ret_1h'] = np.log(close / close.shift(4))
    X['log_ret_4h'] = np.log(close / close.shift(16))

    # 2. Medias Móviles (Distancias porcentuales, no precios absolutos)
    for span in [20, 50, 200]:
        ema = close.ewm(span=span, adjust=False).mean()
        X[f'dist_ema_{span}'] = (close - ema) / ema

    # 3. Momento (RSI)
    X['rsi_14'] = calc_rsi(close, 14)

    # 4. Volatilidad (ATR Relativo)
    # True Range = max(H-L, abs(H-Cp), abs(L-Cp))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(14).mean()
    X['atr_rel_14'] = atr_14 / close  # Adimensional

    # 5. Micro-Estructura de la vela actual
    candle_range = (high - low).replace(0, 1e-8)  # Evitar división por cero
    X['body_ratio'] = (close - open_p).abs() / candle_range
    X['upper_wick_ratio'] = (high - np.maximum(open_p, close)) / candle_range
    X['lower_wick_ratio'] = (np.minimum(open_p, close) - low) / candle_range

    # 6. Volumen Relativo
    vol_sma_50 = vol.rolling(50).mean().replace(0, 1e-8)
    X['vol_rel_50'] = vol / vol_sma_50

    return X

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    print("\n🧠 Iniciando Feature Engineering (Matriz X)...")
    
    for symbol in symbols:
        parquet_path = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
        
        if not os.path.exists(parquet_path):
            continue
            
        print(f"[*] Construyendo features estacionarias para {symbol}...")
        df = pd.read_parquet(parquet_path)
        
        # Construir matriz X
        X = build_features(df)
        
        # Eliminar filas con NaNs iniciales (causadas por los periodos de cálculo como EMA200)
        X.dropna(inplace=True)
        
        # Guardar en disco
        out_dir = os.path.join(FEATURES_DIR, f"symbol={symbol}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "features.parquet")
        
        table = pa.Table.from_pandas(X)
        pq.write_table(table, out_file, compression='snappy')
        
        print(f"  [OK] Matriz X generada con {len(X.columns)} columnas.")
        print(f"  [OK] Muestras útiles tras calentar indicadores: {len(X):,}")

if __name__ == "__main__":
    main()