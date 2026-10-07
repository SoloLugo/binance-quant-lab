import os
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = "/app"
CANONICAL_DIR = os.path.join(BASE_DIR, "data", "canonical")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def calculate_features(df):
    print("    -> Calculando Familias A, B y C (Tech, SMC, Derivados)...")
    # Familia A: Technicals
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['dist_ema_20_atr'] = (df['close'] - df['ema_20']) / df['atr_14']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))

    # Familia B: Smart Money
    df['is_swing_low'] = (df['low'].shift(2) < df['low'].shift(1)) & (df['low'].shift(2) < df['low']) & (df['low'].shift(2) < df['low'].shift(3)) & (df['low'].shift(2) < df['low'].shift(4))
    df['last_swing_low'] = np.where(df['is_swing_low'], df['low'].shift(2), np.nan)
    df['last_swing_low'] = df['last_swing_low'].ffill()
    df['liquidity_sweep_bullish'] = ((df['low'] < df['last_swing_low']) & (df['close'] > df['last_swing_low'])).astype(int)

    # Familia C: Derivados
    if 'funding_rate' in df.columns:
        df['funding_rate_mean'] = df['funding_rate'].rolling(window=672, min_periods=1).mean()
        df['funding_rate_std'] = df['funding_rate'].rolling(window=672, min_periods=1).std()
        df['funding_zscore'] = np.where(df['funding_rate_std'] > 0, (df['funding_rate'] - df['funding_rate_mean']) / df['funding_rate_std'], 0)
    else:
        df['funding_zscore'] = 0.0

    print("    -> Calculando Familia D: CONTEXT ENGINE (Regímenes de Mercado)...")
    # ---------------------------------------------------------
    # FAMILIA D: CONTEXTO MACRO Y REGÍMENES
    # ---------------------------------------------------------
    # 1. Régimen de Tendencia (Trend Regime) - EMA de 200 periodos
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Pendiente de la EMA 200 para saber si el mercado está plano o direccional
    df['ema_200_slope'] = (df['ema_200'] - df['ema_200'].shift(10)) / df['ema_200'].shift(10)
    
    # Clasificación cruda del régimen
    df['regime_bull'] = (df['close'] > df['ema_200']).astype(int)
    df['regime_bear'] = (df['close'] < df['ema_200']).astype(int)

    # 2. Régimen de Volatilidad (Volatility Regime) - Bollinger Band Width (Ancho de Banda)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['bb_width'] = (df['std_20'] * 4) / df['sma_20']
    
    # Si el ancho de banda actual es mayor al promedio de la última semana (672 velas de 15m)
    df['bb_width_mean_1w'] = df['bb_width'].rolling(window=672, min_periods=1).mean()
    df['regime_high_vol'] = (df['bb_width'] > df['bb_width_mean_1w']).astype(int)

    # Limpieza
    cols_to_drop = ['prev_close', 'tr', 'funding_rate_mean', 'funding_rate_std', 'sma_20', 'std_20', 'bb_width_mean_1w']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    return df

def build_features_for_symbol(symbol):
    print(f"\n[*] Construyendo Features con Context Engine para {symbol}...")
    in_file = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
    if not os.path.exists(in_file): return
        
    df = pd.read_parquet(in_file)
    df_features = calculate_features(df)
    
    out_dir = os.path.join(FEATURES_DIR, f"symbol={symbol}")
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df_features), os.path.join(out_dir, "features.parquet"), compression='snappy')
    print(f"  [OK] Features listas: {len(df_features.columns)} variables | {len(df_features):,} filas.")

def main():
    config = load_config()
    symbols = config['ingestion']['symbols']
    for symbol in symbols: build_features_for_symbol(symbol)

if __name__ == "__main__":
    main()