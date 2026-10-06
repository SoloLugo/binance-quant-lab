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
    print("    -> Calculando Familia A: Análisis Técnico...")
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['dist_ema_20_atr'] = (df['close'] - df['ema_20']) / df['atr_14']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    print("    -> Calculando Familia B: Smart Money & Market Structure...")
    df['is_swing_high'] = (df['high'].shift(2) > df['high'].shift(1)) & \
                          (df['high'].shift(2) > df['high']) & \
                          (df['high'].shift(2) > df['high'].shift(3)) & \
                          (df['high'].shift(2) > df['high'].shift(4))
                          
    df['is_swing_low'] = (df['low'].shift(2) < df['low'].shift(1)) & \
                         (df['low'].shift(2) < df['low']) & \
                         (df['low'].shift(2) < df['low'].shift(3)) & \
                         (df['low'].shift(2) < df['low'].shift(4))

    df['last_swing_high'] = np.where(df['is_swing_high'], df['high'].shift(2), np.nan)
    df['last_swing_high'] = df['last_swing_high'].ffill()
    
    df['last_swing_low'] = np.where(df['is_swing_low'], df['low'].shift(2), np.nan)
    df['last_swing_low'] = df['last_swing_low'].ffill()

    df['liquidity_sweep_bullish'] = ((df['low'] < df['last_swing_low']) & 
                                     (df['close'] > df['last_swing_low'])).astype(int)
                                     
    df['liquidity_sweep_bearish'] = ((df['high'] > df['last_swing_high']) & 
                                     (df['close'] < df['last_swing_high'])).astype(int)

    df['strong_bullish_impulse'] = (df['close'] - df['open']) > (1.2 * df['atr_14'])
    df['ob_bullish_created'] = df['strong_bullish_impulse'] & (df['close'].shift(1) < df['open'].shift(1))
    
    df['ob_bullish_level'] = np.where(df['ob_bullish_created'], df['open'].shift(1), np.nan)
    df['ob_bullish_level'] = df['ob_bullish_level'].ffill()
    df['distance_to_bullish_ob_r'] = (df['close'] - df['ob_bullish_level']) / df['atr_14']
    df['distance_to_bullish_ob_r'] = df['distance_to_bullish_ob_r'].fillna(0)

    print("    -> Calculando Familia C: Derivados y Flujo Institucional...")
    # 1. Z-Score del Funding Rate (¿Están las masas pagando demasiado por ir en largo?)
    df['funding_rate_mean'] = df['funding_rate'].rolling(window=672, min_periods=1).mean()
    df['funding_rate_std'] = df['funding_rate'].rolling(window=672, min_periods=1).std()
    df['funding_zscore'] = np.where(df['funding_rate_std'] > 0, 
                                    (df['funding_rate'] - df['funding_rate_mean']) / df['funding_rate_std'], 
                                    0)

    # 2. Análisis de Open Interest (El Verdadero Flujo de Dinero)
    if 'open_interest' in df.columns and df['open_interest'].sum() > 0:
        df['oi_change_pct'] = df['open_interest'].pct_change().fillna(0)
        # Sube el precio y sube el OI -> Están inyectando dinero en largos
        df['oi_bullish_fuel'] = ((df['close'] > df['close'].shift(1)) & (df['oi_change_pct'] > 0)).astype(int)
        # Baja el precio y sube el OI -> Están inyectando dinero en cortos
        df['oi_bearish_fuel'] = ((df['close'] < df['close'].shift(1)) & (df['oi_change_pct'] > 0)).astype(int)
    else:
        # Mecanismo de fallback seguro si falta la data (como nos pasó ahora)
        df['oi_change_pct'] = 0.0
        df['oi_bullish_fuel'] = 0
        df['oi_bearish_fuel'] = 0

    # Limpieza final (Borramos las columnas base que ya no aportan y causan ruido)
    cols_to_drop = ['prev_close', 'tr', 'strong_bullish_impulse', 'ob_bullish_created', 'funding_rate_mean', 'funding_rate_std']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    return df

def build_features_for_symbol(symbol):
    print(f"\n[*] Construyendo Features Avanzadas (Tech + SMC + Derivados) para {symbol}...")
    
    in_file = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
    if not os.path.exists(in_file):
        return
        
    df = pd.read_parquet(in_file)
    
    # Inyectar inteligencia matemática
    df_features = calculate_features(df)
    
    out_dir = os.path.join(FEATURES_DIR, f"symbol={symbol}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "features.parquet")
    
    table = pa.Table.from_pandas(df_features)
    pq.write_table(table, out_file, compression='snappy')
    
    print(f"  [OK] Features generadas: {len(df_features.columns)} variables | {len(df_features):,} filas.")

def main():
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    print("\n" + "="*50)
    print("🧠 FEATURE ENGINE: Technicals + Smart Money + Derivados V2.0")
    print("="*50)
    
    for symbol in symbols:
        build_features_for_symbol(symbol)

if __name__ == "__main__":
    main()