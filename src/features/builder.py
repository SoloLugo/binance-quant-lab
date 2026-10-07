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
    with open(CONFIG_FILE, "r") as f: return yaml.safe_load(f)

def calculate_features(df):
    print("    -> Calculando Familia A: Análisis Técnico (Volatilidad & Momentum)...")
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['dist_ema_20_atr'] = (df['close'] - df['ema_20']) / df['atr_14']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))

    print("    -> Calculando Familia C: Derivados (Funding & Open Interest)...")
    if 'funding_rate' in df.columns:
        df['funding_rate_mean'] = df['funding_rate'].rolling(window=672, min_periods=1).mean()
        df['funding_rate_std'] = df['funding_rate'].rolling(window=672, min_periods=1).std()
        df['funding_zscore'] = np.where(df['funding_rate_std'] > 0, (df['funding_rate'] - df['funding_rate_mean']) / df['funding_rate_std'], 0)
    else:
        df['funding_zscore'] = 0.0

    if 'open_interest' in df.columns and df['open_interest'].sum() > 0:
        # Cambio porcentual del OI (¿Entra o sale dinero del contrato?)
        df['oi_change_pct'] = df['open_interest'].pct_change().fillna(0)
        # Z-Score del OI para saber si el apalancamiento es anormalmente alto
        df['oi_mean'] = df['open_interest'].rolling(window=672, min_periods=1).mean()
        df['oi_std'] = df['open_interest'].rolling(window=672, min_periods=1).std()
        df['oi_zscore'] = np.where(df['oi_std'] > 0, (df['open_interest'] - df['oi_mean']) / df['oi_std'], 0)
    else:
        df['oi_change_pct'] = 0.0
        df['oi_zscore'] = 0.0

    print("    -> Calculando Familia D: CONTEXT ENGINE (Regímenes de Mercado)...")
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_200_slope'] = (df['ema_200'] - df['ema_200'].shift(10)) / df['ema_200'].shift(10)
    df['regime_bull'] = (df['close'] > df['ema_200']).astype(int)
    df['regime_bear'] = (df['close'] < df['ema_200']).astype(int)

    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['bb_width'] = (df['std_20'] * 4) / df['sma_20']
    df['bb_width_mean_1w'] = df['bb_width'].rolling(window=672, min_periods=1).mean()
    df['regime_high_vol'] = (df['bb_width'] > df['bb_width_mean_1w']).astype(int)

    # LIMPIEZA DRÁSTICA: Eliminamos todo el Smart Money inútil y las variables base de precio
    cols_to_drop = [
        'open', 'high', 'low', 'close', 'prev_close', 'tr', 'sma_20', 'std_20', 
        'funding_rate_mean', 'funding_rate_std', 'bb_width_mean_1w', 'oi_mean', 'oi_std', 'ema_20', 'ema_200'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    return df

def build_features_for_symbol(symbol):
    print(f"\n[*] Construyendo Features Purificadas para {symbol}...")
    in_file = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
    if not os.path.exists(in_file): return
    df = pd.read_parquet(in_file)
    df_features = calculate_features(df)
    out_dir = os.path.join(FEATURES_DIR, f"symbol={symbol}")
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df_features), os.path.join(out_dir, "features.parquet"), compression='snappy')
    print(f"  [OK] Features limpias: {len(df_features.columns)} variables | {len(df_features):,} filas.")

def main():
    config = load_config()
    print("\n" + "="*50 + "\n🧠 FEATURE ENGINE: ABLATION COMPLETED\n" + "="*50)
    for symbol in config['ingestion']['symbols']: build_features_for_symbol(symbol)

if __name__ == "__main__": main()