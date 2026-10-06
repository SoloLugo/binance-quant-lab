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
    # ---------------------------------------------------------
    # FAMILIA A: PRICE / TECHNICAL (Volatilidad, Momentum, Tendencia)
    # ---------------------------------------------------------
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # EMAs y distancia al precio (Normalizada por ATR)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['dist_ema_20_atr'] = (df['close'] - df['ema_20']) / df['atr_14']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    print("    -> Calculando Familia B: Smart Money & Market Structure...")
    # ---------------------------------------------------------
    # FAMILIA B: SMART MONEY (Swings, Liquidity Sweeps, Order Blocks)
    # ---------------------------------------------------------
    
    # 1. SWING HIGHS & LOWS (Retraso de 2 velas para 0% leakage)
    # Un Swing High se confirma hoy si la vela de hace 2 periodos fue más alta 
    # que las 2 anteriores y que las 2 siguientes (la de ayer y la de hoy).
    df['is_swing_high'] = (df['high'].shift(2) > df['high'].shift(1)) & \
                          (df['high'].shift(2) > df['high']) & \
                          (df['high'].shift(2) > df['high'].shift(3)) & \
                          (df['high'].shift(2) > df['high'].shift(4))
                          
    df['is_swing_low'] = (df['low'].shift(2) < df['low'].shift(1)) & \
                         (df['low'].shift(2) < df['low']) & \
                         (df['low'].shift(2) < df['low'].shift(3)) & \
                         (df['low'].shift(2) < df['low'].shift(4))

    # Guardar el valor del último Swing confirmado
    df['last_swing_high'] = np.where(df['is_swing_high'], df['high'].shift(2), np.nan)
    df['last_swing_high'] = df['last_swing_high'].ffill()
    
    df['last_swing_low'] = np.where(df['is_swing_low'], df['low'].shift(2), np.nan)
    df['last_swing_low'] = df['last_swing_low'].ffill()

    # 2. LIQUIDITY SWEEPS (Barridos institucionales de liquidez)
    # El precio baja del último Swing Low (asusta a la gente), pero el cierre recupera (trampa).
    df['liquidity_sweep_bullish'] = ((df['low'] < df['last_swing_low']) & 
                                     (df['close'] > df['last_swing_low'])).astype(int)
                                     
    df['liquidity_sweep_bearish'] = ((df['high'] > df['last_swing_high']) & 
                                     (df['close'] < df['last_swing_high'])).astype(int)

    # 3. ORDER BLOCKS (OB) - Aproximación Matemática
    # Detectamos impulsos alcistas fuertes (cierre - apertura > 1.2 ATR)
    df['strong_bullish_impulse'] = (df['close'] - df['open']) > (1.2 * df['atr_14'])
    
    # Si hoy hay un impulso fuerte, la vela bajista previa se marca como Bullish Order Block
    df['ob_bullish_created'] = df['strong_bullish_impulse'] & (df['close'].shift(1) < df['open'].shift(1))
    
    # Guardamos el nivel superior del Order Block (el Open de la vela bajista)
    df['ob_bullish_level'] = np.where(df['ob_bullish_created'], df['open'].shift(1), np.nan)
    df['ob_bullish_level'] = df['ob_bullish_level'].ffill()
    
    # Como pidió el experto: Distancia al OB en términos de Riesgo (R/ATR)
    df['distance_to_bullish_ob_r'] = (df['close'] - df['ob_bullish_level']) / df['atr_14']
    df['distance_to_bullish_ob_r'] = df['distance_to_bullish_ob_r'].fillna(0)

    # Limpieza final: Rellenar NaNs iniciales, borrar columnas temporales e infinitos
    cols_to_drop = ['prev_close', 'tr', 'strong_bullish_impulse', 'ob_bullish_created']
    df.drop(columns=cols_to_drop, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    return df

def build_features_for_symbol(symbol):
    print(f"\n[*] Construyendo Features Avanzadas (SMC) para {symbol}...")
    
    in_file = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
    if not os.path.exists(in_file):
        return
        
    df = pd.read_parquet(in_file)
    
    # Inyectar inteligencia matemática
    df_features = calculate_features(df)
    
    # Guardar contrato de datos final
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
    print("🧠 FEATURE ENGINE: Smart Money & Technicals V1.0")
    print("="*50)
    
    for symbol in symbols:
        build_features_for_symbol(symbol)

if __name__ == "__main__":
    main()