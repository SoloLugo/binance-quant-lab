import os
import yaml
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = "/app"
CANONICAL_DIR = os.path.join(BASE_DIR, "data", "canonical")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def generate_dynamic_labels(df, direction, atr_multiplier, rr_ratio, horizon):
    """Motor de trayectoria con límites dinámicos basados en Volatilidad (ATR)"""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    # Calcular el True Range matemáticamente en NumPy
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]
    
    tr1 = highs - lows
    tr2 = np.abs(highs - prev_closes)
    tr3 = np.abs(lows - prev_closes)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calcular ATR de 14 periodos
    atr = pd.Series(tr).rolling(14).mean().values
    
    # Iteramos dejando un margen inicial para que el ATR se caliente
    for i in range(14, n - horizon - 1):
        entry_price = opens[i + 1]
        current_atr = atr[i]
        
        # Conversión de riesgo a dólares de forma dinámica
        risk_usd = current_atr * atr_multiplier
        reward_usd = risk_usd * rr_ratio
        
        if direction == "LONG":
            sl_price = entry_price - risk_usd
            tp_price = entry_price + reward_usd
        else: # SHORT
            sl_price = entry_price + risk_usd
            tp_price = entry_price - reward_usd
            
        hit_tp_idx = -1
        hit_sl_idx = -1
        
        for j in range(1, horizon + 1):
            idx = i + 1 + j
            if idx >= n: break
            
            curr_high = highs[idx]
            curr_low = lows[idx]
            
            if direction == "LONG":
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price
            else:
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price
            
            if tp_hit and sl_hit:
                hit_tp_idx = idx
                hit_sl_idx = idx
                break
            elif tp_hit:
                hit_tp_idx = idx
                break
            elif sl_hit:
                hit_sl_idx = idx
                break
                
        if hit_tp_idx != -1 and hit_tp_idx == hit_sl_idx:
            labels[i] = -2  # AMBIGUOUS
        elif hit_tp_idx != -1:
            labels[i] = 1   # WIN
        elif hit_sl_idx != -1:
            labels[i] = 0   # LOSS
        else:
            labels[i] = -1  # TIMEOUT
            
    return labels

def main():
    os.makedirs(LABELS_DIR, exist_ok=True)
    config = load_config()
    symbols = config['ingestion']['symbols']
    strat = config['strategy']
    
    # Extraer variables con valores por defecto de seguridad
    atr_mult = strat.get('atr_multiplier', 1.5)
    rr_ratio = strat.get('risk_reward_ratio', 2.0)
    horizon = strat.get('horizon_bars', 30)
    direction = strat.get('direction', 'LONG')
    
    print("\n🎯 Iniciando Motor de Etiquetado Dinámico (V0.3)...")
    print(f"[*] Dirección: {direction} | Stop Loss: {atr_mult}x ATR | Recompensa: {rr_ratio}R | Horizonte: {horizon} velas")
    
    for symbol in symbols:
        parquet_path = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
        if not os.path.exists(parquet_path): continue
            
        print(f"\n[*] Etiquetando {symbol} con barreras de volatilidad...")
        df = pd.read_parquet(parquet_path)
        
        labels_array = generate_dynamic_labels(df, direction, atr_mult, rr_ratio, horizon)
        
        df_labels = pd.DataFrame(index=df.index)
        df_labels['label'] = labels_array
        
        res = df_labels['label'].value_counts()
        print(f"  [>] WINS: {res.get(1, 0):,} | LOSSES: {res.get(0, 0):,} | TIMEOUTS: {res.get(-1, 0):,} | AMBIGUOUS: {res.get(-2, 0):,}")
        
        out_dir = os.path.join(LABELS_DIR, f"symbol={symbol}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "labels.parquet")
        
        table = pa.Table.from_pandas(df_labels)
        pq.write_table(table, out_file, compression='snappy')
        print(f"  [OK] Etiquetas guardadas en: {out_file}")

if __name__ == "__main__":
    main()