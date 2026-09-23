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

def generate_labels(df, direction, risk_pct, rr_ratio, horizon):
    """
    Motor de trayectoria (Path-dependent Labeler).
    Retorna: 1 (WIN), 0 (LOSS), -1 (TIMEOUT), -2 (AMBIGUOUS), NaN (Inválido)
    """
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    # Restamos horizon+1 para no salirnos del índice al mirar el futuro
    for i in range(n - horizon - 1):
        # Regla 1: Entrada en el Open de la VELA SIGUIENTE (t+1)
        entry_price = opens[i + 1]
        
        if direction == "LONG":
            sl_price = entry_price * (1 - risk_pct)
            tp_price = entry_price * (1 + (risk_pct * rr_ratio))
        else: # SHORT
            sl_price = entry_price * (1 + risk_pct)
            tp_price = entry_price * (1 - (risk_pct * rr_ratio))
            
        hit_tp_idx = -1
        hit_sl_idx = -1
        
        # Simular la trayectoria de los precios vela por vela
        for j in range(1, horizon + 1):
            idx = i + 1 + j # Empezamos a evaluar desde t+2 (trayectoria post-entrada)
            if idx >= n: break
            
            curr_high = highs[idx]
            curr_low = lows[idx]
            
            # Verificar toques
            if direction == "LONG":
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price
            else:
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price
            
            if tp_hit and sl_hit:
                # El precio osciló salvajemente tocando ambos en 15 minutos
                hit_tp_idx = idx
                hit_sl_idx = idx
                break
            elif tp_hit:
                hit_tp_idx = idx
                break
            elif sl_hit:
                hit_sl_idx = idx
                break
                
        # Clasificar el resultado de la trayectoria
        if hit_tp_idx != -1 and hit_tp_idx == hit_sl_idx:
            labels[i] = -2  # AMBIGUOUS (Descartar)
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
    
    print("\n🎯 Iniciando Motor de Etiquetado Probabilístico...")
    print(f"[*] Configuración: {strat['direction']} | Riesgo: {strat['risk_pct']*100}% | Recompensa: {strat['risk_reward_ratio']}R | Horizonte: {strat['horizon_bars']} velas")
    
    for symbol in symbols:
        parquet_path = os.path.join(CANONICAL_DIR, f"symbol={symbol}", "data.parquet")
        
        if not os.path.exists(parquet_path):
            print(f"  [!] No hay datos canónicos para {symbol}.")
            continue
            
        print(f"\n[*] Etiquetando {symbol}...")
        df = pd.read_parquet(parquet_path)
        
        # Ejecutar el algoritmo de barreras
        labels_array = generate_labels(
            df, 
            strat['direction'], 
            strat['risk_pct'], 
            strat['risk_reward_ratio'], 
            strat['horizon_bars']
        )
        
        # Guardar en formato Parquet solo el índice y la etiqueta (Data Contract eficiente)
        df_labels = pd.DataFrame(index=df.index)
        df_labels['label'] = labels_array
        
        # Contar resultados para el reporte
        res = df_labels['label'].value_counts()
        wins = res.get(1, 0)
        losses = res.get(0, 0)
        timeouts = res.get(-1, 0)
        ambiguous = res.get(-2, 0)
        
        print(f"  [>] WINS (1): {wins:,} | LOSSES (0): {losses:,} | TIMEOUTS (-1): {timeouts:,} | AMBIGUOUS (-2): {ambiguous:,}")
        
        # Guardar disco
        out_dir = os.path.join(LABELS_DIR, f"symbol={symbol}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "labels.parquet")
        
        table = pa.Table.from_pandas(df_labels)
        pq.write_table(table, out_file, compression='snappy')
        print(f"  [OK] Etiquetas guardadas en: {out_file}")

if __name__ == "__main__":
    main()