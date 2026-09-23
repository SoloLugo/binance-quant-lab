import os
import yaml
import numpy as np
import pandas as pd
import joblib

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "models")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def evaluate_thresholds(symbol, config):
    print(f"\n{'='*70}\nAnálisis de Umbrales con Fricción Real (Net EV) para {symbol}\n{'='*70}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    df = pd.read_parquet(features_path).join(pd.read_parquet(labels_path), how='inner')
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    split_idx = int(len(df) * 0.8)
    horizon = config['strategy'].get('horizon_bars', 30)
    test_df = df.iloc[split_idx + horizon:]
    
    X_test = test_df[pd.read_parquet(features_path).columns]
    y_test = test_df['target'].values
    
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    model_path = os.path.join(symbol_out_dir, "model_v02.joblib")
    scaler_path = os.path.join(symbol_out_dir, "scaler_v02.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"[!] Faltan los archivos del modelo para {symbol}.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    
    # COSTO DE FRICCIÓN (Comisiones + Slippage)
    # Asumimos que nos cuesta 0.10 R abrir y cerrar cada posición
    costo_friccion_r = 0.10 
    
    print(f"[*] Riesgo Bruto: 1.00R | Recompensa: {rr_ratio}R")
    print(f"[*] Fricción (Fees + Slippage): -{costo_friccion_r}R cobrados por cada operación")
    print(f"\n{'Umbral':<10} | {'Ops':<6} | {'Win Rate':<10} | {'EV Bruto':<10} | {'EV Neto (Real)':<12}")
    print("-" * 70)
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        signals = y_pred_prob >= threshold
        total_trades = np.sum(signals)
        
        if total_trades == 0:
            print(f"{threshold:>7.0%}    | {total_trades:<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12}")
            continue
            
        wins = np.sum(y_test[signals] == 1)
        win_rate = wins / total_trades
        
        # EV Bruto = Laboratorio Teórico
        ev_bruto = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
        
        # EV Neto = Mundo Real (EV Bruto - Costos de Binance)
        ev_neto = ev_bruto - costo_friccion_r
        
        bruto_str = f"{ev_bruto:+.3f} R"
        neto_str = f"{ev_neto:+.3f} R"
        
        print(f"{threshold:>7.0%}    | {total_trades:<6} | {win_rate:>7.1%}    | {bruto_str:<10} | {neto_str:<12}")

def main():
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    for symbol in symbols:
        evaluate_thresholds(symbol, config)

if __name__ == "__main__":
    main()