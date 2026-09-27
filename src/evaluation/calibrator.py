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
    print(f"\n{'='*70}\nAnálisis de Umbrales (Probabilidades Calibradas) para {symbol}\n{'='*70}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    df = pd.read_parquet(features_path).join(pd.read_parquet(labels_path), how='inner')
    df.index = pd.to_datetime(df.index)
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    # Split idéntico al trainer para aislar el Test Set (Último 20%)
    horizon_bars = config['strategy'].get('horizon_bars', 30)
    purge_td = pd.Timedelta(minutes=15 * horizon_bars)

    idx_60 = int(len(df) * 0.6)
    idx_80 = int(len(df) * 0.8)

    train_df = df.iloc[:idx_60]
    val_start = train_df.index[-1] + purge_td
    val_df_full = df[df.index > val_start]
    
    val_df = val_df_full.iloc[:(idx_80 - idx_60)]
    test_start = val_df.index[-1] + purge_td
    test_df = df[df.index > test_start]

    X_test = test_df[pd.read_parquet(features_path).columns]
    y_test = test_df['target'].values
    
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    calibrated_model = joblib.load(os.path.join(symbol_out_dir, "model_v02.joblib"))
    scaler = joblib.load(os.path.join(symbol_out_dir, "scaler_v02.joblib"))
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    costo_friccion_r = 0.10 
    
    print(f"[*] Fricción: -{costo_friccion_r}R por operación")
    print(f"\n{'Umbral':<10} | {'Ops':<6} | {'Win Rate':<10} | {'EV Bruto':<10} | {'EV Neto':<12}")
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        signals = y_pred_prob >= threshold
        total_trades = np.sum(signals)
        
        if total_trades == 0:
            continue
            
        wins = np.sum(y_test[signals] == 1)
        win_rate = wins / total_trades
        
        ev_bruto = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
        ev_neto = ev_bruto - costo_friccion_r
        
        print(f"{threshold:>7.0%}    | {total_trades:<6} | {win_rate:>7.1%}    | {ev_bruto:+.3f} R  | {ev_neto:+.3f} R")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        evaluate_thresholds(symbol, config)

if __name__ == "__main__":
    main()