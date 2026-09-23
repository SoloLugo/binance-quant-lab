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
    print(f"\n{'='*55}\nAnálisis de Umbrales (Thresholds) para {symbol}\n{'='*55}")
    
    # 1. Cargar datos de prueba (Test Set)
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    df = pd.read_parquet(features_path).join(pd.read_parquet(labels_path), how='inner')
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    # Hacer el split exacto para probar solo en datos NO vistos por el modelo
    split_idx = int(len(df) * 0.8)
    horizon = config['strategy']['horizon_bars']
    test_df = df.iloc[split_idx + horizon:]
    
    X_test = test_df[pd.read_parquet(features_path).columns]
    y_test = test_df['target'].values
    
    # 2. Cargar el "cerebro" (Modelo y Escalador)
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    model_path = os.path.join(symbol_out_dir, "logreg_v01.joblib")
    scaler_path = os.path.join(symbol_out_dir, "scaler_v01.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"[!] Faltan los archivos .joblib para {symbol}. Entrena el modelo primero.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 3. Predecir probabilidades puras
    X_test_scaled = scaler.transform(X_test)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] # Extraemos P(WIN)
    
    # 4. Análisis de Expected Value (EV)
    rr_ratio = config['strategy']['risk_reward_ratio']
    
    print(f"[*] Evaluando rentabilidad teórica (Riesgo: 1R, Recompensa: {rr_ratio}R)")
    print(f"{'Umbral':<10} | {'Operaciones':<12} | {'Win Rate':<10} | {'EV (en R)':<10}")
    print("-" * 55)
    
    # Probamos umbrales desde 50% hasta 80% de exigencia
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        signals = y_pred_prob >= threshold
        total_trades = np.sum(signals)
        
        if total_trades == 0:
            print(f"{threshold:>7.0%}    | {total_trades:<12} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        wins = np.sum(y_test[signals] == 1)
        win_rate = wins / total_trades
        
        # EV = (P(Ganar) * Recompensa) - (P(Perder) * Riesgo)
        ev = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
        
        # Damos formato visual: Verde si es rentable, rojo si pierde
        ev_str = f"{ev:+.3f} R"
        print(f"{threshold:>7.0%}    | {total_trades:<12} | {win_rate:>7.1%}    | {ev_str:<10}")

def main():
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    for symbol in symbols:
        evaluate_thresholds(symbol, config)

if __name__ == "__main__":
    main()