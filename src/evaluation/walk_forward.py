import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def run_walk_forward(symbol, config):
    print(f"\n{'='*70}\n🔄 WALK-FORWARD (LINEAR ENGINE): {symbol}\n{'='*70}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    if not (os.path.exists(features_path) and os.path.exists(labels_path)): return

    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)
    
    df_features.index = df_features.index.tz_localize(None)
    df_labels.index = df_labels.index.tz_localize(None)
    
    df = df_features.join(df_labels, how='inner')
    df.index = pd.to_datetime(df.index)
    
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    features_cols = pd.read_parquet(features_path).columns.tolist()
    
    horizon_bars = config['strategy'].get('horizon_bars', 30)
    purge_td = pd.Timedelta(minutes=15 * horizon_bars)
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    friccion = 0.10
    umbral = 0.35 

    n_splits = 4
    fold_size = int(len(df) / (n_splits + 1))
    resultados = []

    for fold in range(n_splits):
        print(f"\n[*] Evaluando Época {fold + 1} de {n_splits}...")
        
        train_end_idx = fold_size * (fold + 1)
        test_end_idx = fold_size * (fold + 2)
        
        train_df = df.iloc[:train_end_idx]
        last_train_time = train_df.index[-1]
        
        test_df_full = df.iloc[train_end_idx:test_end_idx]
        test_start_safe = last_train_time + purge_td
        test_df = test_df_full[test_df_full.index > test_start_safe]
        
        if len(test_df) == 0: 
            print("    [!] No hay datos suficientes.")
            continue
            
        print(f"    Train: {train_df.index[0].date()} a {last_train_time.date()} | Test: {test_df.index[0].date()} a {test_df.index[-1].date()}")
        
        X_train, y_train = train_df[features_cols], train_df['target']
        X_test, y_test = test_df[features_cols], test_df['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_splits = min(3, len(train_df[train_df['target'] == 1]))
        if cv_splits < 2: cv_splits = 2
        
        try:
            calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=cv_splits)
            calibrated_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"    [!] Error calibrando: {e}")
            continue
        
        y_pred = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        signals = y_pred >= umbral
        trades = np.sum(signals)
        
        if trades > 0:
            wins = np.sum(y_test[signals] == 1)
            win_rate = wins / trades
            ev_bruto = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
            ev_neto = ev_bruto - friccion
        else:
            win_rate, ev_neto = 0, 0
            
        print(f"    Resultados -> Ops: {trades} | Win Rate: {win_rate:>7.1%} | EV Neto: {ev_neto:+.3f} R")
        resultados.append(ev_neto)

    if resultados:
        ev_promedio = np.mean(resultados)
        epocas_positivas = sum(1 for r in resultados if r > 0)
    else:
        ev_promedio = 0
        epocas_positivas = 0
        
    print(f"\n{'-'*70}")
    print(f"🏆 VEREDICTO DE ROBUSTEZ PARA {symbol}")
    print(f"Épocas rentables: {epocas_positivas} de {n_splits} | EV Neto Promedio: {ev_promedio:+.3f} R")
    if epocas_positivas >= 3:
        print("[APROBADO] El modelo es robusto.")
    else:
        print("[REPROBADO] Requiere más investigación de variables.")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        run_walk_forward(symbol, config)

if __name__ == "__main__":
    main()