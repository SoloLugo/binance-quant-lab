import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, classification_report

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "models")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(symbol, config):
    print(f"\n{'='*50}\nEntrenando Modelo V0.2 (Gradient Boosting) para {symbol}\n{'='*50}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print(f"[!] Faltan datos para {symbol}.")
        return

    X_df = pd.read_parquet(features_path)
    y_df = pd.read_parquet(labels_path)

    df = X_df.join(y_df, how='inner')
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    features_cols = X_df.columns.tolist()

    split_idx = int(len(df) * 0.8)
    horizon = config['strategy']['horizon_bars']
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx + horizon:]
    
    X_train = train_df[features_cols]
    y_train = train_df['target']
    X_test = test_df[features_cols]
    y_test = test_df['target']

    # Escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # NUEVO MOTOR: HistGradientBoosting (No-Lineal)
    print("[*] Ajustando Árboles de Decisión Secuenciales (Gradient Boosting)...")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, 
        max_iter=300,            # 300 árboles construidos uno sobre otro
        max_depth=5,             # Profundidad máxima de cada árbol
        class_weight='balanced', # Compensar que hay más pérdidas que ganancias
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluación
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"[*] ROC-AUC Score : {roc_auc:.4f}")
    
    # GUARDAR EL MODELO V0.2
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    os.makedirs(symbol_out_dir, exist_ok=True)
    
    model_path = os.path.join(symbol_out_dir, "model_v02.joblib")
    scaler_path = os.path.join(symbol_out_dir, "scaler_v02.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  [OK] Modelo V0.2 guardado en: {model_path}")

def main():
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    for symbol in symbols:
        train_and_evaluate(symbol, config)

if __name__ == "__main__":
    main()