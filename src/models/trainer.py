import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "models")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(symbol, config):
    print(f"\n{'='*50}\nEntrenando Modelo V0.4 (Probabilidades Naturales + Sigmoid) para {symbol}\n{'='*50}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        return

    X_df = pd.read_parquet(features_path)
    y_df = pd.read_parquet(labels_path)

    df = X_df.join(y_df, how='inner')
    df.index = pd.to_datetime(df.index)
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    features_cols = X_df.columns.tolist()

    # Split: 60% Train, 20% Val, 20% Test (Con purga temporal)
    horizon_bars = config['strategy'].get('horizon_bars', 30)
    purge_td = pd.Timedelta(minutes=15 * horizon_bars)

    idx_60 = int(len(df) * 0.6)
    idx_80 = int(len(df) * 0.8)

    train_df = df.iloc[:idx_60]
    val_start = train_df.index[-1] + purge_td
    val_df_full = df[df.index > val_start]
    
    val_df = val_df_full.iloc[:(idx_80 - idx_60)]
    if len(val_df) == 0: return
    
    test_start = val_df.index[-1] + purge_td
    test_df = df[df.index > test_start]

    X_train, y_train = train_df[features_cols], train_df['target']
    X_val, y_val = val_df[features_cols], val_df['target']
    X_test, y_test = test_df[features_cols], test_df['target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Entrenar modelo raw SIN class_weight='balanced'
    # Dejamos que el modelo aprenda la distribución natural del mercado
    base_model = HistGradientBoostingClassifier(
        learning_rate=0.05, 
        max_iter=300, 
        max_depth=5, 
        random_state=42
    )
    base_model.fit(X_train_scaled, y_train)

    # 2. Calibración Sigmoide (Platt Scaling) en Validation Set
    # Sigmoid evita el colapso en escalones que genera isotonic en muestras pequeñas
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val_scaled, y_val)

    # 3. Test Locked
    y_pred_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    brier = brier_score_loss(y_test, y_pred_prob)
    
    print(f"[*] ROC-AUC (Test): {roc_auc:.4f} | Brier Score: {brier:.4f}")
    
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    os.makedirs(symbol_out_dir, exist_ok=True)
    joblib.dump(calibrated_model, os.path.join(symbol_out_dir, "model_v02.joblib"))
    joblib.dump(scaler, os.path.join(symbol_out_dir, "scaler_v02.joblib"))

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        train_and_evaluate(symbol, config)

if __name__ == "__main__":
    main()