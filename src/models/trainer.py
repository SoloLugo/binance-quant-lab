import os
import yaml
import json
import subprocess
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
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

def get_git_commit():
    try:
        # Intenta obtener el hash del último commit
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=BASE_DIR).decode('utf-8').strip()
        return commit
    except Exception:
        return "unknown"

def train_and_evaluate(symbol, config):
    print(f"\n{'='*60}\n🚀 ENTRENANDO BASELINE LOGÍSTICO PARA {symbol}\n{'='*60}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        return

    X_df = pd.read_parquet(features_path)
    y_df = pd.read_parquet(labels_path)

    df = X_df.join(y_df, how='inner')
    df.index = pd.to_datetime(df.index)
    
    df = df[df['label'] != -2].dropna()
    df['target_tp_before_sl'] = (df['label'] == 1).astype(int)
    features_cols = X_df.columns.tolist()

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

    X_train, y_train = train_df[features_cols], train_df['target_tp_before_sl']
    X_val, y_val = val_df[features_cols], val_df['target_tp_before_sl']
    X_test, y_test = test_df[features_cols], test_df['target_tp_before_sl']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. ENTRENAMOS Y CALIBRAMOS EL MODELO GANADOR (Logistic Regression)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val_scaled, y_val)
    
    y_pred = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    roc = roc_auc_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)
    
    print(f"[*] ROC-AUC     : {roc:.4f}")
    print(f"[*] Brier Score : {brier:.4f}")

    # 2. GUARDAMOS EL MODELO
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    os.makedirs(symbol_out_dir, exist_ok=True)
    joblib.dump(calibrated_model, os.path.join(symbol_out_dir, "model_v02.joblib"))
    joblib.dump(scaler, os.path.join(symbol_out_dir, "scaler_v02.joblib"))

    # 3. EXPERIMENT TRACKER (Punto 2 del Experto)
    metadata = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "symbol": symbol,
        "dataset_version": "v1.0",
        "feature_version": "v0.2_technical",
        "label_version": "v1.0_tp_before_sl_no_ambiguous",
        "model_type": "LogisticRegression",
        "model_parameters": {"max_iter": 1000, "random_state": 42},
        "calibration": "sigmoid (Platt Scaling)",
        "training_period": {"start": str(train_df.index[0]), "end": str(train_df.index[-1])},
        "validation_period": {"start": str(val_df.index[0]), "end": str(val_df.index[-1])},
        "test_period": {"start": str(test_df.index[0]), "end": str(test_df.index[-1])},
        "code_commit": get_git_commit(),
        "random_seed": 42,
        "metrics_on_test": {
            "roc_auc": float(roc),
            "brier_score": float(brier)
        }
    }
    
    metadata_path = os.path.join(symbol_out_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[*] ✅ Trazabilidad guardada en: {metadata_path}")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        train_and_evaluate(symbol, config)

if __name__ == "__main__":
    main()