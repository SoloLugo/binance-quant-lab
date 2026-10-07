import os
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_production_model(symbol="BTCUSDT"):
    print(f"\n[*] Entrenando Modelo de Producción (V1) para {symbol}...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)
    
    # Neutralizar zonas horarias y ensamblar
    df_features.index = df_features.index.tz_localize(None)
    df_labels.index = df_labels.index.tz_localize(None)
    df = df_features.join(df_labels, how='inner').dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    context_cols = ['regime_bull', 'regime_bear', 'regime_high_vol']
    features_cols = [c for c in df_features.columns if c not in context_cols and c != 'label']
    
    X = df[features_cols]
    y = df['target']
    
    # Entrenar Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar Modelo Lineal Calibrado
    model = LogisticRegression(random_state=42, max_iter=1000)
    calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
    calibrated_model.fit(X_scaled, y)
    
    # Exportar Artefactos
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{symbol}_scaler_v1.joblib"))
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, f"{symbol}_model_v1.joblib"))
    
    print(f"  [OK] Artefactos guardados en {MODELS_DIR}")
    print(f"  -> Features activas ({len(features_cols)}): {features_cols}")

if __name__ == "__main__":
    train_production_model("BTCUSDT")