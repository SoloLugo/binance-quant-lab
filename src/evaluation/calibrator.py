import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "models")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def evaluate_calibration_and_economics(symbol, config):
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        return

    # 1. CARGA Y LIMPIEZA
    df = pd.read_parquet(features_path).join(pd.read_parquet(labels_path), how='inner')
    df.index = pd.to_datetime(df.index)
    df = df[df['label'] != -2].dropna()
    df['target_tp_before_sl'] = (df['label'] == 1).astype(int)
    
    # 2. SPLIT TEMPORAL ESTRICTO (Igual que en el trainer para evaluar el Test Locked)
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

    if len(test_df) == 0: return

    X_test = test_df[pd.read_parquet(features_path).columns]
    y_test = test_df['target_tp_before_sl'].values
    
    # 3. CARGA DE MODELOS CALIBRADOS
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    model_path = os.path.join(symbol_out_dir, "model_v02.joblib")
    scaler_path = os.path.join(symbol_out_dir, "scaler_v02.joblib")
    
    if not os.path.exists(model_path): return
    calibrated_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    y_pred_prob = calibrated_model.predict_proba(scaler.transform(X_test))[:, 1]
    
    # =======================================================
    # CAPA 1: MÉTRICAS DE CALIBRACIÓN PURA
    # =======================================================
    print(f"\n{'='*70}\n[1] MÉTRICAS DE CALIBRACIÓN PURA: {symbol}\n{'='*70}")
    brier = brier_score_loss(y_test, y_pred_prob)
    lloss = log_loss(y_test, y_pred_prob)
    print(f"[*] Brier Score : {brier:.4f} (Mide exactitud general. Menor a 0.25 = Mejor que el azar)")
    print(f"[*] Log Loss    : {lloss:.4f} (Penaliza fuertemente la sobreconfianza)")

    # =======================================================
    # CAPA 2: CURVA DE FIABILIDAD (RELIABILITY DIAGRAM)
    # =======================================================
    print(f"\n{'='*70}\n[2] CURVA DE FIABILIDAD (EXPECTATIVAS VS REALIDAD)\n{'='*70}")
    # Agrupamos las predicciones en 5 bloques (ej. las del 20%, las del 30%, etc.)
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=5, strategy='quantile')
    
    print(f"Si el modelo dice (Confianza)... | En la realidad ocurrió (Win Rate real)")
    print("-" * 70)
    for p_pred, p_true in zip(prob_pred, prob_true):
        desviacion = abs(p_pred - p_true)
        # Si la desviación es menor al 3%, está perfectamente calibrado
        estado = "✅ Preciso" if desviacion < 0.03 else "⚠️ Desviado"
        print(f"         {p_pred:>5.1%}                 |          {p_true:>5.1%}   [{estado}]")

    # =======================================================
    # CAPA 3: ECONOMÍA Y FRICCIÓN (EL BACKTESTER INICIAL)
    # =======================================================
    print(f"\n{'='*70}\n[3] DECISIÓN ECONÓMICA Y NET EV\n{'='*70}")
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    costo_friccion_r = 0.10 
    
    print(f"[*] Riesgo: 1.00R | Recompensa: {rr_ratio}R | Fricción Real: -{costo_friccion_r}R por trade")
    print(f"\n{'Umbral':<8} | {'Ops':<5} | {'Win Rate':<9} | {'EV Bruto':<10} | {'EV Neto':<10}")
    print("-" * 70)
    
    # Umbrales realistas desde el Break-Even (33.3%) hacia arriba
    for threshold in [0.35, 0.38, 0.40, 0.42, 0.45]:
        signals = y_pred_prob >= threshold
        total_trades = np.sum(signals)
        
        if total_trades == 0: continue
            
        wins = np.sum(y_test[signals] == 1)
        win_rate = wins / total_trades
        
        ev_bruto = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
        ev_neto = ev_bruto - costo_friccion_r
        
        print(f"{threshold:>6.0%}   | {total_trades:<5} | {win_rate:>7.1%}   | {ev_bruto:+.3f} R  | {ev_neto:+.3f} R")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        evaluate_calibration_and_economics(symbol, config)

if __name__ == "__main__":
    main()