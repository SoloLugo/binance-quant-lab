import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f: return yaml.safe_load(f)

def run_walk_forward(symbol, config):
    print(f"\n{'='*70}\n🔄 WALK-FORWARD CON REGIME ANALYSIS: {symbol}\n{'='*70}")
    
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
    
    # Separamos variables de predicción de variables de contexto
    context_cols = ['regime_bull', 'regime_bear', 'regime_high_vol']
    features_cols = [c for c in df_features.columns if c not in context_cols and c != 'label']
    
    horizon_bars = config['strategy'].get('horizon_bars', 30)
    purge_td = pd.Timedelta(minutes=15 * horizon_bars)
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    friccion = 0.10
    umbral = 0.35 

    n_splits = 4
    fold_size = int(len(df) / (n_splits + 1))
    
    # Aquí guardaremos el historial de todas las operaciones generadas en las pruebas
    historico_trades = []

    for fold in range(n_splits):
        train_end_idx = fold_size * (fold + 1)
        test_end_idx = fold_size * (fold + 2)
        
        train_df = df.iloc[:train_end_idx]
        last_train_time = train_df.index[-1]
        
        test_df_full = df.iloc[train_end_idx:test_end_idx]
        test_start_safe = last_train_time + purge_td
        test_df = test_df_full[test_df_full.index > test_start_safe]
        
        if len(test_df) == 0: continue
            
        X_train, y_train = train_df[features_cols], train_df['target']
        X_test, y_test = test_df[features_cols], test_df['target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv_splits = max(2, min(3, len(train_df[train_df['target'] == 1])))
        
        try:
            calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=cv_splits)
            calibrated_model.fit(X_train_scaled, y_train)
        except: continue
        
        y_pred = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        # Almacenamos los resultados de esta época con su contexto
        test_resultados = test_df[['target'] + context_cols].copy()
        test_resultados['prob'] = y_pred
        test_resultados['signal'] = (y_pred >= umbral).astype(int)
        historico_trades.append(test_resultados[test_resultados['signal'] == 1])

    if not historico_trades:
        print("[!] El modelo no disparó señales en ninguna época.")
        return

    # Consolidamos todo el historial
    trades_df = pd.concat(historico_trades)
    
    # Función para calcular EV
    def calc_ev(sub_df):
        if len(sub_df) == 0: return 0, 0, 0
        ops = len(sub_df)
        win_rate = sub_df['target'].mean()
        ev = (win_rate * rr_ratio) - ((1 - win_rate) * 1) - friccion
        return ops, win_rate, ev

    total_ops, total_wr, total_ev = calc_ev(trades_df)
    
    print("\n📊 ANÁLISIS DE REGÍMENES (Out-Of-Sample Consolidado)")
    print(f"{'Régimen Macro':<25} | {'Ops':<5} | {'Win Rate':<10} | {'EV Neto':<10}")
    print("-" * 60)
    print(f"{'GLOBAL (Todos los climas)':<25} | {total_ops:<5} | {total_wr:<10.1%} | {total_ev:>+7.3f} R")
    
    # Desglose por Regímenes
    for regime_name, condition in [
        ("📈 Alcista (Bull Trend)", trades_df['regime_bull'] == 1),
        ("📉 Bajista (Bear Trend)", trades_df['regime_bear'] == 1),
        ("🌪️ Alta Volatilidad", trades_df['regime_high_vol'] == 1),
        ("🐢 Baja Volatilidad", trades_df['regime_high_vol'] == 0),
        ("🔥 Bull + Alta Vol", (trades_df['regime_bull'] == 1) & (trades_df['regime_high_vol'] == 1))
    ]:
        ops, wr, ev = calc_ev(trades_df[condition])
        if ops > 0:
            print(f"{regime_name:<25} | {ops:<5} | {wr:<10.1%} | {ev:>+7.3f} R")

    print(f"\n{'-'*70}")
    print("🏆 CONCLUSIÓN DEL CONTEXT ENGINE")
    if total_ev > 0:
        print("[APROBADO GLOBAL] El modelo tiene Edge independientemente del clima.")
    else:
        # Buscamos si hay algún régimen rentable para salvar la estrategia
        best_regime = None
        best_ev = 0
        for regime_name, condition in [("Alcista", trades_df['regime_bull'] == 1), ("Bajista", trades_df['regime_bear'] == 1)]:
            ops, wr, ev = calc_ev(trades_df[condition])
            if ev > best_ev and ops > 30: # Mínimo 30 operaciones para relevancia estadística
                best_ev = ev
                best_regime = regime_name
                
        if best_regime:
            print(f"[EDGE CONDICIONAL] El modelo pierde en general, pero es rentable en Régimen {best_regime}.")
            print("-> Acción recomendada: Implementar filtro de régimen en el bot de ejecución.")
        else:
            print("[REPROBADO] No se encontró Edge estadístico en ningún régimen.")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        run_walk_forward(symbol, config)

if __name__ == "__main__":
    main()