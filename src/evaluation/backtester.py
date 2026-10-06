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

def run_backtest(symbol, config, threshold=0.35, initial_capital=10000.0, risk_pct=0.01):
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)): return

    # 1. PREPARACIÓN DE DATOS (Mismo corte temporal estricto)
    df = pd.read_parquet(features_path).join(pd.read_parquet(labels_path), how='inner')
    df.index = pd.to_datetime(df.index)
    df = df[df['label'] != -2].dropna()
    df['target_tp_before_sl'] = (df['label'] == 1).astype(int)
    
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
    
    # 2. PREDICCIÓN
    symbol_out_dir = os.path.join(OUTPUTS_DIR, f"symbol={symbol}")
    model_path = os.path.join(symbol_out_dir, "model_v02.joblib")
    scaler_path = os.path.join(symbol_out_dir, "scaler_v02.joblib")
    
    if not os.path.exists(model_path): return
    calibrated_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    y_pred_prob = calibrated_model.predict_proba(scaler.transform(X_test))[:, 1]
    
    # 3. MOTOR DE SIMULACIÓN (Event-Driven Logic)
    rr_ratio = config['strategy'].get('risk_reward_ratio', 2.0)
    costo_friccion_r = 0.10 # Comisiones y slippage en términos de R
    
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    
    for i, prob in enumerate(y_pred_prob):
        if prob >= threshold:
            # Dimensionamiento de la posición: Arriesgamos el 1% del capital actual
            riesgo_usd = capital * risk_pct
            
            es_ganadora = y_test[i] == 1
            
            if es_ganadora:
                # Gana 2R bruto
                pnl_bruto = riesgo_usd * rr_ratio
            else:
                # Pierde 1R bruto
                pnl_bruto = -riesgo_usd
                
            # Restamos la fricción física al resultado (0.10R del riesgo)
            costo_friccion_usd = riesgo_usd * costo_friccion_r
            pnl_neto = pnl_bruto - costo_friccion_usd
            
            capital += pnl_neto
            equity_curve.append(capital)
            trades.append(pnl_neto)
            
    if not trades:
        print(f"[*] {symbol}: Cero operaciones en el umbral {threshold:.0%}")
        return
        
    # 4. MÉTRICAS INSTITUCIONALES
    trades = np.array(trades)
    ganancias = trades[trades > 0]
    perdidas = trades[trades <= 0]
    
    win_rate = len(ganancias) / len(trades)
    gross_profit = ganancias.sum() if len(ganancias) > 0 else 0
    gross_loss = abs(perdidas.sum()) if len(perdidas) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss
    
    retorno_total = (capital - initial_capital) / initial_capital
    
    # Drawdown Máximo (Caída desde el pico)
    equity_series = pd.Series(equity_curve)
    picos = equity_series.cummax()
    drawdowns = (equity_series - picos) / picos
    max_drawdown = abs(drawdowns.min())
    
    print(f"\n{'='*60}\n📊 BACKTEST REPORT: {symbol} (Test Set Locked)\n{'='*60}")
    print(f"[*] Umbral de disparo: {threshold:>7.1%}")
    print(f"[*] Capital Inicial  : ${initial_capital:,.2f}")
    print(f"[*] Capital Final    : ${capital:,.2f}")
    print("-" * 60)
    print(f"[*] Retorno Total    : {retorno_total:>7.2%}")
    print(f"[*] Max Drawdown     : {max_drawdown:>7.2%} (Caída máxima de la cuenta)")
    print(f"[*] Profit Factor    : {profit_factor:>7.2f} (Beneficio bruto / Pérdida bruta)")
    print(f"[*] Total Trades     : {len(trades)}")
    print(f"[*] Win Rate         : {win_rate:>7.2%}")

def main():
    config = load_config()
    # Usaremos el umbral del 35% que vimos que funcionaba excelente en ETHUSDT
    threshold_optimo = 0.35 
    
    for symbol in config['ingestion']['symbols']:
        run_backtest(symbol, config, threshold=threshold_optimo)

if __name__ == "__main__":
    main()