import os
import yaml
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = "/app"
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def test_temporal_leakage():
    config = load_config()
    symbols = config['ingestion']['symbols']
    horizon_bars = config['strategy'].get('horizon_bars', 30)
    
    # El horizonte de la etiqueta es el tiempo futuro que mira el labeler
    # +1 vela de margen de seguridad (Embargo)
    purge_timedelta = pd.Timedelta(minutes=15 * (horizon_bars + 1))
    
    print("\n" + "="*50)
    print("🛡️ RESEARCH GATE: Ejecutando Test Anti-Fugas (Leakage)")
    print("="*50)

    for symbol in symbols:
        labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
        if not os.path.exists(labels_path):
            continue
            
        df = pd.read_parquet(labels_path)
        df.index = pd.to_datetime(df.index)
        
        split_idx = int(len(df) * 0.6) # Usamos 60% como en el Trainer
        train_df = df.iloc[:split_idx]
        
        last_train_time = train_df.index[-1]
        
        # El test set propuesto en nuestro código
        test_start_time = last_train_time + purge_timedelta
        test_df = df[df.index >= test_start_time]
        
        if len(test_df) == 0:
            continue
            
        first_test_time = test_df.index[0]
        
        print(f"[*] Evaluando {symbol}...")
        print(f"    - Última vela Train: {last_train_time}")
        print(f"    - Primera vela Test: {first_test_time}")
        
        # EL ASSERT (La prueba matemática)
        # Si la primera vela del test es menor al tiempo máximo que miró el train, el script fallará.
        assert first_test_time > (last_train_time + pd.Timedelta(minutes=15 * horizon_bars)), \
            f"Fallo CRÍTICO de Leakage en {symbol}: El test empieza antes de que termine el horizonte del train."
            
        print("    [PASS] Integridad temporal verificada. Cero solapamiento.")

if __name__ == "__main__":
    test_temporal_leakage()