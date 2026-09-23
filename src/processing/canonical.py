import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = "/app"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CANONICAL_DIR = os.path.join(BASE_DIR, "data", "canonical")
MANIFEST_FILE = os.path.join(RAW_DIR, "manifest.json")

BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", 
    "taker_buy_base", "taker_buy_quote", "ignore"
]

def load_manifest():
    if not os.path.exists(MANIFEST_FILE):
        print(f"[!] No se encontró el manifiesto en: {MANIFEST_FILE}")
        return {}
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)

def validate_invariants(df):
    invariants_ok = (
        (df['low'] <= df['open']) & 
        (df['open'] <= df['high']) &
        (df['low'] <= df['close']) & 
        (df['close'] <= df['high']) &
        (df['volume'] >= 0)
    ).all()
    
    nulls_ok = df.isnull().sum().sum() == 0
    return invariants_ok and nulls_ok

def process_symbol_data(symbol, manifest):
    print(f"\n[*] Procesando dataset canónico para: {symbol}")
    
    files_to_process = [
        f for f, meta in manifest.items() 
        if meta.get("symbol") == symbol and meta.get("status") == "VERIFIED"
    ]
    
    if not files_to_process:
        print(f"  [!] No hay archivos verificados para {symbol}.")
        return

    files_to_process.sort()
    
    df_list = []
    for filename in tqdm(files_to_process, desc=f"Leyendo ZIPs de {symbol}"):
        filepath = os.path.join(RAW_DIR, filename)
        try:
            # Leer todo como texto (str) primero para evitar el cuelgue de tipos
            df_temp = pd.read_csv(filepath, names=BINANCE_COLUMNS, header=None, dtype=str)
            df_list.append(df_temp)
        except Exception as e:
            print(f"  [!] Error leyendo {filename}: {e}")

    if not df_list:
        return

    df_full = pd.concat(df_list, ignore_index=True)

    # --- EL FILTRO ANTI-BASURA DE BINANCE ---
    # Eliminar cualquier fila donde 'open_time' sea el string literal de encabezado
    df_full = df_full[df_full['open_time'] != 'open_time']
    
    # Forzar la conversión a numérico para asegurar operaciones matemáticas limpias
    df_full['open_time'] = pd.to_numeric(df_full['open_time'])

    # Limpieza temporal
    df_full['timestamp'] = pd.to_datetime(df_full['open_time'], unit='ms', utc=True)
    df_full.drop(columns=['open_time', 'close_time', 'ignore'], inplace=True)
    df_full.set_index('timestamp', inplace=True)
    df_full.sort_index(inplace=True)

    # Eliminar posibles duplicados
    df_full = df_full[~df_full.index.duplicated(keep='first')]

    # Optimización de memoria
    float_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
    df_full[float_cols] = df_full[float_cols].astype('float32')
    df_full['trades'] = df_full['trades'].astype('int32')

    if not validate_invariants(df_full):
        print("  [ERROR CRÍTICO] Los datos violan invariantes de precio (ej. Low > High).")
        return

    esperado = pd.date_range(start=df_full.index.min(), end=df_full.index.max(), freq='15min', tz='UTC')
    faltantes = len(esperado) - len(df_full)
    if faltantes > 0:
        print(f"  [WARNING] Se detectaron {faltantes} velas faltantes en la historia completa.")

    symbol_dir = os.path.join(CANONICAL_DIR, f"symbol={symbol}")
    os.makedirs(symbol_dir, exist_ok=True)
    
    parquet_path = os.path.join(symbol_dir, "data.parquet")
    table = pa.Table.from_pandas(df_full)
    pq.write_table(table, parquet_path, compression='snappy')
    
    print(f"  [OK] Dataset canónico guardado en: {parquet_path}")
    print(f"  [INFO] Total de velas de 15m procesadas: {len(df_full):,}")

def main():
    os.makedirs(CANONICAL_DIR, exist_ok=True)
    manifest = load_manifest()
    
    if not manifest:
        print("[!] El manifiesto está vacío o no se pudo cargar.")
        return

    symbols = set(meta.get("symbol") for meta in manifest.values())
    
    print("🚀 Iniciando Construcción del Dataset Canónico...")
    for symbol in symbols:
        process_symbol_data(symbol, manifest)

if __name__ == "__main__":
    main()