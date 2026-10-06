import os
import glob
import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = "/app"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CANONICAL_DIR = os.path.join(BASE_DIR, "data", "canonical")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def process_symbol(symbol):
    print(f"[*] Transformando datos Raw a Canonical (Parquet) para {symbol}...")
    
    # Buscar todos los CSVs descargados para este símbolo
    csv_pattern = os.path.join(RAW_DIR, f"symbol={symbol}", "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"  [!] No se encontraron archivos CSV para {symbol} en {csv_pattern}")
        return
        
    # Columnas estándar de Binance Vision Klines
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    df_list = []
    for file in csv_files:
        try:
            # Leemos el CSV. Binance a veces incluye cabecera y a veces no.
            temp_df = pd.read_csv(file, names=cols, header=None)
            
            # Si la primera fila resultó ser texto (cabecera original del CSV), la saltamos
            if isinstance(temp_df.iloc[0]['open_time'], str) and 'time' in temp_df.iloc[0]['open_time'].lower():
                temp_df = temp_df.iloc[1:]
                
            df_list.append(temp_df)
        except Exception as e:
            print(f"  [!] Error leyendo {file}: {e}")
            
    if not df_list:
        return
        
    # Unificar todos los datos del símbolo
    df = pd.concat(df_list, ignore_index=True)
    
    # Transformación del tiempo (Milisegundos a Datetime)
    df['open_time'] = pd.to_numeric(df['open_time'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    
    # Ordenar cronológicamente y eliminar velas duplicadas (solapamientos)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Conservar únicamente la microestructura OHLCV necesaria y castear a float
    columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
    df = df[columns_to_keep].astype(float)
    
    # Guardar en formato Parquet comprimido (Data Contract estricto)
    out_dir = os.path.join(CANONICAL_DIR, f"symbol={symbol}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "data.parquet")
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_file, compression='snappy')
    
    print(f"  [OK] Guardado exitoso: {len(df):,} velas estandarizadas en {out_file}")

def main():
    os.makedirs(CANONICAL_DIR, exist_ok=True)
    config = load_config()
    symbols = config['ingestion']['symbols']
    
    print("\n" + "="*50)
    print("🏭 ETL: Generando Data Contract (Canonical Parquet)")
    print("="*50)
    
    for symbol in symbols:
        process_symbol(symbol)

if __name__ == "__main__":
    main()