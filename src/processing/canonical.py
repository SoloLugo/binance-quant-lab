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
    with open(CONFIG_FILE, "r") as f: return yaml.safe_load(f)

def process_klines(symbol_dir):
    csv_files = [f for f in glob.glob(os.path.join(symbol_dir, "*.csv")) if "fundingRate" not in f and "openInterest" not in f]
    if not csv_files: return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, names=cols, header=None)
            if isinstance(df.iloc[0]['open_time'], str) and 'time' in df.iloc[0]['open_time'].lower(): df = df.iloc[1:]
            df_list.append(df)
        except: pass
    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df['open_time'] = pd.to_datetime(pd.to_numeric(df['open_time']), unit='ms')
    df.set_index('open_time', inplace=True)
    return df[~df.index.duplicated(keep='first')].sort_index()[['open', 'high', 'low', 'close', 'volume']].astype(float)

def process_open_interest(symbol_dir):
    csv_files = glob.glob(os.path.join(symbol_dir, "*openInterestHist*.csv"))
    if not csv_files: return pd.DataFrame()
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # FIX DE BINANCE VISION: Extraer forzosamente OI y Timestamp
            if 'timestamp' in df.columns and 'sumOpenInterest' in df.columns:
                temp_df = df[['timestamp', 'sumOpenInterest']].copy()
            else:
                # Si no hay headers, sabemos que el index 1 es OI y el index 3 (o el último) es timestamp
                time_col = df.columns[3] if len(df.columns) > 3 else df.columns[-1]
                oi_col = df.columns[1]
                temp_df = df[[time_col, oi_col]].copy()
                
            temp_df.columns = ['time', 'open_interest']
            df_list.append(temp_df)
        except: pass
            
    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df[~df.index.duplicated(keep='first')].sort_index()[['open_interest']].astype(float)

def process_funding_rate(symbol_dir):
    csv_files = glob.glob(os.path.join(symbol_dir, "*fundingRate*.csv"))
    if not csv_files: return pd.DataFrame()
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            time_col = 'calc_time' if 'calc_time' in df.columns else 'funding_time' if 'funding_time' in df.columns else df.columns[0]
            rate_col = 'funding_rate' if 'funding_rate' in df.columns else 'last_funding_rate' if 'last_funding_rate' in df.columns else df.columns[1]
            temp_df = df[[time_col, rate_col]].rename(columns={time_col: 'time', rate_col: 'funding_rate'})
            df_list.append(temp_df)
        except: pass
    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df[~df.index.duplicated(keep='first')].sort_index()[['funding_rate']].astype(float)

def process_symbol(symbol):
    print(f"\n[*] Procesando Contrato de Datos (Canonical) para {symbol}...")
    symbol_dir = os.path.join(RAW_DIR, f"symbol={symbol}")
    
    df_klines = process_klines(symbol_dir)
    if df_klines.empty: return
    df_oi = process_open_interest(symbol_dir)
    df_funding = process_funding_rate(symbol_dir)
    
    print(f"  -> Velas base      : {len(df_klines):,} filas")
    df_master = df_klines.copy()
    
    if not df_oi.empty:
        print(f"  -> Alineando OI    : {len(df_oi):,} filas")
        df_master = df_master.join(df_oi, how='left')
        df_master['open_interest'] = df_master['open_interest'].ffill()
    else: df_master['open_interest'] = 0.0
        
    if not df_funding.empty:
        print(f"  -> Alineando Fund  : {len(df_funding):,} filas")
        df_master = df_master.join(df_funding, how='left')
        df_master['funding_rate'] = df_master['funding_rate'].ffill().fillna(0.0)
    else: df_master['funding_rate'] = 0.0

    df_master.dropna(inplace=True)
    out_dir = os.path.join(CANONICAL_DIR, f"symbol={symbol}")
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df_master), os.path.join(out_dir, "data.parquet"), compression='snappy')
    print(f"  [OK] Master Data Contract guardado.")

def main():
    os.makedirs(CANONICAL_DIR, exist_ok=True)
    config = load_config()
    print("\n" + "="*50 + "\n🏭 ETL: Master Data Contract (Klines + OI + Funding)\n" + "="*50)
    for symbol in config['ingestion']['symbols']: process_symbol(symbol)

if __name__ == "__main__": main()