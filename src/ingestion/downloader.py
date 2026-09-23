import os
import json
import yaml
import hashlib
import requests
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Rutas absolutas estables dentro del contenedor Docker (/app)
BASE_DIR = "/app"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")
MANIFEST_FILE = os.path.join(RAW_DIR, "manifest.json")

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"[!] No se encontró el archivo de configuración en: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_manifest(manifest):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

def download_and_verify(zip_url, checksum_url, local_zip, local_checksum):
    try:
        resp_chk = requests.get(checksum_url, timeout=10)
        if resp_chk.status_code != 200:
            return False, None
        expected_hash = resp_chk.text.split()[0]
    except:
        return False, None

    resp_zip = requests.get(zip_url, stream=True, timeout=10)
    if resp_zip.status_code != 200:
        return False, None
    
    total_size = int(resp_zip.headers.get('content-length', 0))
    with open(local_zip, 'wb') as file, tqdm(
        desc=os.path.basename(local_zip), total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in resp_zip.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    with open(local_checksum, 'w') as f:
        f.write(resp_chk.text)

    sha256_hash = hashlib.sha256()
    with open(local_zip, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    real_hash = sha256_hash.hexdigest()
    if expected_hash == real_hash:
        return True, real_hash
    else:
        if os.path.exists(local_zip):
            os.remove(local_zip)
        return False, None

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    config = load_config()
    manifest = load_manifest()
    
    symbols = config['ingestion']['symbols']
    tf = config['ingestion']['timeframe']
    start_date = pd.to_datetime(config['ingestion']['start_date'])
    end_date = pd.Timestamp(datetime.now().date())
    
    print(f"\n🚀 Iniciando Motor de Ingestión en Orange Pi (ARM64)")
    
    for symbol in symbols:
        print(f"\n{'='*40}\nSímbolo: {symbol}\n{'='*40}")
        meses_cerrados = pd.date_range(start=start_date.replace(day=1), end=end_date.replace(day=1) - pd.Timedelta(days=1), freq='MS')
        
        for mes in meses_cerrados:
            mes_str = mes.strftime("%Y-%m")
            zip_filename = f"{symbol}-{tf}-{mes_str}.zip"
            zip_local = os.path.join(RAW_DIR, zip_filename)
            checksum_local = zip_local + ".CHECKSUM"
            
            if zip_filename in manifest and manifest[zip_filename]['status'] == 'VERIFIED' and os.path.exists(zip_local):
                print(f"  [SKIPPED] {zip_filename} ya está descargado y verificado.")
                continue
                
            print(f"[*] Procesando: {mes_str}")
            base_url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{tf}/"
            success, file_hash = download_and_verify(base_url + zip_filename, base_url + zip_filename + ".CHECKSUM", zip_local, checksum_local)
            
            if success:
                manifest[zip_filename] = {
                    "symbol": symbol,
                    "type": "monthly",
                    "date": mes_str,
                    "hash": file_hash,
                    "downloaded_at": datetime.now().isoformat(),
                    "status": "VERIFIED"
                }
                save_manifest(manifest)

if __name__ == "__main__":
    main()