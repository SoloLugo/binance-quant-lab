import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import pandas as pd
import requests
from config import settings


BASE_VISION_URL = "https://data.binance.vision/data/futures/um/monthly"


def generate_months(start_ym: str, end_ym: str) -> List[str]:
    """
    Genera la lista de meses en formato YYYY-MM entre start_ym y end_ym.
    """
    months = pd.date_range(start=f"{start_ym}-01", end=f"{end_ym}-01", freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def download_and_extract_zip(url: str, dest_folder: str, expected_csv_name: str) -> Tuple[str, bool, str]:
    """
    Descarga un archivo .zip de Binance Vision en memoria RAM y extrae el CSV
    directamente en el disco NVMe. Si el CSV ya existe, lo salta para ahorrar red.
    """
    os.makedirs(dest_folder, exist_ok=True)
    target_path = os.path.join(dest_folder, expected_csv_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
        return expected_csv_name, True, "Ya existe en disco (Omitido)"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return expected_csv_name, False, "No disponible aún en Binance Vision (404)"
        response.raise_for_status()

        # Extraemos el contenido del ZIP en memoria y lo guardamos con el nombre esperado
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                return expected_csv_name, False, "ZIP vacío sin archivo CSV"
            
            with z.open(csv_files[0]) as source, open(target_path, "wb") as target:
                target.write(source.read())

        return expected_csv_name, True, "Descargado OK"
    except Exception as exc:
        return expected_csv_name, False, f"Error: {exc}"


def build_download_tasks() -> List[Tuple[str, str, str]]:
    """
    Construye la lista completa de tareas de descarga requeridas por nuestra arquitectura:
    1. Klines 15m de BTCUSDT (Activo principal y Contexto HTF)
    2. Klines 15m de ETHUSDT (Para medir fuerza relativa BTC -> ETH -> ALTs)
    3. Funding Rate de BTCUSDT (Para la capa de Derivados / Posicionamiento)
    """
    months = generate_months(settings.start_year_month, settings.end_year_month)
    tasks = []

    for ym in months:
        # 1. Klines de BTCUSDT y ETHUSDT
        for sym in [settings.primary_symbol, settings.context_symbol]:
            dest_dir = os.path.join(settings.raw_data_dir, f"symbol={sym}")
            csv_name = f"{sym}-{settings.timeframe}-{ym}.csv"
            zip_url = f"{BASE_VISION_URL}/klines/{sym}/{settings.timeframe}/{sym}-{settings.timeframe}-{ym}.zip"
            tasks.append((zip_url, dest_dir, csv_name))

        # 2. Funding Rate de BTCUSDT
        btc_dir = os.path.join(settings.raw_data_dir, f"symbol={settings.primary_symbol}")
        fr_csv_name = f"{settings.primary_symbol}-fundingRate-{ym}.csv"
        fr_zip_url = f"{BASE_VISION_URL}/fundingRate/{settings.primary_symbol}/{settings.primary_symbol}-fundingRate-{ym}.zip"
        tasks.append((fr_zip_url, btc_dir, fr_csv_name))

    return tasks


def run_downloader():
    tasks = build_download_tasks()
    print(f"Iniciando descarga paralela desde Binance Vision ({len(tasks)} archivos)...")
    print(f"Rango configurado: {settings.start_year_month} hasta {settings.end_year_month}\n")

    ok_count = 0
    # Usamos 6 hilos para descargar rápido sin saturar la red ni que Binance limite la IP
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(download_and_extract_zip, url, dest, name)
            for url, dest, name in tasks
        ]
        for future in as_completed(futures):
            name, success, msg = future.result()
            status = "[OK]" if success else "[WARN]"
            print(f"{status:6} {name:38} -> {msg}")
            if success:
                ok_count += 1

    print(f"\nProceso finalizado: {ok_count}/{len(tasks)} archivos listos en '{settings.raw_data_dir}'.")


if __name__ == "__main__":
    run_downloader()