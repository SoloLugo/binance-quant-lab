import hashlib
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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


def compute_bytes_sha256(data: bytes) -> str:
    """Calcula el hash SHA256 hexadecimal de un bloque de bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Calcula el hash SHA256 hexadecimal de un archivo en disco de forma incremental."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_csv_bytes(data: bytes) -> bool:
    """
    Verifica que un bloque de bytes corresponda a un CSV no vacío y no corrupto
    (sin bytes NUL, decodificable en UTF-8 y con al menos una línea con comas).
    """
    if not data or b"\x00" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and "," in stripped:
            return True
    return False


def is_valid_csv_file(file_path: str) -> bool:
    """Verifica que un archivo CSV en disco exista, no esté vacío y no esté corrupto."""
    if not os.path.isfile(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    try:
        with open(file_path, "rb") as f:
            sample = f.read(65536)
        return is_valid_csv_bytes(sample)
    except OSError:
        return False


def verify_remote_checksum(
    zip_url: str,
    zip_bytes: bytes,
    timeout: int = 15,
) -> Tuple[str, Optional[str]]:
    """
    Verifica opcionalmente el archivo .CHECKSUM publicado por Binance Vision.
    - Si .CHECKSUM devuelve 404, no falla la descarga y retorna ('not_available_404', None).
    - Si está disponible (200), compara el SHA256 del ZIP descargado contra el esperado.
    """
    checksum_url = f"{zip_url}.CHECKSUM"
    try:
        resp = requests.get(checksum_url, timeout=timeout)
        if resp.status_code == 404:
            return "not_available_404", None
        resp.raise_for_status()
        tokens = resp.text.strip().split()
        if not tokens:
            return "invalid_remote_checksum", None
        expected_sha256 = tokens[0].lower()
        actual_sha256 = compute_bytes_sha256(zip_bytes).lower()
        if actual_sha256 == expected_sha256:
            return "verified", expected_sha256
        return "mismatch", expected_sha256
    except requests.RequestException:
        return "checksum_request_error", None


def download_and_extract_zip(
    url: str,
    dest_folder: str,
    expected_csv_name: str,
    overwrite: bool = False,
    verify_checksum: bool = True,
) -> Dict[str, Any]:
    """
    Descarga un archivo .zip de Binance Vision en memoria RAM, verifica su integridad
    (.CHECKSUM opcional + validación ZIP/CSV) y extrae el CSV de forma atómica en disco.
    Si el CSV ya existe y es válido, NO lo sobrescribe salvo que overwrite=True.
    """
    os.makedirs(dest_folder, exist_ok=True)
    target_path = os.path.join(dest_folder, expected_csv_name)

    if os.path.exists(target_path):
        if is_valid_csv_file(target_path):
            if not overwrite:
                return {
                    "filename": expected_csv_name,
                    "path": target_path,
                    "url": url,
                    "success": True,
                    "status": "skipped_existing",
                    "message": "Ya existe en disco y es válido (Omitido; overwrite=False)",
                    "sha256": compute_file_sha256(target_path),
                    "size_bytes": os.path.getsize(target_path),
                    "checksum_status": "not_checked_local",
                    "remote_checksum": None,
                }
        else:
            # Archivo existente vacío o corrupto: se reemplaza en la descarga
            pass

    checksum_status = "skipped"
    remote_checksum: Optional[str] = None

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return {
                "filename": expected_csv_name,
                "path": target_path,
                "url": url,
                "success": False,
                "status": "not_found_404",
                "message": "No disponible aún en Binance Vision (404)",
                "sha256": None,
                "size_bytes": 0,
                "checksum_status": "not_applicable",
                "remote_checksum": None,
            }
        response.raise_for_status()

        zip_bytes = response.content
        if not zip_bytes:
            return {
                "filename": expected_csv_name,
                "path": target_path,
                "url": url,
                "success": False,
                "status": "corrupted_empty_zip",
                "message": "Respuesta ZIP vacía recibida desde el servidor",
                "sha256": None,
                "size_bytes": 0,
                "checksum_status": "failed_empty_zip",
                "remote_checksum": None,
            }

        if verify_checksum:
            checksum_status, remote_checksum = verify_remote_checksum(url, zip_bytes)
            if checksum_status == "mismatch":
                return {
                    "filename": expected_csv_name,
                    "path": target_path,
                    "url": url,
                    "success": False,
                    "status": "checksum_mismatch",
                    "message": (
                        f"Checksum SHA256 inválido para {expected_csv_name} "
                        f"(esperado={remote_checksum}, actual={compute_bytes_sha256(zip_bytes)})"
                    ),
                    "sha256": None,
                    "size_bytes": 0,
                    "checksum_status": checksum_status,
                    "remote_checksum": remote_checksum,
                }

        # Extraemos el contenido del ZIP en memoria y validamos que no esté corrupto ni vacío
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return {
                        "filename": expected_csv_name,
                        "path": target_path,
                        "url": url,
                        "success": False,
                        "status": "empty_zip_no_csv",
                        "message": "ZIP vacío sin archivo CSV",
                        "sha256": None,
                        "size_bytes": 0,
                        "checksum_status": checksum_status,
                        "remote_checksum": remote_checksum,
                    }

                with z.open(csv_files[0]) as source:
                    csv_bytes = source.read()
        except zipfile.BadZipFile as exc:
            return {
                "filename": expected_csv_name,
                "path": target_path,
                "url": url,
                "success": False,
                "status": "corrupted_zip",
                "message": f"Archivo ZIP corrupto: {exc}",
                "sha256": None,
                "size_bytes": 0,
                "checksum_status": checksum_status,
                "remote_checksum": remote_checksum,
            }

        if not is_valid_csv_bytes(csv_bytes):
            return {
                "filename": expected_csv_name,
                "path": target_path,
                "url": url,
                "success": False,
                "status": "corrupted_empty_csv",
                "message": "El CSV extraído está vacío o corrupto",
                "sha256": None,
                "size_bytes": 0,
                "checksum_status": checksum_status,
                "remote_checksum": remote_checksum,
            }

        # Escritura atómica para evitar archivos truncados en disco
        tmp_path = f"{target_path}.tmp"
        with open(tmp_path, "wb") as target:
            target.write(csv_bytes)
        os.replace(tmp_path, target_path)

        file_sha256 = compute_file_sha256(target_path)
        size_bytes = os.path.getsize(target_path)
        return {
            "filename": expected_csv_name,
            "path": target_path,
            "url": url,
            "success": True,
            "status": "downloaded",
            "message": f"Descargado OK (checksum={checksum_status})",
            "sha256": file_sha256,
            "size_bytes": size_bytes,
            "checksum_status": checksum_status,
            "remote_checksum": remote_checksum,
        }
    except Exception as exc:
        return {
            "filename": expected_csv_name,
            "path": target_path,
            "url": url,
            "success": False,
            "status": "error",
            "message": f"Error: {exc}",
            "sha256": None,
            "size_bytes": 0,
            "checksum_status": checksum_status,
            "remote_checksum": remote_checksum,
        }


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


def write_download_manifest(
    results: List[Dict[str, Any]],
    manifests_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Escribe un manifiesto JSON de la descarga en settings.manifests_dir con hashes SHA256
    y el estado de verificación .CHECKSUM de cada archivo.
    """
    target_dir = manifests_dir or settings.manifests_dir
    os.makedirs(target_dir, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    timestamp_tag = now_utc.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = os.path.join(target_dir, f"download_manifest_{timestamp_tag}.json")

    sorted_results = sorted(results, key=lambda r: r.get("filename", ""))
    ok_count = sum(1 for r in sorted_results if r.get("success"))
    skipped_count = sum(1 for r in sorted_results if r.get("status") == "skipped_existing")
    downloaded_count = sum(1 for r in sorted_results if r.get("status") == "downloaded")
    failed_count = len(sorted_results) - ok_count

    payload = {
        "downloaded_at": now_utc.isoformat(),
        "source": BASE_VISION_URL,
        "primary_symbol": settings.primary_symbol,
        "context_symbol": settings.context_symbol,
        "timeframe": settings.timeframe,
        "start_year_month": settings.start_year_month,
        "end_year_month": settings.end_year_month,
        "overwrite": overwrite,
        "summary": {
            "total": len(sorted_results),
            "ok": ok_count,
            "downloaded": downloaded_count,
            "skipped_existing": skipped_count,
            "failed": failed_count,
        },
        "files": sorted_results,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return manifest_path


def run_downloader(
    overwrite: bool = False,
    verify_checksum: bool = True,
    manifests_dir: Optional[str] = None,
) -> str:
    tasks = build_download_tasks()
    print(f"Iniciando descarga paralela desde Binance Vision ({len(tasks)} archivos)...")
    print(f"Rango configurado: {settings.start_year_month} hasta {settings.end_year_month}\n")

    results: List[Dict[str, Any]] = []
    ok_count = 0
    # Usamos 6 hilos para descargar rápido sin saturar la red ni que Binance limite la IP
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                download_and_extract_zip,
                url,
                dest,
                name,
                overwrite=overwrite,
                verify_checksum=verify_checksum,
            )
            for url, dest, name in tasks
        ]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status = "[OK]" if res["success"] else "[WARN]"
            print(f"{status:6} {res['filename']:38} -> {res['message']}")
            if res["success"]:
                ok_count += 1

    manifest_path = write_download_manifest(
        results,
        manifests_dir=manifests_dir,
        overwrite=overwrite,
    )
    print(f"\nProceso finalizado: {ok_count}/{len(tasks)} archivos listos en '{settings.raw_data_dir}'.")
    print(f"Manifiesto de descarga guardado en: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    run_downloader()