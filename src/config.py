import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Cargamos las variables definidas en el archivo .env
load_dotenv()


@dataclass(frozen=True)
class QuantConfig:
    """
    Configuración centralizada e inmutable del sistema cuantitativo.
    Permite alternar entre Testnet y Producción Real (Mainnet) desde el archivo .env.
    """
    # --- Rutas de almacenamiento en el NVMe ---
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/raw")
    model_artifact_path: str = os.getenv("MODEL_PATH", "models/quant_brain.joblib")

    # --- Parámetros de Mercado y Descarga Histórica ---
    primary_symbol: str = "BTCUSDT"
    context_symbol: str = "ETHUSDT"
    ccxt_symbol: str = "BTC/USDT:USDT"
    timeframe: str = "15m"
    
    # Meses históricos a descargar desde data.binance.vision (Enero 2024 a Octubre 2026)
    start_year_month: str = os.getenv("START_YEAR_MONTH", "2024-01")
    end_year_month: str = os.getenv("END_YEAR_MONTH", "2026-10")

    # --- Parámetros del Modelo Triple-Barrier y Bill Benter EV ---
    tp_multiple: float = float(os.getenv("TP_MULTIPLE", "2.0"))
    sl_multiple: float = float(os.getenv("SL_MULTIPLE", "1.0"))
    horizon_bars: int = int(os.getenv("HORIZON_BARS", "24"))  # 24 velas de 15m = 6 horas
    roundtrip_fee: float = float(os.getenv("ROUNDTRIP_FEE", "0.0010"))  # 0.05% apertura + 0.05% cierre
    ev_min_threshold: float = float(os.getenv("EV_MIN_THRESHOLD", "0.12"))  # Umbral mínimo en unidades R
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.20"))  # One-Fifth Kelly por seguridad
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.03"))  # Máximo 3% de la cuenta

    # --- Credenciales y Entorno de Ejecución ---
    use_testnet: bool = os.getenv("USE_TESTNET", "true").lower() == "true"
    binance_api_key: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    binance_secret: str = os.getenv("BINANCE_TESTNET_SECRET", "")

    # --- Recursos de Hardware (Orange Pi 4 Pro - 8 cores ARM64) ---
    n_jobs: int = int(os.getenv("LGBM_N_JOBS", "4"))


# Instancia global para importar en cualquier módulo
settings = QuantConfig()