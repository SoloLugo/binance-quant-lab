import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Cargamos las variables definidas en el archivo .env
load_dotenv()


@dataclass(frozen=True)
class QuantConfig:
    """
    Configuración centralizada e inmutable del sistema cuantitativo.
    Soporta 3 modos de operación mediante EXECUTION_MODE:
      - 'paper'   : Simulación en tiempo real con datos orgánicos de mercado, spread bid/ask,
                    slippage, comisiones taker, pagos de funding y persistencia en disco.
      - 'testnet' : Ejecución de órdenes en Binance Futures Testnet con datos de mercado reales.
      - 'mainnet' : Producción real con dinero real en Binance USD-M Futures.
    """
    # --- Rutas de almacenamiento en el NVMe ---
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/raw")
    model_artifact_path: str = os.getenv("MODEL_PATH", "models/quant_brain.joblib")
    sim_state_path: str = os.getenv("SIM_STATE_PATH", "data/sim_state.json")
    trade_log_path: str = os.getenv("TRADE_LOG_PATH", "data/trades_history.csv")

    # --- Parámetros de Mercado y Descarga Histórica ---
    primary_symbol: str = os.getenv("PRIMARY_SYMBOL", "BTCUSDT")
    context_symbol: str = os.getenv("CONTEXT_SYMBOL", "ETHUSDT")
    ccxt_symbol: str = os.getenv("CCXT_SYMBOL", "BTC/USDT:USDT")
    timeframe: str = os.getenv("TIMEFRAME", "15m")

    # Meses históricos a descargar desde data.binance.vision (Enero 2024 a Octubre 2026)
    start_year_month: str = os.getenv("START_YEAR_MONTH", "2024-01")
    end_year_month: str = os.getenv("END_YEAR_MONTH", "2026-10")

    # --- Parámetros del Modelo Triple-Barrier y Bill Benter EV ---
    tp_multiple: float = float(os.getenv("TP_MULTIPLE", "2.0"))
    sl_multiple: float = float(os.getenv("SL_MULTIPLE", "1.0"))
    horizon_bars: int = int(os.getenv("HORIZON_BARS", "24"))  # 24 velas de 15m = 6 horas
    roundtrip_fee: float = float(os.getenv("ROUNDTRIP_FEE", "0.0010"))  # 0.05% apertura + 0.05% cierre
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "1.5"))  # 1.5 bps de slippage estimado por orden
    ev_min_threshold: float = float(os.getenv("EV_MIN_THRESHOLD", "0.12"))  # Umbral mínimo en unidades R
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.20"))  # One-Fifth Kelly por seguridad
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.03"))  # Máximo 3% de la cuenta
    enable_shorts: bool = os.getenv("ENABLE_SHORTS", "true").lower() == "true"

    # --- Gestión de Riesgo y Límites para Dinero Real ---
    max_leverage: int = int(os.getenv("MAX_LEVERAGE", "5"))
    margin_mode: str = os.getenv("MARGIN_MODE", "ISOLATED")  # ISOLATED o CROSSED
    max_daily_drawdown_pct: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.06"))  # Circuit breaker 6% diario
    min_notional_usdt: float = float(os.getenv("MIN_NOTIONAL_USDT", "100.0"))  # Mínimo exigido por Binance en BTCUSDT
    min_qty_btc: float = float(os.getenv("MIN_QTY_BTC", "0.002"))

    # --- Simulación en Tiempo Real (Paper Trading) ---
    initial_paper_balance: float = float(os.getenv("INITIAL_PAPER_BALANCE", "10000.0"))
    poll_interval_seconds: int = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))

    # --- Credenciales y Entorno de Ejecución ---
    use_testnet: bool = os.getenv("USE_TESTNET", "true").lower() == "true"
    execution_mode: str = os.getenv(
        "EXECUTION_MODE",
        "testnet" if os.getenv("USE_TESTNET", "true").lower() == "true" else "mainnet"
    ).lower()
    binance_api_key: str = os.getenv("BINANCE_TESTNET_API_KEY", os.getenv("BINANCE_API_KEY", ""))
    binance_secret: str = os.getenv("BINANCE_TESTNET_SECRET", os.getenv("BINANCE_API_SECRET", ""))
    mainnet_api_key: str = os.getenv("BINANCE_MAINNET_API_KEY", os.getenv("BINANCE_API_KEY", ""))
    mainnet_secret: str = os.getenv("BINANCE_MAINNET_SECRET", os.getenv("BINANCE_API_SECRET", ""))

    # --- Recursos de Hardware (Orange Pi 4 Pro - 8 cores ARM64) ---
    n_jobs: int = int(os.getenv("LGBM_N_JOBS", "4"))


# Instancia global para importar en cualquier módulo
settings = QuantConfig()