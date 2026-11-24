import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Cargamos las variables definidas en el archivo .env (si existe)
load_dotenv()

ALLOWED_EXECUTION_MODES = ("paper", "testnet")


def validate_execution_mode(mode: str, allow_mainnet: bool = False) -> str:
    """
    Valida el modo de ejecución aplicando el bloqueo estricto de Mainnet (Plan Maestro §1.4).
    Mainnet y cualquier operación con capital real están bloqueados por seguridad.
    """
    normalized = (mode or "paper").lower().strip()
    if normalized == "mainnet" or allow_mainnet:
        raise PermissionError(
            "BLOQUEO DE SEGURIDAD: El modo 'mainnet' y las operaciones con capital real están "
            "deshabilitados en Binance Quant Lab hasta completar todas las fases de validación "
            "y recibir autorización independiente."
        )
    if normalized not in ALLOWED_EXECUTION_MODES:
        raise ValueError(
            f"Modo de ejecución inválido: '{mode}'. Modos permitidos: {ALLOWED_EXECUTION_MODES}."
        )
    return normalized


@dataclass(frozen=True)
class QuantConfig:
    """
    Configuración centralizada e inmutable del sistema cuantitativo.
    Modos de operación permitidos mediante EXECUTION_MODE:
      - 'paper'   : (Predeterminado) Simulación con datos de mercado y estado aislado.
      - 'testnet' : Ejecución de pruebas en Binance Futures Testnet (con DRY_RUN=true por defecto).
    Mainnet está bloqueado permanentemente por diseño en esta fase del proyecto.
    """
    # --- Rutas de almacenamiento ---
    raw_data_dir: str = "data/raw"
    model_artifact_path: str = "models/quant_brain.joblib"
    sim_state_path: str = "data/paper_sim_state.json"
    trade_log_path: str = "data/paper_trades_history.csv"
    manifests_dir: str = "data/manifests"
    experiments_dir: str = "data/experiments"

    # --- Parámetros de Mercado y Descarga Histórica ---
    primary_symbol: str = "BTCUSDT"
    context_symbol: str = "ETHUSDT"
    ccxt_symbol: str = "BTC/USDT:USDT"
    timeframe: str = "15m"

    # Meses históricos a descargar desde data.binance.vision
    start_year_month: str = "2024-01"
    end_year_month: str = "2026-10"

    # --- Parámetros del Modelo Triple-Barrier y Bill Benter EV ---
    tp_multiple: float = 2.0
    sl_multiple: float = 1.0
    horizon_bars: int = 24  # 24 velas de 15m = 6 horas
    roundtrip_fee: float = 0.0010  # 0.05% apertura + 0.05% cierre
    slippage_bps: float = 1.5  # 1.5 bps de slippage estimado por orden
    half_spread_bps: float = 0.5  # 0.5 bps de half-spread estimado
    ev_min_threshold: float = 0.12  # Umbral mínimo en unidades R
    kelly_fraction: float = 0.20  # One-Fifth Kelly por seguridad
    max_risk_per_trade: float = 0.03  # Máximo 3% de la cuenta
    enable_shorts: bool = True

    # --- Gestión de Riesgo y Límites ---
    max_leverage: int = 5
    margin_mode: str = "ISOLATED"  # ISOLATED o CROSSED
    max_daily_drawdown_pct: float = 0.06  # Circuit breaker 6% diario
    min_notional_usdt: float = 100.0  # Mínimo exigido por Binance en BTCUSDT
    min_qty_btc: float = 0.002

    # --- Simulación en Tiempo Real (Paper Trading) ---
    initial_paper_balance: float = 10000.0
    poll_interval_seconds: int = 15

    # --- Seguridad, Credenciales y Entorno de Ejecución ---
    use_testnet: bool = True
    dry_run: bool = True
    allow_mainnet: bool = False
    execution_mode: str = "paper"
    binance_api_key: str = ""
    binance_secret: str = ""
    # Credenciales Mainnet deshabilitadas por seguridad (nunca cargadas)
    mainnet_api_key: str = ""
    mainnet_secret: str = ""

    # --- Reproducibilidad y Hardware ---
    random_seed: int = 42
    n_jobs: int = 4

    def __post_init__(self) -> None:
        if not self.use_testnet:
            raise PermissionError(
                "BLOQUEO DE SEGURIDAD: USE_TESTNET=false está deshabilitado. "
                "Mainnet no está permitido en esta fase del proyecto."
            )
        validate_execution_mode(self.execution_mode, allow_mainnet=self.allow_mainnet)

    @classmethod
    def from_env(cls) -> "QuantConfig":
        """Construye una instancia de configuración leyendo variables de entorno en tiempo de llamada."""
        use_testnet = os.getenv("USE_TESTNET", "true").lower().strip() == "true"
        allow_mainnet = os.getenv("ALLOW_MAINNET", "false").lower().strip() == "true"
        dry_run = os.getenv("DRY_RUN", "true").lower().strip() == "true"
        # Paper Trading es el modo predeterminado (Plan Maestro §1.4)
        execution_mode = os.getenv("EXECUTION_MODE", "paper").lower().strip()

        return cls(
            raw_data_dir=os.getenv("RAW_DATA_DIR", "data/raw"),
            model_artifact_path=os.getenv("MODEL_PATH", "models/quant_brain.joblib"),
            sim_state_path=os.getenv("SIM_STATE_PATH", "data/paper_sim_state.json"),
            trade_log_path=os.getenv("TRADE_LOG_PATH", "data/paper_trades_history.csv"),
            manifests_dir=os.getenv("MANIFESTS_DIR", "data/manifests"),
            experiments_dir=os.getenv("EXPERIMENTS_DIR", "data/experiments"),
            primary_symbol=os.getenv("PRIMARY_SYMBOL", "BTCUSDT"),
            context_symbol=os.getenv("CONTEXT_SYMBOL", "ETHUSDT"),
            ccxt_symbol=os.getenv("CCXT_SYMBOL", "BTC/USDT:USDT"),
            timeframe=os.getenv("TIMEFRAME", "15m"),
            start_year_month=os.getenv("START_YEAR_MONTH", "2024-01"),
            end_year_month=os.getenv("END_YEAR_MONTH", "2026-10"),
            tp_multiple=float(os.getenv("TP_MULTIPLE", "2.0")),
            sl_multiple=float(os.getenv("SL_MULTIPLE", "1.0")),
            horizon_bars=int(os.getenv("HORIZON_BARS", "24")),
            roundtrip_fee=float(os.getenv("ROUNDTRIP_FEE", "0.0010")),
            slippage_bps=float(os.getenv("SLIPPAGE_BPS", "1.5")),
            half_spread_bps=float(os.getenv("HALF_SPREAD_BPS", "0.5")),
            ev_min_threshold=float(os.getenv("EV_MIN_THRESHOLD", "0.12")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.20")),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.03")),
            enable_shorts=os.getenv("ENABLE_SHORTS", "true").lower() == "true",
            max_leverage=int(os.getenv("MAX_LEVERAGE", "5")),
            margin_mode=os.getenv("MARGIN_MODE", "ISOLATED"),
            max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.06")),
            min_notional_usdt=float(os.getenv("MIN_NOTIONAL_USDT", "100.0")),
            min_qty_btc=float(os.getenv("MIN_QTY_BTC", "0.002")),
            initial_paper_balance=float(os.getenv("INITIAL_PAPER_BALANCE", "10000.0")),
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "15")),
            use_testnet=use_testnet,
            dry_run=dry_run,
            allow_mainnet=allow_mainnet,
            execution_mode=execution_mode,
            binance_api_key=os.getenv("BINANCE_TESTNET_API_KEY", ""),
            binance_secret=os.getenv("BINANCE_TESTNET_SECRET", ""),
            mainnet_api_key="",
            mainnet_secret="",
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
            n_jobs=int(os.getenv("LGBM_N_JOBS", "4")),
        )


# Instancia global para importar en cualquier módulo
settings = QuantConfig.from_env()