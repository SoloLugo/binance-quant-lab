"""
Contratos compartidos (esquemas de datos e interfaces) entre las capas de Binance Quant Lab.
Módulo compartido: requiere aprobación del Coordinador antes de modificarse.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DataQualityReport:
    """Resultado estructurado del Data Quality Gate (Fase 1)."""
    symbol: str
    timeframe: str
    row_count: int
    duplicate_count: int
    missing_bars_count: int
    invalid_ohlc_count: int
    negative_volume_count: int
    invalid_price_count: int
    out_of_order_count: int
    extreme_outlier_count: int
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gap_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataManifest:
    """Manifiesto reproducible y versionado de un dataset (Fase 1)."""
    dataset_version: str
    symbol: str
    timeframe: str
    source: str
    start_timestamp: str
    end_timestamp: str
    row_count: int
    downloaded_at: str
    schema_version: str
    dataset_hash: str
    file_hashes: Dict[str, str] = field(default_factory=dict)
    transformations: List[str] = field(default_factory=list)
    quality_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelSpec:
    """Especificación versionada del Label Engine (Fase 2)."""
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    horizon_bars: int = 24
    entry_mode: str = "next_open"
    label_version: str = "v2.0"
    exclude_ambiguous: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostModel:
    """Modelo explícito y configurable de costes de ejecución (Fase 7)."""
    taker_fee_rate: float = 0.0005       # 0.05% por lado
    maker_fee_rate: float = 0.0002       # 0.02% por lado
    half_spread_bps: float = 0.5         # 0.5 bps por lado (~1 bp spread completo)
    slippage_bps: float = 1.5            # 1.5 bps por ejecución a mercado
    expected_funding_rate_8h: float = 0.0001
    horizon_hours: float = 6.0

    @property
    def roundtrip_fee_rate(self) -> float:
        return 2.0 * self.taker_fee_rate

    @property
    def roundtrip_slippage_rate(self) -> float:
        return 2.0 * (self.slippage_bps / 10000.0)

    @property
    def roundtrip_spread_rate(self) -> float:
        return 2.0 * (self.half_spread_bps / 10000.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitSpec:
    """Especificación de partición temporal con purging, embargo y Test Final Sellado."""
    train_start: str
    train_end: str
    calib_start: str
    calib_end: str
    val_start: str
    val_end: str
    sealed_test_start: str
    sealed_test_end: str
    purge_bars: int
    embargo_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskDecision:
    """Decisión independiente emitida por el Risk Engine (Fase 12)."""
    approved: bool
    qty_btc: float
    risk_fraction: float
    reason: str
    effective_leverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRecord:
    """Registro trazable de experimento (Fase 11)."""
    experiment_id: str
    hypothesis: str
    dataset_version: str
    feature_version: str
    label_version: str
    model_name: str
    parameters: Dict[str, Any]
    train_period: str
    val_period: str
    test_period: str
    cost_assumptions: Dict[str, Any]
    metrics: Dict[str, Any]
    git_commit: str
    random_seed: int
    result: str
    conclusion: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
