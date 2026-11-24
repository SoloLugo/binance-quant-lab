"""
Script de entrenamiento y ablación progresiva M0–M4 (Fases 9, 10 y 11).

Elimina definitivamente el bug histórico AUD-03:
- Valida la calidad de datos con `DataQualityGate` antes de entrenar.
- Aísla el Test Final Sellado mediante `TemporalDatasetPartitioner`.
- Selecciona el mejor modelo de forma independiente para LONG y SHORT
  EXCLUSIVAMENTE sobre Walk-Forward Validation (`run_ablation_study`).
- Registra todos los experimentos en `ExperimentTracker` con `test_period="SEALED_NOT_TOUCHED"`.
- JAMÁS accede al Test Final Sellado (`get_sealed_test` no se invoca nunca).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import pandas as pd

from ablation import run_ablation_study
from brain import BenterQuantBrain
from config import settings
from contracts import CostModel
from data_loader import HistoricalDataLoader
from data_quality import DataQualityGate, timeframe_to_timedelta
from experiment_tracker import ExperimentTracker
from pipeline import QuantFeaturePipeline
from splits import TemporalDatasetPartitioner


def _fit_export_brain(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    partitioner: TemporalDatasetPartitioner,
    feature_cols: List[str],
    direction: str,
    calibration_method: str = "isotonic",
) -> BenterQuantBrain:
    """
    Ajusta el adaptador `BenterQuantBrain` exclusivamente sobre el bloque de desarrollo
    `(df_train, df_calib)` purgado y con embargo, sin tocar jamás el Test Final Sellado.
    """
    df_tr_part, df_ca_part, _ = partitioner.get_dev_split()

    X_tr_raw = features.reindex(df_tr_part.index)
    y_tr_raw = outcomes.reindex(df_tr_part.index)["target_tp_before_sl"]
    tr_mask = X_tr_raw[feature_cols].notna().all(axis=1) & y_tr_raw.notna()

    X_ca_raw = features.reindex(df_ca_part.index)
    y_ca_raw = outcomes.reindex(df_ca_part.index)["target_tp_before_sl"]
    ca_mask = X_ca_raw[feature_cols].notna().all(axis=1) & y_ca_raw.notna()

    brain = BenterQuantBrain(
        feature_cols=feature_cols,
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        roundtrip_fee=settings.roundtrip_fee,
        direction=direction,
        calibration_method=calibration_method,
        random_state=settings.random_seed,
    )
    brain.fit_and_calibrate(
        df_train=X_tr_raw.loc[tr_mask],
        y_train=y_tr_raw.loc[tr_mask].astype(int),
        df_calib=X_ca_raw.loc[ca_mask],
        y_calib=y_ca_raw.loc[ca_mask].astype(int),
        purge_bars=settings.horizon_bars,
        enforce_purge=True,
        bar_delta=timeframe_to_timedelta(settings.timeframe),
    )
    return brain


def run_ablation_and_export(
    n_splits: int = 3,
    model_name: str = "lightgbm",
    calibration_method: str = "isotonic",
) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de ablación M0–M4 sobre Walk-Forward Validation
    para LONG y SHORT, registra resultados en `ExperimentTracker` y exporta
    el artefacto compatible sin tocar nunca el Test Final Sellado.
    """
    print("[1/5] Cargando y validando datos históricos con DataQualityGate...")
    loader = HistoricalDataLoader(raw_dir=settings.raw_data_dir)
    df_btc = loader.load_klines(settings.primary_symbol, settings.timeframe)
    df_eth = loader.load_klines(settings.context_symbol, settings.timeframe)
    df_funding = loader.load_funding_rate(settings.primary_symbol)

    gate = DataQualityGate()
    btc_report = gate.validate_klines(
        df_btc, symbol=settings.primary_symbol, timeframe=settings.timeframe
    )
    if not btc_report.passed:
        raise ValueError(
            f"DataQualityGate bloqueó {settings.primary_symbol}: {btc_report.errors}"
        )

    eth_report = gate.validate_klines(
        df_eth, symbol=settings.context_symbol, timeframe=settings.timeframe
    )
    if not eth_report.passed:
        raise ValueError(
            f"DataQualityGate bloqueó {settings.context_symbol}: {eth_report.errors}"
        )

    print("[2/5] Construyendo features causales y etiquetas Triple-Barrier...")
    pipe = QuantFeaturePipeline(
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        horizon_bars=settings.horizon_bars,
    )
    features = pipe.build_features(df_btc, df_eth, df_funding)
    outcomes_long = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="LONG"
    )
    outcomes_short = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="SHORT"
    )

    print("[3/5] Aislando Test Final Sellado con TemporalDatasetPartitioner (purging + embargo)...")
    bar_delta = timeframe_to_timedelta(settings.timeframe)
    partitioner = TemporalDatasetPartitioner(
        df=df_btc,
        purge_bars=settings.horizon_bars,
        embargo_bars=settings.horizon_bars,
        sealed_test_ratio=0.15,
        bar_delta=bar_delta,
        horizon_bars=settings.horizon_bars,
    )

    cost_model = CostModel(
        taker_fee_rate=settings.roundtrip_fee / 2.0,
        half_spread_bps=settings.half_spread_bps,
        slippage_bps=settings.slippage_bps,
        horizon_hours=(settings.horizon_bars * bar_delta.total_seconds()) / 3600.0,
    )
    tracker = ExperimentTracker(experiments_dir=settings.experiments_dir)
    dataset_version = loader.compute_dataframe_sha256(df_btc)[:12]

    print("[4/5] Ejecutando Ablation Study M0–M4 en Walk-Forward Validation (LONG y SHORT)...")
    ablation_long = run_ablation_study(
        df_ohlcv=df_btc,
        features=features,
        outcomes=outcomes_long,
        partitioner=partitioner,
        direction="LONG",
        model_name=model_name,
        calibration_method=calibration_method,
        cost_model=cost_model,
        ev_min_threshold=settings.ev_min_threshold,
        n_splits=n_splits,
        random_state=settings.random_seed,
        tracker=tracker,
        dataset_version=dataset_version,
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        horizon_bars=settings.horizon_bars,
        df_funding=df_funding,
    )

    ablation_short = run_ablation_study(
        df_ohlcv=df_btc,
        features=features,
        outcomes=outcomes_short,
        partitioner=partitioner,
        direction="SHORT",
        model_name=model_name,
        calibration_method=calibration_method,
        cost_model=cost_model,
        ev_min_threshold=settings.ev_min_threshold,
        n_splits=n_splits,
        random_state=settings.random_seed,
        tracker=tracker,
        dataset_version=dataset_version,
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        horizon_bars=settings.horizon_bars,
        df_funding=df_funding,
    )

    # Garantía de integridad: 0 accesos al Test Final Sellado
    assert len(partitioner.sealed_test_access_log) == 0, (
        "Violación crítica: el Test Final Sellado fue accedido durante la ablación."
    )

    print("\n================ ABLATION WALK-FORWARD VALIDATION (LONG) ================")
    print(ablation_long["comparison_table"].to_string(index=False))
    print(f"Mejor configuración LONG (WF Validation): {ablation_long['best_model_name']}")

    print("\n================ ABLATION WALK-FORWARD VALIDATION (SHORT) ===============")
    print(ablation_short["comparison_table"].to_string(index=False))
    print(f"Mejor configuración SHORT (WF Validation): {ablation_short['best_model_name']}")
    print("=========================================================================\n")

    # Selección de columnas para el adaptador BenterQuantBrain (fallback a M1 si ganó M0)
    long_cols = (
        ablation_long["best_feature_cols"]
        if ablation_long["best_feature_cols"]
        else ablation_long["configs"]["M1_Technical"]["feature_cols"]
    )
    short_cols = (
        ablation_short["best_feature_cols"]
        if ablation_short["best_feature_cols"]
        else ablation_short["configs"]["M1_Technical"]["feature_cols"]
    )

    best_brain_long = _fit_export_brain(
        features=features,
        outcomes=outcomes_long,
        partitioner=partitioner,
        feature_cols=long_cols,
        direction="LONG",
        calibration_method=calibration_method,
    )
    best_brain_short = _fit_export_brain(
        features=features,
        outcomes=outcomes_short,
        partitioner=partitioner,
        feature_cols=short_cols,
        direction="SHORT",
        calibration_method=calibration_method,
    )

    os.makedirs(os.path.dirname(settings.model_artifact_path) or "models", exist_ok=True)
    artifact = {
        "brain": best_brain_long,
        "brain_long": best_brain_long,
        "brain_short": best_brain_short,
        "best_model_name": ablation_long["best_model_name"],
        "best_model_name_long": ablation_long["best_model_name"],
        "best_model_name_short": ablation_short["best_model_name"],
        "ablation_results_long": ablation_long["comparison_table"].to_dict(orient="records"),
        "ablation_results_short": ablation_short["comparison_table"].to_dict(orient="records"),
        "experiment_ids_long": ablation_long["experiment_ids"],
        "experiment_ids_short": ablation_short["experiment_ids"],
        "sealed_test_touched": False,
        "pipeline": pipe,
    }
    joblib.dump(artifact, settings.model_artifact_path)
    print(
        f"[5/5] Artefacto exportado en {settings.model_artifact_path} "
        f"(LONG={ablation_long['best_model_name']}, SHORT={ablation_short['best_model_name']}, "
        f"Test Final Sellado INTACTO)."
    )
    return artifact


if __name__ == "__main__":
    run_ablation_and_export()