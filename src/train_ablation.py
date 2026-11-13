import os
import joblib
import pandas as pd
from config import settings
from data_loader import HistoricalDataLoader
from pipeline import QuantFeaturePipeline
from brain import BenterQuantBrain


def _split_with_purge(dataset: pd.DataFrame, purge_bars: int = 24):
    """
    División temporal estricta con ventana de purga (embargo = horizon_bars)
    entre Entrenamiento (70%), Calibración (15%) y Test Fuera de Muestra (15%)
    para eliminar por completo el solapamiento de etiquetas Triple-Barrier.
    """
    n = len(dataset)
    i_train = int(n * 0.70)
    i_calib = int(n * 0.85)

    df_train = dataset.iloc[:i_train]
    y_train = df_train["target"].astype(int)

    calib_start = min(i_train + purge_bars, i_calib)
    df_calib = dataset.iloc[calib_start:i_calib]
    y_calib = df_calib["target"].astype(int)

    test_start = min(i_calib + purge_bars, n)
    df_test = dataset.iloc[test_start:]
    y_test = df_test["target"].astype(int)

    return df_train, y_train, df_calib, y_calib, df_test, y_test


def run_ablation_and_export():
    print("[1/4] Cargando datos históricos de BTCUSDT, ETHUSDT y Funding Rate...")
    loader = HistoricalDataLoader(raw_dir=settings.raw_data_dir)
    df_btc = loader.load_klines(settings.primary_symbol, settings.timeframe)
    df_eth = loader.load_klines(settings.context_symbol, settings.timeframe)
    df_funding = loader.load_funding_rate(settings.primary_symbol)

    print("[2/4] Ejecutando Context Engine, Feature Engine y Triple-Barrier Labeling...")
    pipe = QuantFeaturePipeline(
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        horizon_bars=settings.horizon_bars
    )
    features = pipe.build_features(df_btc, df_eth, df_funding)
    outcomes_long = pipe.compute_triple_barrier_outcomes(df_btc, features["atr_pct"], direction="LONG")
    outcomes_short = pipe.compute_triple_barrier_outcomes(df_btc, features["atr_pct"], direction="SHORT")

    dataset_long = features.copy()
    dataset_long["target"] = outcomes_long["target_tp_before_sl"]
    dataset_long["realized_r_gross"] = outcomes_long["realized_r_gross"]
    dataset_long["exit_bars"] = outcomes_long["exit_bars"]
    dataset_long = dataset_long.dropna()

    df_train, y_train, df_calib, y_calib, df_test, y_test = _split_with_purge(
        dataset_long, purge_bars=settings.horizon_bars
    )

    # Separación de columnas por capa arquitectónica
    ta_cols = [c for c in features.columns if c.startswith("ta_")]
    sm_cols = [c for c in features.columns if c.startswith("sm_")]
    deriv_cols = [c for c in features.columns if c.startswith("deriv_")]
    ctx_cols = [c for c in features.columns if c.startswith("ctx_")]

    models_config = {
        "M1_Technical": ta_cols,
        "M2_TA+SmartMoney": ta_cols + sm_cols,
        "M3_TA+SM+Derivatives": ta_cols + sm_cols + deriv_cols,
        "M4_Full_ContextEngine": ta_cols + sm_cols + deriv_cols + ctx_cols,
    }

    print("\n[3/4] Entrenando, Calibrando y Evaluando Modelos Fuera de Muestra (Secuencial Sin Solapamiento)...")
    results = {}
    trained_brains = {}

    for name, cols in models_config.items():
        brain = BenterQuantBrain(
            feature_cols=cols,
            tp_mult=settings.tp_multiple,
            sl_mult=settings.sl_multiple,
            roundtrip_fee=settings.roundtrip_fee,
            direction="LONG"
        )
        brain.fit_and_calibrate(df_train, y_train, df_calib, y_calib, purge_bars=settings.horizon_bars)
        metrics = brain.evaluate_model(df_test, y_test, ev_min_threshold=0.10, non_overlapping=True)
        results[name] = metrics
        trained_brains[name] = brain

    results_df = pd.DataFrame(results).T
    print("\n================ RESULTADOS OUT-OF-SAMPLE LONG (BILL BENTER ABLATION) ================")
    print(results_df.to_string())
    print("======================================================================================\n")

    # Selección dinámica del mejor modelo (prioriza modelos con >=5 trades, mayor Total_Net_R y menor Brier)
    eligible = [
        name for name, m in results.items()
        if m["Trades"] >= 5 and m["Total_Net_R"] > 0
    ]
    if eligible:
        best_model_name = max(
            eligible,
            key=lambda k: (-results[k]["Brier"], results[k]["AUC"], results[k]["Total_Net_R"])
        )
    else:
        best_model_name = max(results.keys(), key=lambda k: (results[k]["AUC"], -results[k]["Brier"]))

    best_brain_long = trained_brains[best_model_name]

    # Entrenamos también el motor gemelo para dirección SHORT con las mismas features del mejor modelo
    dataset_short = features.copy()
    dataset_short["target"] = outcomes_short["target_tp_before_sl"]
    dataset_short["realized_r_gross"] = outcomes_short["realized_r_gross"]
    dataset_short["exit_bars"] = outcomes_short["exit_bars"]
    dataset_short = dataset_short.dropna()

    df_tr_s, y_tr_s, df_ca_s, y_ca_s, df_te_s, y_te_s = _split_with_purge(
        dataset_short, purge_bars=settings.horizon_bars
    )
    best_brain_short = BenterQuantBrain(
        feature_cols=best_brain_long.feature_cols,
        tp_mult=settings.tp_multiple,
        sl_mult=settings.sl_multiple,
        roundtrip_fee=settings.roundtrip_fee,
        direction="SHORT"
    )
    best_brain_short.fit_and_calibrate(df_tr_s, y_tr_s, df_ca_s, y_ca_s, purge_bars=settings.horizon_bars)
    short_metrics = best_brain_short.evaluate_model(df_te_s, y_te_s, ev_min_threshold=0.10, non_overlapping=True)
    print(f"Modelo Gemelo SHORT ({best_model_name}) Out-of-Sample -> {short_metrics}\n")

    os.makedirs(os.path.dirname(settings.model_artifact_path) or "models", exist_ok=True)
    joblib.dump(
        {
            "brain": best_brain_long,
            "brain_long": best_brain_long,
            "brain_short": best_brain_short,
            "best_model_name": best_model_name,
            "ablation_results_long": results,
            "ablation_results_short": short_metrics,
            "pipeline": pipe,
        },
        settings.model_artifact_path
    )
    print(
        f"[4/4] Artefacto de producción guardado exitosamente en "
        f"{settings.model_artifact_path} ({best_model_name} LONG + SHORT)"
    )


if __name__ == "__main__":
    run_ablation_and_export()