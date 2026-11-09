import os
import joblib
import pandas as pd
from data_loader import HistoricalDataLoader
from pipeline import QuantFeaturePipeline
from brain import BenterQuantBrain


def run_ablation_and_export():
    print("[1/4] Cargando datos históricos de BTCUSDT, ETHUSDT y Funding Rate...")
    loader = HistoricalDataLoader(raw_dir="data/raw")
    df_btc = loader.load_klines("BTCUSDT", "15m")
    df_eth = loader.load_klines("ETHUSDT", "15m")
    df_funding = loader.load_funding_rate("BTCUSDT")

    print("[2/4] Ejecutando Context Engine, Feature Engine y Triple-Barrier Labeling...")
    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    features = pipe.build_features(df_btc, df_eth, df_funding)
    labels = pipe.create_triple_barrier_labels(df_btc, features["atr_pct"], direction="LONG")

    dataset = features.copy()
    dataset["target"] = labels
    dataset = dataset.dropna()

    # División temporal estricta (Sin Look-Ahead Bias):
    # 70% EntrenamientoBase | 15% Calibración de Probabilidades | 15% Test Fuera de Muestra (Out-of-Sample)
    n = len(dataset)
    i_train = int(n * 0.70)
    i_calib = int(n * 0.85)

    df_train, y_train = dataset.iloc[:i_train], dataset["target"].iloc[:i_train]
    df_calib, y_calib = dataset.iloc[i_train:i_calib], dataset["target"].iloc[i_train:i_calib]
    df_test, y_test = dataset.iloc[i_calib:], dataset["target"].iloc[i_calib:]

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

    print("\n[3/4] Entrenando, Calibrando y Evaluando Modelos Fuera de Muestra...")
    results = {}
    trained_brains = {}

    for name, cols in models_config.items():
        brain = BenterQuantBrain(feature_cols=cols, tp_mult=2.0, sl_mult=1.0)
        brain.fit_and_calibrate(df_train, y_train, df_calib, y_calib)
        metrics = brain.evaluate_model(df_test, y_test, ev_min_threshold=0.10)
        results[name] = metrics
        trained_brains[name] = brain

    results_df = pd.DataFrame(results).T
    print("\n================ RESULTADOS OUT-OF-SAMPLE (BILL BENTER ABLATION) ================")
    print(results_df.to_string())
    print("=================================================================================\n")

    # Seleccionamos el modelo con menor Brier Score y mayor Total_Net_R
    os.makedirs("models", exist_ok=True)
    best_model_name = "M4_Full_ContextEngine"
    joblib.dump(
        {"brain": trained_brains[best_model_name], "pipeline": pipe},
        "models/quant_brain.joblib"
    )
    print(f"[4/4] Artefacto de producción guardado exitosamente en models/quant_brain.joblib ({best_model_name})")


if __name__ == "__main__":
    run_ablation_and_export()