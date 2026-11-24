import joblib
import numpy as np
import pandas as pd
import pytest

from brain import BenterQuantBrain
from pipeline import QuantFeaturePipeline


def _make_synthetic_ohlcv(
    rng: np.random.Generator,
    index: pd.DatetimeIndex,
    base_price: float,
    volatility: float = 0.0045,
) -> pd.DataFrame:
    """Genera un DataFrame OHLCV sintético determinista con velas válidas y volúmenes positivos."""
    n = len(index)
    log_rets = rng.normal(loc=0.0, scale=volatility, size=n)
    closes = base_price * np.exp(np.cumsum(log_rets))
    opens = np.empty(n, dtype=float)
    opens[0] = base_price
    opens[1:] = closes[:-1]

    wick_up = np.abs(rng.normal(loc=0.0015, scale=0.001, size=n)) * closes
    wick_dn = np.abs(rng.normal(loc=0.0015, scale=0.001, size=n)) * closes
    highs = np.maximum(opens, closes) + wick_up
    lows = np.maximum(1e-6, np.minimum(opens, closes) - wick_dn)

    volume = rng.uniform(50.0, 500.0, size=n)
    buy_ratio = rng.uniform(0.3, 0.7, size=n)
    taker_buy_volume = volume * buy_ratio
    quote_volume = volume * closes
    count = rng.integers(500, 5000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "quote_volume": quote_volume,
            "count": count,
            "taker_buy_volume": taker_buy_volume,
        },
        index=index,
    )


@pytest.fixture
def sample_ohlcv_and_funding():
    """
    Genera DataFrames sintéticos deterministas (seed=42, 400 velas 15m UTC) de
    BTCUSDT, ETHUSDT y Funding Rate (cada 8h) con OHLC válidos y volúmenes positivos.
    """
    rng = np.random.default_rng(42)
    idx_15m = pd.date_range("2026-01-01 00:00:00", periods=400, freq="15min", tz="UTC")

    df_btc = _make_synthetic_ohlcv(rng, idx_15m, base_price=70000.0, volatility=0.0045)
    df_eth = _make_synthetic_ohlcv(rng, idx_15m, base_price=3500.0, volatility=0.0055)

    idx_funding = pd.date_range(
        start=idx_15m[0],
        end=idx_15m[-1],
        freq="8h",
        tz="UTC",
    )
    funding_rates = rng.normal(loc=0.0001, scale=0.00005, size=len(idx_funding))
    df_funding = pd.DataFrame({"funding_rate": funding_rates}, index=idx_funding)

    return df_btc, df_eth, df_funding


@pytest.fixture
def trained_model_artifact(tmp_path, sample_ohlcv_and_funding) -> str:
    """
    Entrena un BenterQuantBrain real (pequeño, rápido y determinista) sobre
    sample_ohlcv_and_funding y lo guarda con joblib.dump en tmp_path / 'quant_brain.joblib'.
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    features = pipe.build_features(df_btc, df_eth, df_funding)
    outcomes_long = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="LONG"
    )
    outcomes_short = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="SHORT"
    )

    feature_cols = [
        c for c in features.columns
        if c.startswith(("ta_", "sm_", "deriv_", "ctx_"))
    ]

    def _fit_brain(outcomes: pd.DataFrame, direction: str) -> BenterQuantBrain:
        ds = features.copy()
        ds["target"] = outcomes["target_tp_before_sl"]
        ds = ds.dropna()

        n = len(ds)
        i_train = int(n * 0.70)
        df_train = ds.iloc[:i_train]
        y_train = df_train["target"].astype(int)
        df_calib = ds.iloc[i_train:]
        y_calib = df_calib["target"].astype(int)

        brain = BenterQuantBrain(
            feature_cols=feature_cols,
            tp_mult=2.0,
            sl_mult=1.0,
            roundtrip_fee=0.0010,
            direction=direction,
        )
        # Configuración pequeña, rápida y determinista para pruebas unitarias/E2E
        brain.base_clf.set_params(
            n_estimators=20,
            max_depth=3,
            num_leaves=8,
            min_child_samples=10,
            random_state=42,
            n_jobs=1,
        )
        brain.fit_and_calibrate(df_train, y_train, df_calib, y_calib, purge_bars=24)
        return brain

    brain_long = _fit_brain(outcomes_long, "LONG")
    brain_short = _fit_brain(outcomes_short, "SHORT")

    artifact_path = tmp_path / "quant_brain.joblib"
    joblib.dump(
        {
            "brain": brain_long,
            "brain_long": brain_long,
            "brain_short": brain_short,
            "best_model_name": "M4_Full_ContextEngine",
            "pipeline": pipe,
        },
        artifact_path,
    )
    return str(artifact_path)
