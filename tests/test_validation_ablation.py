"""
Pruebas reales (sin mocks sustitutivos) para las Fases 9, 10 y 11:
- ExperimentTracker (src/experiment_tracker.py)
- Walk-Forward Validation y Robustez (src/validation.py)
- Ablation Studies M0–M4 e Interpretabilidad (src/ablation.py)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from ablation import (
    ABLATION_ORDER,
    INTERPRETABILITY_CAUSALITY_WARNING,
    compute_feature_importance,
    compute_feature_stability,
    compute_permutation_importance,
    run_ablation_study,
)
from contracts import CostModel, ExperimentRecord
from experiment_tracker import ExperimentTracker
from pipeline import QuantFeaturePipeline
from splits import TemporalDatasetPartitioner
from validation import (
    analyze_by_regime,
    bootstrap_expectancy_ci,
    deflated_sharpe_ratio,
    monte_carlo_drawdown,
    run_walk_forward_validation,
)


def _make_synthetic_market_data(
    n_bars: int = 900,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera datos sintéticos deterministas (900 velas 15m UTC) de BTCUSDT, ETHUSDT
    y Funding Rate 8h con estructura de tendencia/volatilidad realista para pruebas reales.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 00:00:00", periods=n_bars, freq="15min", tz="UTC")

    t = np.arange(n_bars, dtype=float)
    # Régimen cíclico + ruido controlado para asegurar eventos WIN, LOSS y TIMEOUT
    drift = 0.0004 * np.sin(2.0 * np.pi * t / 180.0)
    btc_ret = drift + rng.normal(0.0, 0.0045, size=n_bars)
    btc_close = 60000.0 * np.exp(np.cumsum(btc_ret))
    btc_open = np.empty(n_bars, dtype=float)
    btc_open[0] = btc_close[0]
    btc_open[1:] = btc_close[:-1]

    intrabar_spread = rng.uniform(0.001, 0.005, size=n_bars)
    btc_high = np.maximum(btc_open, btc_close) * (1.0 + intrabar_spread)
    btc_low = np.minimum(btc_open, btc_close) * (1.0 - intrabar_spread)
    btc_vol = rng.uniform(80.0, 450.0, size=n_bars)
    btc_taker_buy = btc_vol * rng.uniform(0.35, 0.65, size=n_bars)

    df_btc = pd.DataFrame(
        {
            "open": btc_open,
            "high": btc_high,
            "low": btc_low,
            "close": btc_close,
            "volume": btc_vol,
            "quote_volume": btc_vol * btc_close,
            "count": rng.integers(500, 3000, size=n_bars).astype(float),
            "taker_buy_volume": btc_taker_buy,
        },
        index=idx,
    )

    eth_ret = 0.85 * btc_ret + rng.normal(0.0, 0.0025, size=n_bars)
    eth_close = 3000.0 * np.exp(np.cumsum(eth_ret))
    eth_open = np.empty(n_bars, dtype=float)
    eth_open[0] = eth_close[0]
    eth_open[1:] = eth_close[:-1]
    eth_high = np.maximum(eth_open, eth_close) * (1.0 + rng.uniform(0.001, 0.004, size=n_bars))
    eth_low = np.minimum(eth_open, eth_close) * (1.0 - rng.uniform(0.001, 0.004, size=n_bars))
    eth_vol = rng.uniform(500.0, 2500.0, size=n_bars)

    df_eth = pd.DataFrame(
        {
            "open": eth_open,
            "high": eth_high,
            "low": eth_low,
            "close": eth_close,
            "volume": eth_vol,
        },
        index=idx,
    )

    funding_idx = pd.date_range(
        idx[0].floor("8h"),
        idx[-1],
        freq="8h",
        tz="UTC",
    )
    df_funding = pd.DataFrame(
        {
            "funding_rate": rng.normal(0.0001, 0.00015, size=len(funding_idx)),
        },
        index=funding_idx,
    )
    return df_btc, df_eth, df_funding


@pytest.fixture(scope="module")
def synthetic_pipeline_bundle():
    df_btc, df_eth, df_funding = _make_synthetic_market_data(n_bars=900, seed=42)
    pipe = QuantFeaturePipeline(tp_mult=2.0, sl_mult=1.0, horizon_bars=24)
    features = pipe.build_features(df_btc, df_eth, df_funding)
    outcomes_long = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="LONG"
    )
    outcomes_short = pipe.compute_triple_barrier_outcomes(
        df_btc, features["atr_pct"], direction="SHORT"
    )
    return df_btc, df_eth, df_funding, features, outcomes_long, outcomes_short


class TestExperimentTracker:
    def test_sequential_ids_persistence_and_exact_reconstruction(self, tmp_path: Path):
        exp_dir = tmp_path / "experiments"
        tracker = ExperimentTracker(experiments_dir=str(exp_dir))

        rec1 = tracker.log_experiment(
            hypothesis="Baseline M0 vs M1 on WF Validation",
            dataset_version="sha256_abc123",
            feature_version="v2.0",
            label_version="v2.0",
            model_name="M0_Baseline:base_rate",
            parameters={"n_splits": 3, "direction": "LONG"},
            train_period="2025-01-01 -> 2025-01-06",
            val_period="2025-01-07 -> 2025-01-09",
            test_period="SEALED_NOT_TOUCHED",
            cost_assumptions=CostModel().to_dict(),
            metrics={"mean_brier": 0.241, "mean_auc": 0.50, "nan_metric": float("nan")},
            random_seed=42,
            result="COMPLETED",
            conclusion="Baseline reference recorded.",
        )

        rec2 = tracker.log_experiment(
            hypothesis="Technical features M1 on WF Validation",
            dataset_version="sha256_abc123",
            feature_version="v2.0",
            label_version="v2.0",
            model_name="M1_Technical:lightgbm",
            parameters={"n_splits": 3, "direction": "LONG"},
            train_period="2025-01-01 -> 2025-01-06",
            val_period="2025-01-07 -> 2025-01-09",
            test_period="SEALED_NOT_TOUCHED",
            cost_assumptions=CostModel().to_dict(),
            metrics={"mean_brier": 0.232, "mean_auc": 0.56},
            random_seed=42,
            result="COMPLETED",
            conclusion="Improved Brier on WF Validation.",
        )

        assert isinstance(rec1, ExperimentRecord)
        assert isinstance(rec2, ExperimentRecord)
        assert rec1.experiment_id == "EXP-0001"
        assert rec2.experiment_id == "EXP-0002"
        assert isinstance(rec1.git_commit, str) and len(rec1.git_commit) > 0

        # Persistencia JSON individual y JSONL
        json1 = exp_dir / "EXP-0001.json"
        json2 = exp_dir / "EXP-0002.json"
        registry = exp_dir / "registry.jsonl"
        assert json1.exists() and json2.exists() and registry.exists()

        lines = [ln for ln in registry.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2

        loaded_1 = tracker.get("EXP-0001")
        loaded_2 = tracker.get("EXP-0002")
        assert loaded_1["experiment_id"] == "EXP-0001"
        assert loaded_1["test_period"] == "SEALED_NOT_TOUCHED"
        assert loaded_1["metrics"]["nan_metric"] is None
        assert loaded_2 == json.loads(json2.read_text(encoding="utf-8"))

        all_recs = tracker.load_all()
        assert [r["experiment_id"] for r in all_recs] == ["EXP-0001", "EXP-0002"]

        # Nueva instancia en el mismo directorio continúa la secuencia en EXP-0003
        tracker_reloaded = ExperimentTracker(experiments_dir=str(exp_dir))
        rec3 = tracker_reloaded.log_experiment(
            hypothesis="Continuation test",
            dataset_version="sha256_abc123",
            feature_version="v2.0",
            label_version="v2.0",
            model_name="M2_TA+Microstructure:lightgbm",
            parameters={},
            train_period="2025-01-01 -> 2025-01-06",
            val_period="2025-01-07 -> 2025-01-09",
            test_period="SEALED_NOT_TOUCHED",
        )
        assert rec3.experiment_id == "EXP-0003"


class TestRobustnessStatistics:
    def test_bootstrap_expectancy_ci_reproducible_and_valid(self):
        r = np.array([1.8, -1.0, 1.9, -1.0, 0.3, 1.7, -1.0, 1.6, 0.5, -1.0, 2.0, -1.0])
        ci_a = bootstrap_expectancy_ci(r, n_resamples=500, alpha=0.05, random_state=42)
        ci_b = bootstrap_expectancy_ci(r, n_resamples=500, alpha=0.05, random_state=42)

        assert ci_a == ci_b
        assert ci_a["n_trades"] == len(r)
        assert ci_a["ci_low"] <= ci_a["mean_r"] <= ci_a["ci_high"]
        assert 0.0 <= ci_a["prob_positive_ev"] <= 1.0

        empty_ci = bootstrap_expectancy_ci(np.array([]))
        assert empty_ci["n_trades"] == 0 and empty_ci["mean_r"] == 0.0

    def test_monte_carlo_drawdown_quantiles_and_ruin(self):
        r = np.array([-1.0, -1.0, 1.8, -1.0, 1.9, -1.0, -1.0, 1.7, -1.0, 0.2] * 4)
        mc_a = monte_carlo_drawdown(
            r, risk_per_trade=0.02, n_simulations=500, ruin_threshold_pct=10.0, random_state=42
        )
        mc_b = monte_carlo_drawdown(
            r, risk_per_trade=0.02, n_simulations=500, ruin_threshold_pct=10.0, random_state=42
        )

        assert mc_a == mc_b
        assert 0.0 <= mc_a["median_max_dd_pct"] <= mc_a["p95_max_dd_pct"] <= mc_a["p99_max_dd_pct"]
        assert 0.0 <= mc_a["prob_ruin_pct"] <= 100.0

    def test_deflated_sharpe_ratio_penalizes_multiple_trials(self):
        rng = np.random.default_rng(123)
        returns = rng.normal(loc=0.15, scale=1.0, size=120)

        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1, benchmark_sharpe=0.0)
        dsr_5 = deflated_sharpe_ratio(returns, n_trials=5, benchmark_sharpe=0.0)
        dsr_50 = deflated_sharpe_ratio(returns, n_trials=50, benchmark_sharpe=0.0)

        assert dsr_1["n_obs"] == 120
        assert dsr_1["observed_sharpe"] == pytest.approx(dsr_5["observed_sharpe"])
        assert dsr_1["expected_max_sharpe"] < dsr_5["expected_max_sharpe"] < dsr_50["expected_max_sharpe"]
        assert dsr_1["dsr_probability"] >= dsr_5["dsr_probability"] >= dsr_50["dsr_probability"]
        assert 0.0 <= dsr_50["dsr_probability"] <= 1.0

    def test_analyze_by_regime_exact_breakdown(self):
        idx = pd.date_range("2025-01-01 00:00:00", periods=6, freq="15min", tz="UTC")
        trades = pd.DataFrame(
            {
                "signal_time": [idx[0], idx[1], idx[3], idx[4]],
                "realized_r_net": [2.0, -1.0, 1.5, 1.5],
                "net_pnl_usdt": [200.0, -100.0, 150.0, 150.0],
            }
        )
        regimes = pd.Series(
            ["bull", "bull", "bear", "bear", "bear", "bull"],
            index=idx,
        )

        res = analyze_by_regime(trades, regimes)
        assert set(res.keys()) == {"bull", "bear"}
        assert res["bull"]["n_trades"] == 2
        assert res["bull"]["win_rate_pct"] == pytest.approx(50.0)
        assert res["bull"]["expectancy_r"] == pytest.approx(0.5)
        assert res["bull"]["net_pnl"] == pytest.approx(100.0)
        assert res["bull"]["profit_factor"] == pytest.approx(2.0)

        assert res["bear"]["n_trades"] == 2
        assert res["bear"]["win_rate_pct"] == pytest.approx(100.0)
        assert res["bear"]["expectancy_r"] == pytest.approx(1.5)
        assert res["bear"]["net_pnl"] == pytest.approx(300.0)
        assert math.isinf(res["bear"]["profit_factor"])


class TestWalkForwardAndAblation:
    def test_walk_forward_validation_never_touches_sealed_test(self, synthetic_pipeline_bundle):
        df_btc, _, df_funding, features, outcomes_long, _ = synthetic_pipeline_bundle
        partitioner = TemporalDatasetPartitioner(
            df=df_btc,
            purge_bars=24,
            embargo_bars=24,
            sealed_test_ratio=0.15,
            bar_delta=pd.Timedelta(minutes=15),
            horizon_bars=24,
        )
        ta_cols = [c for c in features.columns if c.startswith("ta_")]

        res_a = run_walk_forward_validation(
            df_ohlcv=df_btc,
            features=features,
            outcomes=outcomes_long,
            partitioner=partitioner,
            feature_cols=ta_cols,
            direction="LONG",
            model_name="lightgbm",
            calibration_method="isotonic",
            cost_model=CostModel(),
            ev_min_threshold=0.05,
            n_splits=3,
            random_state=42,
            df_funding=df_funding,
        )
        res_b = run_walk_forward_validation(
            df_ohlcv=df_btc,
            features=features,
            outcomes=outcomes_long,
            partitioner=partitioner,
            feature_cols=ta_cols,
            direction="LONG",
            model_name="lightgbm",
            calibration_method="isotonic",
            cost_model=CostModel(),
            ev_min_threshold=0.05,
            n_splits=3,
            random_state=42,
            df_funding=df_funding,
        )

        # CERO accesos al Test Final Sellado
        assert partitioner.sealed_test_access_log == []

        assert len(res_a["fold_metrics"]) == 3
        summary = res_a["summary"]
        for key in (
            "mean_brier",
            "std_brier",
            "mean_log_loss",
            "mean_auc",
            "mean_ece",
            "total_trades",
            "mean_expectancy_r",
            "positive_ev_fold_ratio",
            "bootstrap_ci",
            "monte_carlo_dd",
            "dsr",
        ):
            assert key in summary

        # Reproducibilidad exacta
        assert summary["mean_brier"] == pytest.approx(res_b["summary"]["mean_brier"])
        assert summary["mean_log_loss"] == pytest.approx(res_b["summary"]["mean_log_loss"])
        assert summary["total_trades"] == res_b["summary"]["total_trades"]

    @pytest.mark.parametrize("direction", ["LONG", "SHORT"])
    def test_ablation_study_m0_to_m4_sealed_intact_and_interpretability(
        self,
        synthetic_pipeline_bundle,
        tmp_path: Path,
        direction: str,
    ):
        df_btc, _, df_funding, features, outcomes_long, outcomes_short = synthetic_pipeline_bundle
        outcomes = outcomes_long if direction == "LONG" else outcomes_short

        partitioner = TemporalDatasetPartitioner(
            df=df_btc,
            purge_bars=24,
            embargo_bars=24,
            sealed_test_ratio=0.15,
            bar_delta=pd.Timedelta(minutes=15),
            horizon_bars=24,
        )
        tracker = ExperimentTracker(experiments_dir=str(tmp_path / f"exp_{direction.lower()}"))

        ablation = run_ablation_study(
            df_ohlcv=df_btc,
            features=features,
            outcomes=outcomes,
            partitioner=partitioner,
            direction=direction,
            model_name="lightgbm",
            calibration_method="isotonic",
            cost_model=CostModel(),
            ev_min_threshold=0.05,
            n_splits=3,
            random_state=42,
            tracker=tracker,
            df_funding=df_funding,
        )

        # Test Final Sellado 100% intacto
        assert partitioner.sealed_test_access_log == []

        # Se evaluaron las 5 configuraciones progresivas M0–M4
        assert list(ablation["results"].keys()) == list(ABLATION_ORDER)
        assert ablation["best_model_name"] in ABLATION_ORDER
        assert len(ablation["comparison_table"]) == 5
        assert len(ablation["experiment_ids"]) == 5
        assert isinstance(ablation["causality_warning"], str)
        assert len(ablation["causality_warning"].strip()) > 0
        assert ablation["causality_warning"] == INTERPRETABILITY_CAUSALITY_WARNING

        logged = tracker.load_all()
        assert len(logged) == 5
        assert all(rec["test_period"] == "SEALED_NOT_TOUCHED" for rec in logged)

        # Interpretabilidad y estabilidad de features sobre M4_Full_Context
        m4_artifacts = ablation["results"]["M4_Full_Context"]["fold_artifacts"]
        m4_cols = ablation["configs"]["M4_Full_Context"]["feature_cols"]

        native_dfs = [
            compute_feature_importance(art["model"], m4_cols)
            for art in m4_artifacts
        ]
        assert len(native_dfs) == 3
        for df_imp in native_dfs:
            assert list(df_imp.columns) == ["feature", "importance", "importance_norm", "rank"]
            assert len(df_imp) == len(m4_cols)

        perm_dfs_run1 = [
            compute_permutation_importance(
                art["model"],
                art["X_val"],
                art["y_val"],
                m4_cols,
                n_repeats=3,
                random_state=42,
            )
            for art in m4_artifacts
        ]
        perm_dfs_run2 = [
            compute_permutation_importance(
                art["model"],
                art["X_val"],
                art["y_val"],
                m4_cols,
                n_repeats=3,
                random_state=42,
            )
            for art in m4_artifacts
        ]

        # Reproducibilidad exacta de permutation importance
        for df1, df2 in zip(perm_dfs_run1, perm_dfs_run2):
            pd.testing.assert_frame_equal(df1, df2)
            assert list(df1.columns) == ["feature", "importance_mean", "importance_std", "rank"]
            assert len(df1) == len(m4_cols)

        stability = compute_feature_stability(perm_dfs_run1, top_k=5)
        assert stability["n_folds"] == 3
        assert stability["top_k"] == 5
        assert -1.0 <= stability["mean_spearman_corr"] <= 1.0
        assert 0.0 <= stability["mean_top_k_overlap"] <= 1.0
        assert len(stability["consensus_top_features"]) == min(5, len(m4_cols))
        assert isinstance(stability["causality_warning"], str)
        assert len(stability["causality_warning"].strip()) > 0
        assert stability["causality_warning"] == INTERPRETABILITY_CAUSALITY_WARNING
