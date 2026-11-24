"""
Tests unitarios y de integración para EV Engine y Cost Model (Fase 7).
"""
import numpy as np
import pandas as pd
import pytest

from contracts import CostModel, LabelSpec
from ev_engine import EVEngine, TimeoutEconomics, estimate_timeout_economics
from label_engine import LabelEngine


EXPECTED_COLUMNS = [
    "p_raw",
    "p_calibrated",
    "p_sl",
    "p_timeout",
    "expected_timeout_r",
    "gross_win_r",
    "gross_loss_r",
    "fee_in_r",
    "spread_in_r",
    "slippage_in_r",
    "funding_in_r",
    "total_cost_in_r",
    "ev_gross_r",
    "ev_net_r",
    "reward_risk_ratio_net",
    "accepted",
    "rejection_reason",
]


def test_cost_breakdown_exact_long_vs_short():
    """Verifica el desglose exacto de costes en unidades 1R y el signo de funding en LONG vs SHORT."""
    cm = CostModel(
        taker_fee_rate=0.0005,        # roundtrip = 0.0010
        maker_fee_rate=0.0002,
        half_spread_bps=0.5,          # roundtrip = 0.0001
        slippage_bps=1.5,             # roundtrip = 0.0003
        expected_funding_rate_8h=0.0004,
        horizon_hours=6.0,            # 6/8 * 0.0004 = 0.0003
    )
    atr_pct = 0.01  # 1% ATR
    sl_mult = 1.0
    tp_mult = 2.0
    r_dist_pct = atr_pct * sl_mult  # 0.01

    engine_long = EVEngine(tp_mult=tp_mult, sl_mult=sl_mult, cost_model=cm, direction="LONG")
    engine_short = EVEngine(tp_mult=tp_mult, sl_mult=sl_mult, cost_model=cm, direction="SHORT")

    res_long = engine_long.evaluate(p_calibrated=0.50, atr_pct=atr_pct)
    res_short = engine_short.evaluate(p_calibrated=0.50, atr_pct=atr_pct)

    assert list(res_long.columns) == EXPECTED_COLUMNS
    assert list(res_short.columns) == EXPECTED_COLUMNS

    expected_fee_r = 0.0010 / r_dist_pct       # 0.10
    expected_spread_r = 0.0001 / r_dist_pct    # 0.01
    expected_slippage_r = 0.0003 / r_dist_pct  # 0.03
    expected_funding_long_r = (1.0 * 0.0004 * (6.0 / 8.0)) / r_dist_pct   # +0.03
    expected_funding_short_r = (-1.0 * 0.0004 * (6.0 / 8.0)) / r_dist_pct # -0.03

    assert res_long.loc[0, "fee_in_r"] == pytest.approx(expected_fee_r, rel=1e-12)
    assert res_long.loc[0, "spread_in_r"] == pytest.approx(expected_spread_r, rel=1e-12)
    assert res_long.loc[0, "slippage_in_r"] == pytest.approx(expected_slippage_r, rel=1e-12)
    assert res_long.loc[0, "funding_in_r"] == pytest.approx(expected_funding_long_r, rel=1e-12)
    assert res_long.loc[0, "total_cost_in_r"] == pytest.approx(
        expected_fee_r + expected_spread_r + expected_slippage_r + expected_funding_long_r,
        rel=1e-12,
    )

    # En SHORT con funding > 0, el funding es un crédito (coste negativo)
    assert res_short.loc[0, "funding_in_r"] == pytest.approx(expected_funding_short_r, rel=1e-12)
    assert res_short.loc[0, "funding_in_r"] < 0.0
    assert res_long.loc[0, "funding_in_r"] == pytest.approx(-res_short.loc[0, "funding_in_r"], rel=1e-12)
    assert res_short.loc[0, "total_cost_in_r"] < res_long.loc[0, "total_cost_in_r"]
    assert res_short.loc[0, "ev_net_r"] > res_long.loc[0, "ev_net_r"]


def test_timeout_modeling_neutral_beats_penalized_minus_one_r():
    """
    Modelar TIMEOUT como evento neutro (0R) o con su retorno empírico real produce un EV neto
    estrictamente mayor que penalizarlo artificialmente como -1R (timeout_share = 0).
    """
    cm = CostModel(
        taker_fee_rate=0.0005,
        half_spread_bps=0.5,
        slippage_bps=1.5,
        expected_funding_rate_8h=0.0001,
        horizon_hours=6.0,
    )
    engine = EVEngine(tp_mult=2.0, sl_mult=1.0, cost_model=cm, ev_min_threshold=0.12)

    p_cal = pd.Series([0.38, 0.42, 0.48], index=pd.date_range("2025-01-01", periods=3, freq="15min"))
    atr_pct = pd.Series([0.01, 0.01, 0.01], index=p_cal.index)

    # Escenario A: TIMEOUT penalizado como SL (-1R), i.e., timeout_share = 0
    te_penalized = TimeoutEconomics(timeout_share_given_not_tp=0.0, mean_timeout_r=0.0, n_timeout_samples=0)
    res_penalized = engine.evaluate(p_calibrated=p_cal, atr_pct=atr_pct, timeout_economics=te_penalized)

    # Escenario B: 40% de los no-WIN son TIMEOUT con retorno bruto medio neutro (0.0R)
    te_neutral = TimeoutEconomics(timeout_share_given_not_tp=0.40, mean_timeout_r=0.0, n_timeout_samples=100)
    res_neutral = engine.evaluate(p_calibrated=p_cal, atr_pct=atr_pct, timeout_economics=te_neutral)

    assert (res_neutral.index == p_cal.index).all()
    assert (res_neutral["ev_gross_r"] > res_penalized["ev_gross_r"]).all()
    assert (res_neutral["ev_net_r"] > res_penalized["ev_net_r"]).all()

    # Verificar diferencia exacta: delta EV = (1 - p_cal) * 0.40 * (0.0 - (-1.0)) = 0.40 * (1 - p_cal)
    expected_delta = 0.40 * (1.0 - p_cal)
    np.testing.assert_allclose(
        res_neutral["ev_net_r"] - res_penalized["ev_net_r"],
        expected_delta,
        rtol=1e-12,
        atol=1e-12,
    )


def test_explicit_p_sl_and_p_timeout_override_timeout_economics():
    """Si se proveen p_sl y p_timeout explícitos, EVEngine los utiliza en lugar de timeout_economics."""
    cm = CostModel(taker_fee_rate=0.0, half_spread_bps=0.0, slippage_bps=0.0, expected_funding_rate_8h=0.0)
    engine = EVEngine(tp_mult=2.0, sl_mult=1.0, cost_model=cm, ev_min_threshold=0.10)

    te = TimeoutEconomics(timeout_share_given_not_tp=0.90, mean_timeout_r=0.25, n_timeout_samples=50)
    res = engine.evaluate(
        p_calibrated=0.40,
        atr_pct=0.01,
        p_raw=0.45,
        p_sl=0.30,
        p_timeout=0.30,
        timeout_economics=te,
    )

    assert res.loc[0, "p_raw"] == pytest.approx(0.45)
    assert res.loc[0, "p_sl"] == pytest.approx(0.30)
    assert res.loc[0, "p_timeout"] == pytest.approx(0.30)
    assert res.loc[0, "expected_timeout_r"] == pytest.approx(0.25)

    # EV = 0.40 * 2.0 + 0.30 * (-1.0) + 0.30 * 0.25 = 0.80 - 0.30 + 0.075 = 0.575
    assert res.loc[0, "ev_gross_r"] == pytest.approx(0.575, rel=1e-12)
    assert res.loc[0, "ev_net_r"] == pytest.approx(0.575, rel=1e-12)
    assert bool(res.loc[0, "accepted"]) is True
    assert res.loc[0, "rejection_reason"] == "ACCEPTED"


def test_estimate_timeout_economics_on_real_label_engine():
    """
    Ejecuta LabelEngine real sobre un OHLC sintético controlado que genera WIN, LOSS,
    TIMEOUT, AMBIGUOUS e INCOMPLETE, verificando que estimate_timeout_economics ignora
    AMBIGUOUS e INCOMPLETE y calcula exactamente P(TIMEOUT | not WIN) y E[r | TIMEOUT].
    """
    # horizon_bars = 2, tp_mult = 2.0, sl_mult = 1.0, atr_pct = 0.01 -> sl_dist = 1.0, tp_dist = 2.0 para entry=100
    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=2, entry_mode="next_open", exclude_ambiguous=True)
    label_engine = LabelEngine(spec=spec)

    # Construimos 12 barras donde controlamos los resultados de los índices 0, 3, 6, 9:
    # Con horizon=2:
    # - i=0: entry=open[1]=100, TP=102, SL=99. Bar 1 toca TP (high=102.5, low=99.5) -> WIN
    # - i=1: entry=open[2]=100, TP=102, SL=99. Bar 2 toca SL (high=100.5, low=98.5) -> LOSS
    # - i=2: entry=open[3]=100, TP=102, SL=99. Bar 3 toca ambos (high=103.0, low=98.0) -> AMBIGUOUS
    # - i=3: entry=open[4]=100, TP=102, SL=99. Bars 4,5 no tocan ni TP ni SL; close[5]=100.5 -> TIMEOUT (+0.5R)
    # - i=4: entry=open[5]=100, TP=102, SL=99. Bars 5,6 no tocan ni TP ni SL; close[6]=99.7 -> TIMEOUT (-0.3R)
    # - i=5: entry=open[6]=100, TP=102, SL=99. Bar 6 no toca, bar 7 toca SL -> LOSS
    # - i=6, 7: INCOMPLETE (últimas horizon=2 barras)
    rows = [
        # idx 0 (signal bar)
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        # idx 1: WIN for i=0
        {"open": 100.0, "high": 102.5, "low": 99.5, "close": 101.0},
        # idx 2: LOSS for i=1
        {"open": 100.0, "high": 100.5, "low": 98.5, "close": 99.0},
        # idx 3: AMBIGUOUS for i=2
        {"open": 100.0, "high": 103.0, "low": 98.0, "close": 100.0},
        # idx 4: calm bar (for i=3)
        {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2},
        # idx 5: calm bar, close=100.5 -> TIMEOUT (+0.5R) for i=3
        {"open": 100.0, "high": 100.9, "low": 99.5, "close": 100.5},
        # idx 6: calm bar, close=99.7 -> TIMEOUT (-0.3R) for i=4
        {"open": 100.0, "high": 100.4, "low": 99.4, "close": 99.7},
        # idx 7: SL hit -> LOSS for i=5
        {"open": 100.0, "high": 100.2, "low": 98.5, "close": 98.8},
    ]
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="15min")
    df_btc = pd.DataFrame(rows, index=idx)
    atr_pct = pd.Series(0.01, index=idx)

    outcomes = label_engine.compute_outcomes(df_btc=df_btc, atr_pct=atr_pct, direction="LONG")

    assert list(outcomes["outcome_state"]) == [
        "WIN",
        "LOSS",
        "AMBIGUOUS",
        "TIMEOUT",
        "TIMEOUT",
        "LOSS",
        "INCOMPLETE",
        "INCOMPLETE",
    ]

    te = estimate_timeout_economics(outcomes)

    # Not-WIN válidos (excluyendo AMBIGUOUS e INCOMPLETE): 2 LOSS + 2 TIMEOUT = 4
    # P(TIMEOUT | not WIN) = 2 / 4 = 0.5
    # E[realized_r_gross | TIMEOUT] = (0.5 + (-0.3)) / 2 = 0.1
    assert te.n_timeout_samples == 2
    assert te.timeout_share_given_not_tp == pytest.approx(0.5, rel=1e-12)
    assert te.mean_timeout_r == pytest.approx(0.1, rel=1e-12)


def test_rejection_reasons_exact():
    """Verifica todos los motivos de rechazo y aceptación exactos exigidos por la especificación."""
    cm = CostModel(
        taker_fee_rate=0.0005,        # roundtrip = 0.0010
        half_spread_bps=0.5,          # roundtrip = 0.0001
        slippage_bps=1.5,             # roundtrip = 0.0003
        expected_funding_rate_8h=0.0,
        horizon_hours=6.0,
    )
    engine = EVEngine(tp_mult=2.0, sl_mult=1.0, cost_model=cm, ev_min_threshold=0.12, direction="LONG")

    p_cal = np.array([
        0.55,    # 0: ACCEPTED (EV alto, ATR sano)
        0.35,    # 1: EV_BELOW_THRESHOLD (EV bruto = 0.05 < 0.12)
        np.nan,  # 2: INVALID_PROBA (NaN)
        1.20,    # 3: INVALID_PROBA (> 1)
        -0.05,   # 4: INVALID_PROBA (< 0)
        0.60,    # 5: INVALID_ATR (atr = 0)
        0.60,    # 6: INVALID_ATR (atr = NaN)
        0.95,    # 7: NON_POSITIVE_NET_REWARD (atr diminuto -> total_cost_in_r >= gross_win_r = 2.0)
    ])
    atr_pct = np.array([
        0.01,    # 0: total_cost_in_r = 0.14
        0.01,    # 1: total_cost_in_r = 0.14
        0.01,    # 2
        0.01,    # 3
        0.01,    # 4
        0.0,     # 5: inválido
        np.nan,  # 6: inválido
        0.0005,  # 7: total_cost_in_r = 0.0014 / 0.0005 = 2.8 >= 2.0
    ])

    res = engine.evaluate(p_calibrated=p_cal, atr_pct=atr_pct)

    expected_reasons = [
        "ACCEPTED",
        "EV_BELOW_THRESHOLD",
        "INVALID_PROBA",
        "INVALID_PROBA",
        "INVALID_PROBA",
        "INVALID_ATR",
        "INVALID_ATR",
        "NON_POSITIVE_NET_REWARD",
    ]
    assert list(res["rejection_reason"]) == expected_reasons
    assert list(res["accepted"]) == [True, False, False, False, False, False, False, False]

    # Probabilidades explícitas cuya suma != 1.0 deben marcarse como INVALID_PROBA
    res_bad_sum = engine.evaluate(p_calibrated=0.5, atr_pct=0.01, p_sl=0.4, p_timeout=0.4)
    assert res_bad_sum.loc[0, "rejection_reason"] == "INVALID_PROBA"
    assert bool(res_bad_sum.loc[0, "accepted"]) is False


def test_sensitivity_report_monotonic_with_increasing_costs():
    """
    Bajo escenarios de costes estrictamente crecientes, el EV neto medio debe ser
    estrictamente decreciente y la tasa de aceptación no creciente.
    """
    engine = EVEngine(tp_mult=2.0, sl_mult=1.0, ev_min_threshold=0.12, direction="LONG")
    p_cal = np.array([0.40, 0.45, 0.50, 0.55, 0.60])
    atr_pct = np.array([0.008, 0.010, 0.012, 0.010, 0.015])
    te = TimeoutEconomics(timeout_share_given_not_tp=0.25, mean_timeout_r=0.05, n_timeout_samples=40)

    scenarios = {
        "zero_cost": CostModel(
            taker_fee_rate=0.0,
            half_spread_bps=0.0,
            slippage_bps=0.0,
            expected_funding_rate_8h=0.0,
            horizon_hours=6.0,
        ),
        "low_cost": CostModel(
            taker_fee_rate=0.0002,
            half_spread_bps=0.3,
            slippage_bps=0.5,
            expected_funding_rate_8h=0.00005,
            horizon_hours=6.0,
        ),
        "base_cost": CostModel(
            taker_fee_rate=0.0005,
            half_spread_bps=0.5,
            slippage_bps=1.5,
            expected_funding_rate_8h=0.0001,
            horizon_hours=6.0,
        ),
        "stress_cost": CostModel(
            taker_fee_rate=0.0010,
            half_spread_bps=1.5,
            slippage_bps=4.0,
            expected_funding_rate_8h=0.0003,
            horizon_hours=6.0,
        ),
    }

    report = engine.sensitivity_report(
        p_calibrated=p_cal,
        atr_pct=atr_pct,
        cost_scenarios=scenarios,
        timeout_economics=te,
    )

    assert list(report["scenario"]) == ["zero_cost", "low_cost", "base_cost", "stress_cost"]
    assert (report["n_total"] == len(p_cal)).all()

    # Costes totales monótonamente crecientes
    costs = report["mean_total_cost_in_r"].to_numpy()
    assert np.all(np.diff(costs) > 0.0)

    # EV neto monótonamente decreciente
    evs = report["mean_ev_net_r"].to_numpy()
    assert np.all(np.diff(evs) < 0.0)

    # EV bruto invariante entre escenarios de coste
    gross_evs = report["mean_ev_gross_r"].to_numpy()
    np.testing.assert_allclose(gross_evs, gross_evs[0], rtol=1e-12, atol=1e-12)

    # Tasa de aceptación no creciente
    acc_rates = report["acceptance_rate"].to_numpy()
    assert np.all(np.diff(acc_rates) <= 0.0)


def test_mathematical_consistency_controlled_conditions():
    """Verifica consistencia matemática exacta de todas las fórmulas bajo condiciones controladas."""
    cm = CostModel(
        taker_fee_rate=0.0004,        # roundtrip = 0.0008
        half_spread_bps=1.0,          # roundtrip = 0.0002
        slippage_bps=2.0,             # roundtrip = 0.0004
        expected_funding_rate_8h=0.0002,
        horizon_hours=8.0,            # 8/8 * 0.0002 = 0.0002
    )
    tp_mult = 3.0
    sl_mult = 1.5
    engine = EVEngine(tp_mult=tp_mult, sl_mult=sl_mult, cost_model=cm, ev_min_threshold=0.10, direction="LONG")

    p_cal = 0.45
    atr_pct = 0.02  # r_dist_pct = 0.02 * 1.5 = 0.03
    te = TimeoutEconomics(timeout_share_given_not_tp=0.20, mean_timeout_r=-0.10, n_timeout_samples=25)

    res = engine.evaluate(p_calibrated=p_cal, atr_pct=atr_pct, timeout_economics=te)
    row = res.iloc[0]

    # Verificaciones exactas
    expected_gross_win_r = 3.0 / 1.5  # 2.0
    expected_gross_loss_r = -1.0
    expected_p_to = (1.0 - 0.45) * 0.20  # 0.11
    expected_p_sl = (1.0 - 0.45) * 0.80  # 0.44
    expected_ev_gross = 0.45 * 2.0 + 0.44 * (-1.0) + 0.11 * (-0.10)  # 0.90 - 0.44 - 0.011 = 0.449

    expected_total_cost = (0.0008 + 0.0002 + 0.0004 + 0.0002) / 0.03  # 0.0016 / 0.03
    expected_ev_net = expected_ev_gross - expected_total_cost
    expected_rrr_net = (expected_gross_win_r - expected_total_cost) / (1.0 + expected_total_cost)

    assert row["gross_win_r"] == pytest.approx(expected_gross_win_r, rel=1e-12)
    assert row["gross_loss_r"] == pytest.approx(expected_gross_loss_r, rel=1e-12)
    assert row["p_timeout"] == pytest.approx(expected_p_to, rel=1e-12)
    assert row["p_sl"] == pytest.approx(expected_p_sl, rel=1e-12)
    assert row["p_calibrated"] + row["p_sl"] + row["p_timeout"] == pytest.approx(1.0, rel=1e-12)
    assert row["ev_gross_r"] == pytest.approx(expected_ev_gross, rel=1e-12)
    assert row["total_cost_in_r"] == pytest.approx(expected_total_cost, rel=1e-12)
    assert row["ev_net_r"] == pytest.approx(expected_ev_net, rel=1e-12)
    assert row["reward_risk_ratio_net"] == pytest.approx(expected_rrr_net, rel=1e-12)
    assert bool(row["accepted"]) is True
    assert row["rejection_reason"] == "ACCEPTED"


def test_edge_cases_empty_and_validation():
    """Verifica casos límite: DataFrame vacío, floor 1e-6 de r_dist_pct y validaciones de entrada."""
    empty_outcomes = pd.DataFrame(columns=["outcome_state", "realized_r_gross"])
    te_empty = estimate_timeout_economics(empty_outcomes)
    assert te_empty == TimeoutEconomics(0.0, 0.0, 0)

    engine = EVEngine()
    empty_eval = engine.evaluate(p_calibrated=pd.Series([], dtype=float), atr_pct=pd.Series([], dtype=float))
    assert len(empty_eval) == 0
    assert list(empty_eval.columns) == EXPECTED_COLUMNS

    empty_sens = engine.sensitivity_report(p_calibrated=[0.5], atr_pct=[0.01], cost_scenarios={})
    assert len(empty_sens) == 0

    with pytest.raises(ValueError):
        EVEngine(direction="SIDEWAYS")
    with pytest.raises(ValueError):
        EVEngine(tp_mult=0.0)
    with pytest.raises(ValueError):
        TimeoutEconomics(timeout_share_given_not_tp=1.5)
