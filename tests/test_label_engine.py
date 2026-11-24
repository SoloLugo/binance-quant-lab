"""
Tests exhaustivos del Label Engine (Fase 2 — Plan Maestro §4.5).
Verifica:
- Entrada ejecutable en `next_open` (y trazabilidad en `close`).
- TP primero, SL primero, TIMEOUT con retorno económico real distinto de -1R.
- AMBIGUOUS en misma vela excluido del entrenamiento principal y reportado en sensibilidad.
- Horizonte incompleto (`INCOMPLETE` / NaN en últimas `horizon_bars` velas y ATR inválido).
- Límites temporales exactos (`j=1`, `j=horizon_bars`, y no lectura de `j=horizon_bars+1`).
- Gap entre `close[i]` y `open[i+1]` en LONG y SHORT.
"""
import numpy as np
import pandas as pd
import pytest

from contracts import LabelSpec
from label_engine import LabelEngine


def _flat_ohlc(n: int = 30, price: float = 10000.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(n, price, dtype=float),
            "high": np.full(n, price + 10.0, dtype=float),
            "low": np.full(n, price - 10.0, dtype=float),
            "close": np.full(n, price, dtype=float),
        },
        index=idx,
    )


def test_long_tp_first_sl_first_and_timeout_real_return():
    """Verifica WIN, LOSS y TIMEOUT con retorno real distinto de -1R en LONG."""
    df = _flat_ohlc(n=35, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)  # ATR = 100 -> TP=+200 (10200), SL=-100 (9900)

    # Caso 1 (i=0): vela i+2 toca TP (10250) antes que SL
    df.iloc[2, df.columns.get_loc("high")] = 10250.0

    # Caso 2 (i=5): vela i+3 (idx=8) toca SL (9850) antes que TP
    df.iloc[8, df.columns.get_loc("low")] = 9850.0

    # Caso 3 (i=12): TIMEOUT tras horizon=5 (vela 17 cierra en 10060 -> +0.6R, no -1.0R)
    df.iloc[17, df.columns.get_loc("close")] = 10060.0
    df.iloc[17, df.columns.get_loc("high")] = 10070.0

    # Caso 4 (i=20): TIMEOUT negativo parcial (vela 25 cierra en 9960 -> -0.4R, no -1.0R)
    df.iloc[25, df.columns.get_loc("close")] = 9960.0
    df.iloc[25, df.columns.get_loc("low")] = 9955.0

    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=5, entry_mode="next_open")
    outcomes = LabelEngine(spec).compute_outcomes(df, atr_pct, direction="LONG")

    # WIN en i=0
    assert outcomes["outcome_state"].iloc[0] == "WIN"
    assert outcomes["exit_type"].iloc[0] == "TP"
    assert outcomes["exit_bars"].iloc[0] == 2
    assert outcomes["target_tp_before_sl"].iloc[0] == 1.0
    assert outcomes["target_tp_before_horizon"].iloc[0] == 1.0
    assert outcomes["target_sl_before_tp"].iloc[0] == 0.0
    assert outcomes["realized_r_gross"].iloc[0] == pytest.approx(2.0)
    assert outcomes["entry_price"].iloc[0] == pytest.approx(10000.0)
    assert outcomes["tp_price"].iloc[0] == pytest.approx(10200.0)
    assert outcomes["sl_price"].iloc[0] == pytest.approx(9900.0)

    # LOSS en i=5
    assert outcomes["outcome_state"].iloc[5] == "LOSS"
    assert outcomes["exit_type"].iloc[5] == "SL"
    assert outcomes["exit_bars"].iloc[5] == 3
    assert outcomes["target_tp_before_sl"].iloc[5] == 0.0
    assert outcomes["target_tp_before_horizon"].iloc[5] == 0.0
    assert outcomes["target_sl_before_tp"].iloc[5] == 1.0
    assert outcomes["realized_r_gross"].iloc[5] == pytest.approx(-1.0)

    # TIMEOUT positivo parcial en i=12 (+0.6R)
    assert outcomes["outcome_state"].iloc[12] == "TIMEOUT"
    assert outcomes["exit_type"].iloc[12] == "TIME"
    assert outcomes["exit_bars"].iloc[12] == 5
    assert outcomes["target_tp_before_sl"].iloc[12] == 0.0
    assert outcomes["target_tp_before_horizon"].iloc[12] == 0.0
    assert outcomes["target_sl_before_tp"].iloc[12] == 0.0
    assert outcomes["realized_r_gross"].iloc[12] == pytest.approx(0.6)

    # TIMEOUT negativo parcial en i=20 (-0.4R, no -1.0R)
    assert outcomes["outcome_state"].iloc[20] == "TIMEOUT"
    assert outcomes["realized_r_gross"].iloc[20] == pytest.approx(-0.4)


def test_short_tp_sl_timeout_and_ambiguous():
    """Verifica WIN, LOSS, TIMEOUT con retorno real y AMBIGUOUS en dirección SHORT."""
    df = _flat_ohlc(n=35, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)  # ATR = 100 -> SHORT TP=9800, SL=10100

    # i=0: vela i+1 toca TP SHORT (low <= 9800) sin tocar SL
    df.iloc[1, df.columns.get_loc("low")] = 9790.0

    # i=5: vela i+2 (idx=7) toca SL SHORT (high >= 10100) sin tocar TP
    df.iloc[7, df.columns.get_loc("high")] = 10120.0

    # i=12: vela i+1 (idx=13) toca AMBOS TP (<=9800) y SL (>=10100) -> AMBIGUOUS
    df.iloc[13, df.columns.get_loc("low")] = 9750.0
    df.iloc[13, df.columns.get_loc("high")] = 10150.0

    # i=20: TIMEOUT SHORT con close[25] = 9930 -> ganancia de +70 / 100 = +0.7R
    df.iloc[25, df.columns.get_loc("close")] = 9930.0
    df.iloc[25, df.columns.get_loc("low")] = 9920.0

    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=5, entry_mode="next_open")
    outcomes = LabelEngine(spec).compute_outcomes(df, atr_pct, direction="SHORT")

    assert outcomes["outcome_state"].iloc[0] == "WIN"
    assert outcomes["exit_type"].iloc[0] == "TP"
    assert outcomes["exit_bars"].iloc[0] == 1
    assert outcomes["realized_r_gross"].iloc[0] == pytest.approx(2.0)

    assert outcomes["outcome_state"].iloc[5] == "LOSS"
    assert outcomes["exit_type"].iloc[5] == "SL"
    assert outcomes["exit_bars"].iloc[5] == 2
    assert outcomes["target_sl_before_tp"].iloc[5] == 1.0
    assert outcomes["realized_r_gross"].iloc[5] == pytest.approx(-1.0)

    assert outcomes["outcome_state"].iloc[12] == "AMBIGUOUS"
    assert outcomes["exit_type"].iloc[12] == "AMBIGUOUS"
    assert outcomes["exit_bars"].iloc[12] == 1
    assert np.isnan(outcomes["target_tp_before_sl"].iloc[12])
    assert np.isnan(outcomes["target_tp_before_horizon"].iloc[12])
    assert np.isnan(outcomes["target_sl_before_tp"].iloc[12])
    assert np.isnan(outcomes["realized_r_gross"].iloc[12])

    assert outcomes["outcome_state"].iloc[20] == "TIMEOUT"
    assert outcomes["realized_r_gross"].iloc[20] == pytest.approx(0.7)


def test_ambiguous_excluded_and_sensitivity_summary():
    """Verifica exclusión de AMBIGUOUS del target y cálculo exacto de sensibilidad."""
    df = _flat_ohlc(n=15, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)  # TP=10200, SL=9900

    # horizon=2 -> filas evaluables: i=0..12 (13 filas), i=13..14 son INCOMPLETE
    # i=0: WIN en i+1 (idx=1)
    df.iloc[1, df.columns.get_loc("high")] = 10250.0
    # i=3: AMBIGUOUS en i+1 (idx=4)
    df.iloc[4, df.columns.get_loc("high")] = 10250.0
    df.iloc[4, df.columns.get_loc("low")] = 9850.0
    # i=6: LOSS en i+1 (idx=7)
    df.iloc[7, df.columns.get_loc("low")] = 9850.0

    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=2, exclude_ambiguous=True)
    engine = LabelEngine(spec)
    outcomes = engine.compute_outcomes(df, atr_pct, direction="LONG")

    assert outcomes["outcome_state"].iloc[3] == "AMBIGUOUS"
    assert np.isnan(outcomes["target_tp_before_sl"].iloc[3])
    assert np.isnan(outcomes["target_tp_before_horizon"].iloc[3])
    assert np.isnan(outcomes["target_sl_before_tp"].iloc[3])

    summary = engine.summarize_outcomes(outcomes)
    evaluated = summary["evaluated_rows"]
    assert evaluated == 13
    assert summary["counts"]["INCOMPLETE"] == 2
    assert summary["ambiguous_count"] >= 1
    assert summary["ambiguous_rate"] == pytest.approx(summary["ambiguous_count"] / evaluated)

    expected_win_if_amb_win = (summary["counts"]["WIN"] + summary["ambiguous_count"]) / evaluated
    expected_win_if_amb_loss = summary["counts"]["WIN"] / evaluated
    assert summary["sensitivity"]["win_rate_if_ambiguous_win"] == pytest.approx(expected_win_if_amb_win)
    assert summary["sensitivity"]["win_rate_if_ambiguous_loss"] == pytest.approx(expected_win_if_amb_loss)
    assert summary["sensitivity"]["win_rate_spread"] == pytest.approx(
        expected_win_if_amb_win - expected_win_if_amb_loss
    )


def test_incomplete_horizon_and_invalid_atr():
    """Las últimas `horizon_bars` velas y las velas con ATR NaN/<=0 quedan INCOMPLETE y target NaN."""
    df = _flat_ohlc(n=12, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)
    atr_pct.iloc[0] = np.nan
    atr_pct.iloc[1] = 0.0

    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=4)
    outcomes = LabelEngine(spec).compute_outcomes(df, atr_pct, direction="LONG")

    # ATR inválido en 0 y 1
    assert outcomes["outcome_state"].iloc[0] == "INCOMPLETE"
    assert outcomes["outcome_state"].iloc[1] == "INCOMPLETE"
    assert outcomes["target_tp_before_sl"].iloc[:2].isna().all()

    # Últimas 4 filas (indices 8, 9, 10, 11) deben ser INCOMPLETE con targets NaN
    assert (outcomes["outcome_state"].iloc[-4:] == "INCOMPLETE").all()
    assert (outcomes["exit_type"].iloc[-4:] == "NONE").all()
    assert outcomes["target_tp_before_sl"].iloc[-4:].isna().all()
    assert outcomes["target_tp_before_horizon"].iloc[-4:].isna().all()
    assert outcomes["target_sl_before_tp"].iloc[-4:].isna().all()
    assert outcomes["realized_r_gross"].iloc[-4:].isna().all()


def test_temporal_horizon_boundaries():
    """
    Verifica límites temporales exactos:
    - Toque en j=horizon_bars se detecta como WIN.
    - Toque en j=horizon_bars+1 NO se detecta (permanece TIMEOUT).
    """
    df = _flat_ohlc(n=20, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)  # TP=10200, SL=9900

    # Para horizon=4:
    # i=0 evalúa velas 1, 2, 3, 4 -> ponemos TP en vela 4 (j=4) -> debe ser WIN
    df.iloc[4, df.columns.get_loc("high")] = 10210.0

    # i=8 evalúa velas 9, 10, 11, 12 -> ponemos TP en vela 13 (j=5 = horizon+1) -> debe ser TIMEOUT
    df.iloc[13, df.columns.get_loc("high")] = 10250.0

    spec = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=4, entry_mode="next_open")
    outcomes = LabelEngine(spec).compute_outcomes(df, atr_pct, direction="LONG")

    assert outcomes["outcome_state"].iloc[0] == "WIN"
    assert outcomes["exit_bars"].iloc[0] == 4

    assert outcomes["outcome_state"].iloc[8] == "TIMEOUT"
    assert outcomes["exit_bars"].iloc[8] == 4


def test_price_gap_between_close_and_next_open_long_and_short():
    """
    Verifica que en `entry_mode='next_open'` un gap entre `close[i]` y `open[i+1]`
    ancla entry, TP y SL a `open[i+1]` usando `atr_pct[i] * open[i+1]`,
    a diferencia de `entry_mode='close'`.
    """
    df = _flat_ohlc(n=15, price=10000.0)
    atr_pct = pd.Series(0.01, index=df.index)

    # Gap alcista en vela 1: close[0]=10000, pero open[1]=10500
    # Con next_open: entry=10500, atr_val=105 -> LONG TP=10710, SL=10395
    # Con close:     entry=10000, atr_val=100 -> LONG TP=10200, SL=9900
    df.iloc[1, df.columns.get_loc("open")] = 10500.0
    df.iloc[1, df.columns.get_loc("high")] = 10550.0
    df.iloc[1, df.columns.get_loc("low")] = 10450.0
    df.iloc[1, df.columns.get_loc("close")] = 10520.0
    # Velas 2..4 se mantienen alrededor de 10500 sin tocar 10710 ni 10395
    for k in (2, 3, 4):
        df.iloc[k, df.columns.get_loc("open")] = 10500.0
        df.iloc[k, df.columns.get_loc("high")] = 10550.0
        df.iloc[k, df.columns.get_loc("low")] = 10450.0
        df.iloc[k, df.columns.get_loc("close")] = 10500.0

    spec_next = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=4, entry_mode="next_open")
    spec_close = LabelSpec(tp_mult=2.0, sl_mult=1.0, horizon_bars=4, entry_mode="close")

    out_next_long = LabelEngine(spec_next).compute_outcomes(df, atr_pct, direction="LONG")
    out_close_long = LabelEngine(spec_close).compute_outcomes(df, atr_pct, direction="LONG")

    # En next_open la entrada es 10500 -> ni TP(10710) ni SL(10395) se tocan -> TIMEOUT (0.0R)
    assert out_next_long["entry_price"].iloc[0] == pytest.approx(10500.0)
    assert out_next_long["tp_price"].iloc[0] == pytest.approx(10710.0)
    assert out_next_long["sl_price"].iloc[0] == pytest.approx(10395.0)
    assert out_next_long["outcome_state"].iloc[0] == "TIMEOUT"
    assert out_next_long["realized_r_gross"].iloc[0] == pytest.approx(0.0)

    # En close (legacy/trazabilidad) la entrada sería 10000 y el gap daría WIN artificial en j=1
    assert out_close_long["entry_price"].iloc[0] == pytest.approx(10000.0)
    assert out_close_long["outcome_state"].iloc[0] == "WIN"
    assert out_close_long["exit_bars"].iloc[0] == 1

    # En SHORT con next_open: entry=10500, TP=10290, SL=10605 -> TIMEOUT también
    out_next_short = LabelEngine(spec_next).compute_outcomes(df, atr_pct, direction="SHORT")
    assert out_next_short["entry_price"].iloc[0] == pytest.approx(10500.0)
    assert out_next_short["tp_price"].iloc[0] == pytest.approx(10290.0)
    assert out_next_short["sl_price"].iloc[0] == pytest.approx(10605.0)
    assert out_next_short["outcome_state"].iloc[0] == "TIMEOUT"
