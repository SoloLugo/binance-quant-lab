import math
import numpy as np
import pandas as pd
import pytest

from backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from contracts import CostModel


def _make_ohlcv(
    n: int,
    start: str = "2024-01-01 00:00:00",
    freq: str = "15min",
    base_price: float = 50000.0,
) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    prices = np.full(n, base_price, dtype=float)
    return pd.DataFrame(
        {
            "open": prices.copy(),
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices.copy(),
            "volume": np.full(n, 100.0, dtype=float),
        },
        index=idx,
    )


def _make_signals(
    index: pd.Index,
    atr_pct: float = 0.01,
    sides: dict = None,
    risk_fraction: float = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "side": [None] * len(index),
            "atr_pct": np.full(len(index), atr_pct, dtype=float),
        },
        index=index,
    )
    if risk_fraction is not None:
        df["risk_fraction"] = float(risk_fraction)
    if sides:
        for k, v in sides.items():
            if isinstance(k, int):
                df.iloc[k, df.columns.get_loc("side")] = v
            else:
                df.loc[k, "side"] = v
    return df


def test_exact_accounting_multi_trade_long_and_short():
    """
    Verifica la identidad contable exacta al centavo:
    final_equity == initial_balance + sum(gross_pnl_usdt) - sum(total_fees_usdt) - sum(funding_paid_usdt)
    combinando trades LONG y SHORT con TP, SL, TIMEOUT, fees, slippage y funding.
    """
    df = _make_ohlcv(80, start="2024-01-01 06:00:00", base_price=50000.0)

    # Trade 1 (LONG): señal en i=0 -> entrada en open[1]=50000, SL=49500, TP=51000
    # Vela 5 toca TP (high=51500)
    df.iloc[5, df.columns.get_loc("high")] = 51500.0
    df.iloc[5, df.columns.get_loc("close")] = 51200.0

    # Trade 2 (SHORT): señal en i=7 -> entrada en open[8]=50000, SL=50500, TP=49000
    # Vela 12 cruza 08:00 UTC (i=8 es 08:00, pongamos señal en i=6 -> entrada en 07:45, vela 8 es 08:00)
    # Mejor: Trade 2 señal en i=6 (07:30) -> entrada en i=7 (07:45), vela 8 es 08:00 UTC (paga/cobra funding),
    # vela 10 toca SL (high=50800).
    df.iloc[10, df.columns.get_loc("high")] = 50800.0

    # Trade 3 (LONG): señal en i=15 -> entrada en i=16, no toca ni SL ni TP en 10 velas -> TIMEOUT en i=25
    df.iloc[25, df.columns.get_loc("close")] = 50350.0

    signals = _make_signals(
        df.index,
        atr_pct=0.01,
        sides={0: "LONG", 6: "SHORT", 15: "LONG"},
    )
    signals["p_calibrated"] = 0.62
    signals["ev_net_r"] = 0.35

    cfg = BacktestConfig(
        initial_balance=10000.0,
        tp_mult=2.0,
        sl_mult=1.0,
        horizon_bars=10,
        risk_per_trade_pct=0.02,
        cost_model=CostModel(
            taker_fee_rate=0.0005,
            half_spread_bps=0.5,
            slippage_bps=1.5,
            expected_funding_rate_8h=0.0002,
        ),
    )
    res = BacktestEngine(cfg).run(df, signals)

    assert res.metrics["total_trades"] == 3
    gross_sum = float(res.trades["gross_pnl_usdt"].sum())
    fees_sum = float(res.trades["total_fees_usdt"].sum())
    funding_sum = float(res.trades["funding_paid_usdt"].sum())
    expected_final = cfg.initial_balance + gross_sum - fees_sum - funding_sum

    assert abs(res.metrics["final_equity_usdt"] - expected_final) < 1e-6
    assert abs(float(res.equity_curve.iloc[-1]) - expected_final) < 1e-6
    assert abs(float(res.trades["net_pnl_usdt"].sum()) - (expected_final - cfg.initial_balance)) < 1e-6

    # Verificar que todas las métricas requeridas del Plan Maestro (§10) existen
    expected_metric_keys = {
        "total_return_pct",
        "cagr_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "profit_factor",
        "expectancy_r",
        "expectancy_usdt",
        "win_rate_pct",
        "total_trades",
        "rejected_entries",
        "exposure_pct",
        "turnover",
        "r_distribution",
        "avg_duration_bars",
        "total_fees_usdt",
        "total_funding_usdt",
        "total_slippage_usdt",
        "exit_reason_counts",
        "ambiguous_trades_count",
        "final_equity_usdt",
    }
    assert expected_metric_keys.issubset(res.metrics.keys())
    assert set(res.metrics["r_distribution"].keys()) == {"mean", "std", "p10", "p50", "p90", "min", "max"}


def test_determinism_and_isolated_state_between_runs(tmp_path, monkeypatch):
    """
    Dos corridas independientes producen exactamente las mismas métricas, trades y equity_curve,
    y no mutan estado ni escriben archivos en disco.
    """
    monkeypatch.chdir(tmp_path)

    df = _make_ohlcv(50, start="2024-02-01 00:00:00", base_price=40000.0)
    df.iloc[4, df.columns.get_loc("high")] = 41200.0
    df.iloc[12, df.columns.get_loc("low")] = 39200.0
    df.iloc[25, df.columns.get_loc("close")] = 40200.0

    signals = _make_signals(df.index, atr_pct=0.01, sides={0: "LONG", 8: "LONG", 18: "SHORT"})

    cfg = BacktestConfig(initial_balance=15000.0, horizon_bars=8)
    engine = BacktestEngine(cfg)

    res1 = engine.run(df, signals)

    # Intercalamos una corrida con un dataset distinto sobre la MISMA instancia
    df_other = _make_ohlcv(30, start="2024-03-01 00:00:00", base_price=60000.0)
    df_other.iloc[3, df_other.columns.get_loc("low")] = 58000.0
    sig_other = _make_signals(df_other.index, atr_pct=0.01, sides={0: "LONG"})
    _ = engine.run(df_other, sig_other)

    res2 = engine.run(df, signals)
    res3 = BacktestEngine(cfg).run(df, signals)

    assert res1.metrics == res2.metrics == res3.metrics
    pd.testing.assert_frame_equal(res1.trades, res2.trades)
    pd.testing.assert_frame_equal(res1.trades, res3.trades)
    pd.testing.assert_series_equal(res1.equity_curve, res2.equity_curve)
    pd.testing.assert_series_equal(res1.equity_curve, res3.equity_curve)

    # Verifica que no se crearon archivos de estado de Paper Trading en el directorio de trabajo
    assert list(tmp_path.iterdir()) == []


def test_single_position_and_no_reentry_on_exit_bar():
    """
    Solo puede haber 1 posición activa a la vez y no se permite re-entrada en la
    misma barra de cierre (next_free_idx = exit_idx + 1).
    """
    df = _make_ohlcv(30, start="2024-01-01 00:00:00", base_price=50000.0)
    # Trade 1 entra en i=0 -> entry_idx=1, cierra por TP en exit_idx=3
    df.iloc[3, df.columns.get_loc("high")] = 51500.0
    # Trade 2 debe esperar a signal_idx >= 4 -> entry_idx=5, cierra por TP en exit_idx=7
    df.iloc[7, df.columns.get_loc("high")] = 51500.0

    # Señales en TODAS las barras
    signals = _make_signals(df.index, atr_pct=0.01, sides={i: "LONG" for i in range(len(df))})

    cfg = BacktestConfig(horizon_bars=10)
    res = BacktestEngine(cfg).run(df, signals)

    assert len(res.trades) >= 2
    t0 = res.trades.iloc[0]
    t1 = res.trades.iloc[1]

    assert t0["signal_idx"] == 0
    assert t0["entry_idx"] == 1
    assert t0["exit_idx"] == 3

    # La señal en exit_idx=3 DEBE ser ignorada; la siguiente válida es signal_idx=4 -> entry_idx=5
    assert t1["signal_idx"] == 4
    assert t1["entry_idx"] == 5

    # Ningún par consecutivo de trades se solapa
    for idx in range(1, len(res.trades)):
        prev_exit = int(res.trades.iloc[idx - 1]["exit_idx"])
        curr_signal = int(res.trades.iloc[idx]["signal_idx"])
        curr_entry = int(res.trades.iloc[idx]["entry_idx"])
        assert curr_signal >= prev_exit + 1
        assert curr_entry > prev_exit


def test_adverse_price_gap_worse_than_stop_loss():
    """
    Si open[k] abre ya más allá del sl_price, la salida se ejecuta a open[k] + slippage adverso
    (peor que sl_price), resultando en pérdida > 1R bruto.
    """
    # Caso LONG
    df_long = _make_ohlcv(15, start="2024-01-01 00:00:00", base_price=50000.0)
    # Entrada en open[1]=50000, atr_pct=0.01 -> sl_price ~ 49500
    # En k=3 abre con gap bajista fuerte a 48000 (mucho peor que 49500)
    df_long.iloc[3, df_long.columns.get_loc("open")] = 48000.0
    df_long.iloc[3, df_long.columns.get_loc("high")] = 48200.0
    df_long.iloc[3, df_long.columns.get_loc("low")] = 47900.0
    df_long.iloc[3, df_long.columns.get_loc("close")] = 48100.0

    sig_long = _make_signals(df_long.index, atr_pct=0.01, sides={0: "LONG"})
    cost = CostModel(taker_fee_rate=0.0, half_spread_bps=0.0, slippage_bps=10.0, expected_funding_rate_8h=0.0)
    res_long = BacktestEngine(BacktestConfig(cost_model=cost)).run(df_long, sig_long)

    assert len(res_long.trades) == 1
    tr_l = res_long.trades.iloc[0]
    assert tr_l["exit_reason"] == "STOP_LOSS"
    assert tr_l["exit_price"] == pytest.approx(48000.0 * (1.0 - 0.001), rel=1e-9)
    assert tr_l["exit_price"] < tr_l["sl_price"]
    assert tr_l["realized_r_gross"] < -3.0

    # Caso SHORT
    df_short = _make_ohlcv(15, start="2024-01-01 00:00:00", base_price=50000.0)
    # Entrada en open[1]=50000, sl_price ~ 50500; gap alcista en k=3 a 52000
    df_short.iloc[3, df_short.columns.get_loc("open")] = 52000.0
    df_short.iloc[3, df_short.columns.get_loc("high")] = 52200.0
    df_short.iloc[3, df_short.columns.get_loc("low")] = 51900.0
    df_short.iloc[3, df_short.columns.get_loc("close")] = 52100.0

    sig_short = _make_signals(df_short.index, atr_pct=0.01, sides={0: "SHORT"})
    res_short = BacktestEngine(BacktestConfig(cost_model=cost)).run(df_short, sig_short)

    assert len(res_short.trades) == 1
    tr_s = res_short.trades.iloc[0]
    assert tr_s["exit_reason"] == "STOP_LOSS"
    assert tr_s["exit_price"] == pytest.approx(52000.0 * (1.0 + 0.001), rel=1e-9)
    assert tr_s["exit_price"] > tr_s["sl_price"]
    assert tr_s["realized_r_gross"] < -3.0


def test_funding_8h_sign_convention_long_and_short():
    """
    Verifica la liquidación de Funding Rate cada 8h (00:00, 08:00, 16:00 UTC)
    con signo correcto en LONG y SHORT, tanto con funding positivo como negativo.
    """
    # Barras de 07:30 a 09:30 UTC:
    # i=0 (07:30) señal -> entry_idx=1 (07:45) -> k=2 es 08:00 UTC (liquidación)
    df = _make_ohlcv(10, start="2024-01-01 07:30:00", base_price=50000.0)
    zero_cost = CostModel(
        taker_fee_rate=0.0,
        half_spread_bps=0.0,
        slippage_bps=0.0,
        expected_funding_rate_8h=0.0,
    )
    cfg = BacktestConfig(initial_balance=10000.0, horizon_bars=5, cost_model=zero_cost)

    # Funding positivo (+0.001) a las 08:00 UTC
    df_fund_pos = pd.DataFrame(
        {"funding_rate": [0.001]},
        index=pd.DatetimeIndex(["2024-01-01 08:00:00"], tz="UTC"),
    )
    res_long_pos = BacktestEngine(cfg).run(
        df, _make_signals(df.index, atr_pct=0.01, sides={0: "LONG"}), df_funding=df_fund_pos
    )
    res_short_pos = BacktestEngine(cfg).run(
        df, _make_signals(df.index, atr_pct=0.01, sides={0: "SHORT"}), df_funding=df_fund_pos
    )

    # Con funding > 0: LONG paga (+), SHORT cobra (-)
    long_f = res_long_pos.metrics["total_funding_usdt"]
    short_f = res_short_pos.metrics["total_funding_usdt"]
    assert long_f > 0.0
    assert short_f < 0.0
    assert long_f == pytest.approx(-short_f, rel=1e-9)
    assert res_long_pos.metrics["final_equity_usdt"] == pytest.approx(10000.0 - long_f, rel=1e-9)
    assert res_short_pos.metrics["final_equity_usdt"] == pytest.approx(10000.0 - short_f, rel=1e-9)

    # Funding negativo (-0.001) a las 08:00 UTC: LONG cobra (-), SHORT paga (+)
    df_fund_neg = pd.DataFrame(
        {"funding_rate": [-0.001]},
        index=pd.DatetimeIndex(["2024-01-01 08:00:00"], tz="UTC"),
    )
    res_long_neg = BacktestEngine(cfg).run(
        df, _make_signals(df.index, atr_pct=0.01, sides={0: "LONG"}), df_funding=df_fund_neg
    )
    res_short_neg = BacktestEngine(cfg).run(
        df, _make_signals(df.index, atr_pct=0.01, sides={0: "SHORT"}), df_funding=df_fund_neg
    )

    assert res_long_neg.metrics["total_funding_usdt"] < 0.0
    assert res_short_neg.metrics["total_funding_usdt"] > 0.0
    assert res_long_neg.metrics["final_equity_usdt"] > 10000.0
    assert res_short_neg.metrics["final_equity_usdt"] < 10000.0


def test_adverse_scenarios_partial_fill_failures_entry_gap_and_high_costs():
    """
    Escenarios adversos:
    - partial_fill_ratio = 0.5 reduce tamaño y retorno neto proporcionalmente.
    - execution_failures rechaza la entrada e incrementa rejected_entries.
    - max_entry_gap_pct rechaza entradas con gap excesivo.
    - CostModel elevado reduce el retorno neto frente a costes bajos.
    """
    df = _make_ohlcv(20, start="2024-01-01 00:00:00", base_price=50000.0)
    # Trade ganador claro en k=3
    df.iloc[3, df.columns.get_loc("high")] = 51500.0
    signals = _make_signals(df.index, atr_pct=0.01, sides={0: "LONG"})

    # 1. partial_fill_ratio = 0.5 vs 1.0
    res_full = BacktestEngine(BacktestConfig(partial_fill_ratio=1.0)).run(df, signals)
    res_half = BacktestEngine(BacktestConfig(partial_fill_ratio=0.5)).run(df, signals)

    assert res_half.trades.iloc[0]["qty"] == pytest.approx(res_full.trades.iloc[0]["qty"] * 0.5, rel=1e-9)
    assert res_half.trades.iloc[0]["net_pnl_usdt"] == pytest.approx(
        res_full.trades.iloc[0]["net_pnl_usdt"] * 0.5, rel=1e-9
    )

    # 2. execution_failures (por índice entero y por timestamp)
    res_fail_idx = BacktestEngine(BacktestConfig()).run(df, signals, execution_failures={0})
    assert res_fail_idx.metrics["total_trades"] == 0
    assert res_fail_idx.metrics["rejected_entries"] == 1

    res_fail_ts = BacktestEngine(BacktestConfig()).run(df, signals, execution_failures={df.index[0]})
    assert res_fail_ts.metrics["total_trades"] == 0
    assert res_fail_ts.metrics["rejected_entries"] == 1

    # 3. max_entry_gap_pct
    df_gap = df.copy()
    df_gap.iloc[1, df_gap.columns.get_loc("open")] = 51000.0  # 2% gap respecto a close[0]=50000
    res_gap_reject = BacktestEngine(BacktestConfig(max_entry_gap_pct=0.005)).run(df_gap, signals)
    assert res_gap_reject.metrics["total_trades"] == 0
    assert res_gap_reject.metrics["rejected_entries"] == 1

    # 4. Costes elevados reducen el retorno neto
    low_cost = CostModel(taker_fee_rate=0.0001, half_spread_bps=0.1, slippage_bps=0.2, expected_funding_rate_8h=0.0)
    high_cost = CostModel(taker_fee_rate=0.0020, half_spread_bps=5.0, slippage_bps=10.0, expected_funding_rate_8h=0.0)
    res_low = BacktestEngine(BacktestConfig(cost_model=low_cost)).run(df, signals)
    res_high = BacktestEngine(BacktestConfig(cost_model=high_cost)).run(df, signals)

    assert res_high.metrics["final_equity_usdt"] < res_low.metrics["final_equity_usdt"]
    assert res_high.metrics["total_fees_usdt"] > res_low.metrics["total_fees_usdt"]
    assert res_high.metrics["total_slippage_usdt"] > res_low.metrics["total_slippage_usdt"]


def test_exit_reasons_sl_tp_timeout_and_intrabar_ambiguous():
    """
    Verifica salidas STOP_LOSS, TAKE_PROFIT, TIMEOUT (con retorno real distinto de -1R)
    y detección de intrabar_ambiguous bajo políticas 'conservative_sl' y 'skip'.
    """
    zero_cost = CostModel(taker_fee_rate=0.0, half_spread_bps=0.0, slippage_bps=0.0, expected_funding_rate_8h=0.0)

    # 1. TIMEOUT con retorno real positivo distinto de -1R (+0.5R)
    df_to = _make_ohlcv(12, start="2024-01-01 00:00:00", base_price=50000.0)
    df_to.iloc[5, df_to.columns.get_loc("close")] = 50250.0  # +250 sobre 500 de sl_dist => +0.5R
    sig_to = _make_signals(df_to.index, atr_pct=0.01, sides={0: "LONG"})
    res_to = BacktestEngine(BacktestConfig(horizon_bars=5, cost_model=zero_cost)).run(df_to, sig_to)

    assert res_to.metrics["exit_reason_counts"] == {"TIMEOUT": 1}
    assert res_to.trades.iloc[0]["realized_r_gross"] == pytest.approx(0.5, rel=1e-9)
    assert res_to.trades.iloc[0]["realized_r_gross"] != -1.0

    # 2. Intrabar ambiguous: misma vela toca tanto SL (49500) como TP (51000)
    df_amb = _make_ohlcv(10, start="2024-01-01 00:00:00", base_price=50000.0)
    df_amb.iloc[2, df_amb.columns.get_loc("high")] = 51500.0
    df_amb.iloc[2, df_amb.columns.get_loc("low")] = 49000.0
    sig_amb = _make_signals(df_amb.index, atr_pct=0.01, sides={0: "LONG"})

    # Política conservative_sl
    res_cons = BacktestEngine(
        BacktestConfig(ambiguous_policy="conservative_sl", cost_model=zero_cost)
    ).run(df_amb, sig_amb)
    assert res_cons.metrics["ambiguous_trades_count"] == 1
    assert res_cons.metrics["total_trades"] == 1
    assert bool(res_cons.trades.iloc[0]["intrabar_ambiguous"]) is True
    assert res_cons.trades.iloc[0]["exit_reason"] == "STOP_LOSS"
    assert res_cons.trades.iloc[0]["realized_r_gross"] == pytest.approx(-1.0, rel=1e-9)

    # Política skip
    res_skip = BacktestEngine(
        BacktestConfig(ambiguous_policy="skip", cost_model=zero_cost)
    ).run(df_amb, sig_amb)
    assert res_skip.metrics["ambiguous_trades_count"] == 1
    assert res_skip.metrics["total_trades"] == 0
    assert res_skip.metrics["final_equity_usdt"] == pytest.approx(10000.0, rel=1e-9)


def test_empty_dataframe_and_invalid_config_validation():
    """Verifica manejo limpio de DataFrames vacíos y validación de BacktestConfig."""
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close"])
    empty_sig = pd.DataFrame(columns=["side", "atr_pct"])

    res = BacktestEngine(BacktestConfig(initial_balance=5000.0)).run(empty_df, empty_sig)
    assert res.metrics["total_trades"] == 0
    assert res.metrics["final_equity_usdt"] == 5000.0
    assert res.trades.empty
    assert res.equity_curve.empty

    with pytest.raises(ValueError):
        BacktestConfig(ambiguous_policy="optimistic_tp")
    with pytest.raises(ValueError):
        BacktestConfig(partial_fill_ratio=0.0)
    with pytest.raises(ValueError):
        BacktestConfig(initial_balance=-100.0)
