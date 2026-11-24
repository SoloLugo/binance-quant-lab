import math
import pytest

from contracts import RiskDecision
from risk_engine import RiskEngine, RiskLimits


def _base_order_kwargs():
    """Parámetros válidos con probabilidad muy alta (0.95) y EV positivo (1.0R)."""
    return {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "equity_usdt": 10000.0,
        "available_balance_usdt": 10000.0,
        "peak_equity_usdt": 10000.0,
        "daily_start_equity_usdt": 10000.0,
        "entry_price": 70000.0,
        "sl_distance_usdt": 500.0,
        "ev_net_r": 1.0,
        "p_calibrated": 0.95,
        "b_net_odds": 2.0,
        "l_net_odds": 1.0,
        "open_positions": [],
        "max_corr_with_open": 0.0,
    }


def test_kill_switch_rejects_even_with_p95_and_resets():
    engine = RiskEngine()
    assert engine.kill_switch_active is False
    assert engine.kill_switch_reason is None

    engine.activate_kill_switch("ANOMALOUS_SLIPPAGE_SPIKE")
    assert engine.kill_switch_active is True
    assert engine.kill_switch_reason == "ANOMALOUS_SLIPPAGE_SPIKE"

    decision = engine.evaluate_order(**_base_order_kwargs())
    assert isinstance(decision, RiskDecision)
    assert decision.approved is False
    assert decision.reason == "KILL_SWITCH_ACTIVE"
    assert decision.qty_btc == 0.0

    engine.reset_kill_switch()
    assert engine.kill_switch_active is False
    assert engine.kill_switch_reason is None

    approved = engine.evaluate_order(**_base_order_kwargs())
    assert approved.approved is True
    assert approved.reason == "APPROVED"


@pytest.mark.parametrize(
    "override",
    [
        {"equity_usdt": 0.0},
        {"available_balance_usdt": -10.0},
        {"entry_price": 0.0},
        {"sl_distance_usdt": 0.0},
        {"side": "INVALID"},
        {"equity_usdt": float("nan")},
    ],
)
def test_invalid_inputs_rejected(override):
    engine = RiskEngine()
    kwargs = _base_order_kwargs()
    kwargs.update(override)
    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "INVALID_INPUTS"


def test_max_drawdown_rejects_even_with_p95():
    engine = RiskEngine(RiskLimits(max_drawdown_pct=0.15))
    kwargs = _base_order_kwargs()
    # Peak = 10000, Equity = 8400 -> 16% drawdown >= 15%
    kwargs["peak_equity_usdt"] = 10000.0
    kwargs["daily_start_equity_usdt"] = 8500.0
    kwargs["equity_usdt"] = 8400.0
    kwargs["available_balance_usdt"] = 8400.0

    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "MAX_DRAWDOWN_BREACHED"


def test_daily_loss_limit_rejects_even_with_p95():
    engine = RiskEngine(RiskLimits(max_daily_loss_pct=0.06, max_drawdown_pct=0.15))
    kwargs = _base_order_kwargs()
    # Daily start = 10000, Equity = 9350 -> 6.5% daily loss >= 6%
    kwargs["peak_equity_usdt"] = 10000.0
    kwargs["daily_start_equity_usdt"] = 10000.0
    kwargs["equity_usdt"] = 9350.0
    kwargs["available_balance_usdt"] = 9350.0

    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "DAILY_LOSS_LIMIT_BREACHED"


def test_max_open_positions_rejects_even_with_p95():
    engine = RiskEngine(RiskLimits(max_open_positions=1))
    kwargs = _base_order_kwargs()
    kwargs["open_positions"] = [{"symbol": "ETHUSDT", "notional_usdt": 5000.0}]

    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "MAX_OPEN_POSITIONS_REACHED"


def test_correlation_limit_rejects_when_positions_open():
    engine = RiskEngine(RiskLimits(max_open_positions=3, max_correlation_threshold=0.70))
    kwargs = _base_order_kwargs()
    kwargs["open_positions"] = [{"symbol": "ETHUSDT", "notional_usdt": 5000.0}]
    kwargs["max_corr_with_open"] = 0.75

    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "CORRELATION_LIMIT_BREACHED"

    # Sin posiciones abiertas, la correlación no aplica
    kwargs["open_positions"] = []
    decision_no_open = engine.evaluate_order(**kwargs)
    assert decision_no_open.approved is True


def test_non_positive_ev_rejected():
    engine = RiskEngine()
    kwargs = _base_order_kwargs()
    kwargs["ev_net_r"] = 0.0
    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "NON_POSITIVE_EV"


def test_uncertainty_adjusted_kelly_zero_rejected():
    limits = RiskLimits(probability_uncertainty_margin=0.02, kelly_fraction=0.20)
    engine = RiskEngine(limits)
    # Con b=2.0, l=1.0, punto de equilibrio p=1/3=0.3333.
    # p_calibrated=0.35 tiene EV positivo antes del margen, pero p_cons=0.33 -> ev_cons = 0.66 - 0.67 = -0.01 <= 0
    assert engine.compute_conservative_kelly(p_calibrated=0.35, b_net_odds=2.0, l_net_odds=1.0) == 0.0

    kwargs = _base_order_kwargs()
    kwargs["p_calibrated"] = 0.35
    kwargs["b_net_odds"] = 2.0
    kwargs["ev_net_r"] = 0.05
    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "UNCERTAINTY_ADJUSTED_KELLY_ZERO"


def test_exposure_and_symbol_concentration_limits_enforce_caps_and_rejection():
    limits = RiskLimits(
        max_open_positions=3,
        max_leverage=5.0,
        max_total_exposure_notional_mult=2.0,  # 20,000 USDT total max para equity 10,000
        max_symbol_concentration_pct=0.50,     # 10,000 USDT max por símbolo
        min_notional_usdt=100.0,
        min_qty_btc=0.002,
    )
    engine = RiskEngine(limits)

    # Ya tenemos 9,950 USDT en BTCUSDT -> sólo quedan 50 USDT de concentración (< min_notional 100 USDT)
    kwargs = _base_order_kwargs()
    kwargs["open_positions"] = [{"symbol": "BTCUSDT", "notional_usdt": 9950.0}]
    decision_conc = engine.evaluate_order(**kwargs)
    assert decision_conc.approved is False
    assert decision_conc.reason == "SIZE_BELOW_SAFE_MINIMUM"

    # Ya tenemos 19,950 USDT en otro símbolo -> sólo quedan 50 USDT de exposición total
    kwargs["open_positions"] = [{"symbol": "ETHUSDT", "notional_usdt": 19950.0}]
    decision_total = engine.evaluate_order(**kwargs)
    assert decision_total.approved is False
    assert decision_total.reason == "SIZE_BELOW_SAFE_MINIMUM"

    # Con 15,000 USDT en ETHUSDT quedan 5,000 USDT totales -> debe limitar qty a floor(5000/70000, 3) = 0.071 BTC
    kwargs["open_positions"] = [{"symbol": "ETHUSDT", "notional_usdt": 15000.0}]
    decision_capped = engine.evaluate_order(**kwargs)
    assert decision_capped.approved is True
    assert decision_capped.qty_btc == pytest.approx(0.071)
    assert decision_capped.qty_btc * 70000.0 <= 5000.0


def test_unsafe_minimum_lot_rejected():
    engine = RiskEngine(RiskLimits(max_risk_per_trade=0.03, min_qty_btc=0.002, min_notional_usdt=100.0))
    kwargs = _base_order_kwargs()
    # Balance 20 USDT -> riesgo máximo 0.60 USDT. Lote mínimo 0.002 BTC con SL 500 USDT arriesga 1.0 USDT (> 0.60)
    kwargs["equity_usdt"] = 20.0
    kwargs["available_balance_usdt"] = 20.0
    kwargs["peak_equity_usdt"] = 20.0
    kwargs["daily_start_equity_usdt"] = 20.0

    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is False
    assert decision.reason == "SIZE_BELOW_SAFE_MINIMUM"


def test_approved_order_exact_sizing_and_leverage_cap():
    limits = RiskLimits(
        max_risk_per_trade=0.03,
        kelly_fraction=0.20,
        probability_uncertainty_margin=0.02,
        max_leverage=5.0,
    )
    engine = RiskEngine(limits)

    # p_calibrated=0.52 -> p_cons=0.50; b=2.0, l=1.0 -> ev_cons = 0.50*2 - 0.50*1 = 0.50
    # kelly_full = 0.50 / 2.0 = 0.25 -> scaled = 0.25 * 0.20 = 0.05 -> capped at max_risk_per_trade = 0.03
    k = engine.compute_conservative_kelly(p_calibrated=0.52, b_net_odds=2.0, l_net_odds=1.0)
    assert k == pytest.approx(0.03)

    # p_calibrated=0.42 -> p_cons=0.40; b=2.0, l=1.0 -> ev_cons = 0.80 - 0.60 = 0.20
    # kelly_full = 0.10 -> scaled = 0.02 (< 0.03)
    k_small = engine.compute_conservative_kelly(p_calibrated=0.42, b_net_odds=2.0, l_net_odds=1.0)
    assert k_small == pytest.approx(0.02)

    kwargs = _base_order_kwargs()
    kwargs["p_calibrated"] = 0.42
    kwargs["sl_distance_usdt"] = 500.0
    # risk_budget = 10000 * 0.02 = 200 USDT -> qty_by_risk = 200 / 500 = 0.400 BTC
    # notional = 0.400 * 70000 = 28,000 USDT -> leverage = 2.8x <= 5.0x
    decision = engine.evaluate_order(**kwargs)
    assert decision.approved is True
    assert decision.reason == "APPROVED"
    assert decision.qty_btc == pytest.approx(0.400)
    assert decision.risk_fraction == pytest.approx(0.02)
    assert decision.effective_leverage == pytest.approx(2.8)

    # Caso con SL muy estrecho donde aplica el tope de apalancamiento 5x (max notional = 50,000 -> 0.714 BTC)
    kwargs_tight_sl = _base_order_kwargs()
    kwargs_tight_sl["sl_distance_usdt"] = 100.0
    decision_lev = engine.evaluate_order(**kwargs_tight_sl)
    expected_qty = math.floor((50000.0 / 70000.0) * 1000.0) / 1000.0
    assert decision_lev.approved is True
    assert decision_lev.qty_btc == pytest.approx(expected_qty)
    assert decision_lev.effective_leverage <= 5.0
