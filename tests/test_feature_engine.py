"""
Tests de FeatureEngine (Fase 3 completa) y ContextEngine (Fase 4 completa).
Verifica:
- Activación y desactivación independiente de familias (`ta`, `microstructure`, `derivatives`, `market_structure`).
- Definiciones 100% algorítmicas y estrictamente causales de Market Structure (`ms_*`):
  Swing High/Low con lag `swing_right`, BOS, CHoCH, Equal Highs/Lows, Liquidity Sweeps,
  FVG alcista/bajista, Order Blocks (`dist` y `age`) y Premium/Discount.
- Derivados ampliados opcionales (`df_derivatives`) con alineación estricta `ffill` (sin `.bfill()`).
- Los 6 regímenes causales de `ContextEngine` y el Control Temporal Cuádruple (`available_at <= bar_timestamp`).
- Invarianza causal (modificar el futuro no altera ninguna feature pasada).
"""
import numpy as np
import pandas as pd
import pytest

from context_engine import ContextEngine, ExternalEvent
from feature_engine import ALL_FEATURE_FAMILIES, FeatureEngine
from pipeline import QuantFeaturePipeline


EXPECTED_TA_COLS = {
    "atr_pct",
    "ta_ret_1",
    "ta_ret_4",
    "ta_ret_16",
    "ta_dist_ema20",
    "ta_dist_ema50",
    "ta_rsi_14",
    "ta_bb_width",
    "ta_momentum_16",
    "ta_volatility_24",
    "ta_rel_volume_20",
    "ta_vwap_dist_24",
    "ta_dist_high_24",
    "ta_dist_low_24",
    "ta_body_ratio",
}

EXPECTED_SM_COLS = {
    "sm_ofi_1",
    "sm_ofi_cum_4",
    "sm_ofi_cum_16",
    "sm_avg_trade_size_zscore",
    "sm_whale_trade_zscore",
    "sm_lower_wick_ratio",
    "sm_upper_wick_ratio",
}

EXPECTED_DERIV_COLS = {
    "deriv_funding_rate",
    "deriv_funding_zscore",
    "deriv_funding_change_24h",
}

EXPECTED_MS_COLS = {
    "ms_dist_last_swing_high",
    "ms_dist_last_swing_low",
    "ms_bullish_bos",
    "ms_bearish_bos",
    "ms_bullish_choch",
    "ms_bearish_choch",
    "ms_equal_highs",
    "ms_equal_lows",
    "ms_bullish_sweep_strength",
    "ms_bearish_sweep_strength",
    "ms_bullish_fvg",
    "ms_bearish_fvg",
    "ms_fvg_size_atr",
    "ms_dist_bull_ob",
    "ms_bull_ob_age",
    "ms_dist_bear_ob",
    "ms_bear_ob_age",
    "ms_premium_discount_pos",
}


def _make_ohlcv_from_arrays(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="15min", tz="UTC")
    vol = np.full(n, 100.0, dtype=float) if volumes is None else volumes.astype(float)
    return pd.DataFrame(
        {
            "open": opens.astype(float),
            "high": highs.astype(float),
            "low": lows.astype(float),
            "close": closes.astype(float),
            "volume": vol,
            "quote_volume": vol * closes.astype(float),
            "count": np.full(n, 500.0, dtype=float),
            "taker_buy_volume": vol * 0.5,
        },
        index=idx,
    )


def test_independent_family_toggling(sample_ohlcv_and_funding):
    """Cada familia (`ta`, `microstructure`, `derivatives`, `market_structure`) puede activarse o desactivarse por separado."""
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding

    assert set(ALL_FEATURE_FAMILIES) == {"ta", "microstructure", "derivatives", "market_structure"}

    # Todas activadas por defecto
    full = FeatureEngine().build_features(df_btc, df_funding)
    assert set(full.columns) == (
        EXPECTED_TA_COLS | EXPECTED_SM_COLS | EXPECTED_DERIV_COLS | EXPECTED_MS_COLS
    )
    assert (
        full["sm_avg_trade_size_zscore"].dropna()
        .equals(full["sm_whale_trade_zscore"].dropna())
    )

    # Solo TA
    only_ta = FeatureEngine(enabled_families=("ta",)).build_features(df_btc, df_funding)
    assert set(only_ta.columns) == EXPECTED_TA_COLS

    # Solo Microstructure
    only_sm = FeatureEngine(enabled_families=("microstructure",)).build_features(df_btc, df_funding)
    assert set(only_sm.columns) == EXPECTED_SM_COLS

    # Solo Derivatives
    only_deriv = FeatureEngine(enabled_families=("derivatives",)).build_features(df_btc, df_funding)
    assert set(only_deriv.columns) == EXPECTED_DERIV_COLS

    # Solo Market Structure
    only_ms = FeatureEngine(enabled_families=("market_structure",)).build_features(df_btc, df_funding)
    assert set(only_ms.columns) == EXPECTED_MS_COLS

    # Combinación parcial (ta + derivatives sin microstructure ni market_structure)
    ta_deriv = FeatureEngine(enabled_families=("ta", "derivatives")).build_features(df_btc, df_funding)
    assert set(ta_deriv.columns) == (EXPECTED_TA_COLS | EXPECTED_DERIV_COLS)

    # QuantFeaturePipeline con include_market_structure=False excluye ms_*
    pipe_no_ms = QuantFeaturePipeline(include_market_structure=False)
    feat_no_ms = pipe_no_ms.build_features(df_btc, df_eth, df_funding)
    assert not any(c.startswith("ms_") for c in feat_no_ms.columns)

    # QuantFeaturePipeline por defecto incluye ms_*
    pipe_with_ms = QuantFeaturePipeline()
    feat_with_ms = pipe_with_ms.build_features(df_btc, df_eth, df_funding)
    assert EXPECTED_MS_COLS.issubset(set(feat_with_ms.columns))

    # Ninguna familia habilitada -> DataFrame vacío con mismo índice
    empty = FeatureEngine(enabled_families=()).build_features(df_btc, df_funding)
    assert list(empty.columns) == []
    assert empty.index.equals(df_btc.index)

    # Familia desconocida -> ValueError
    with pytest.raises(ValueError, match="desconocidas"):
        FeatureEngine(enabled_families=("ta", "nonexistent_family"))


def test_market_structure_causal_swing_confirmation_lag():
    """
    Un Swing High ocurrido en la barra `p` solo puede confirmarse en `t = p + swing_right`
    (NUNCA antes). Perturbar barras `> t` no altera `<= t`.
    """
    n = 35
    # Serie plana con rango constante (high=101, low=99 -> ATR=2.0 desde barra 13)
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)

    # Para evitar swings espurios por mesetas, creamos una rampa suave hasta p=20 y bajada después
    for i in range(n):
        base = 100.0 + 0.1 * i if i <= 20 else 102.0 - 0.1 * (i - 20)
        opens[i] = base
        closes[i] = base
        highs[i] = base + 1.0
        lows[i] = base - 1.0

    # Pico inequívoco en p = 20 con high = 120.0
    p = 20
    highs[p] = 120.0

    df = _make_ohlcv_from_arrays(opens, highs, lows, closes)
    swing_right = 3
    fe = FeatureEngine(
        enabled_families=("market_structure",),
        swing_left=3,
        swing_right=swing_right,
    )
    ms = fe.build_features(df)

    # En t = p, p+1, p+2 el pico en p=20 NO puede haber sido confirmado todavía
    for t_before in (p, p + 1, p + 2):
        assert np.isnan(ms["ms_dist_last_swing_high"].iloc[t_before])

    # Exactamente en t = p + swing_right = 23 se confirma el Swing High = 120.0
    t_confirm = p + swing_right
    assert np.isfinite(ms["ms_dist_last_swing_high"].iloc[t_confirm])
    assert ms["ms_dist_last_swing_high"].iloc[t_confirm] < 0.0  # close ~101.7 < 120.0


def test_market_structure_bos_choch_and_order_blocks():
    """
    Escenario sintético controlado que dispara en orden:
    1. Swing High confirmado + Swing Low confirmado.
    2. Bullish BOS (cierre > last_swing_high) + Bullish Order Block (última vela bajista previa).
    3. Nuevo Swing Low confirmado -> Bearish CHoCH (cambio de carácter bajista) + Bearish Order Block.
    4. Nuevo Swing High confirmado -> Bullish CHoCH (cambio de carácter alcista).
    """
    n = 55
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)

    # Rampa suave inicial para warm-up ATR (0..14) sin swings
    for i in range(15):
        base = 100.0 + 0.2 * i
        opens[i] = base
        closes[i] = base + 0.1
        highs[i] = base + 1.0
        lows[i] = base - 1.0

    # Swing High #1 en p=16 (high=110.0), confirmado en t=19 (swing_right=3)
    # Bajamos suavemente en 17..20
    opens[15], closes[15], highs[15], lows[15] = 103.0, 104.0, 105.0, 102.0
    opens[16], closes[16], highs[16], lows[16] = 105.0, 108.0, 110.0, 104.0
    opens[17], closes[17], highs[17], lows[17] = 107.0, 105.0, 108.0, 104.0
    opens[18], closes[18], highs[18], lows[18] = 105.0, 103.0, 106.0, 102.0
    opens[19], closes[19], highs[19], lows[19] = 103.0, 102.0, 104.0, 101.0  # confirma SH=110

    # Swing Low #1 en p=21 (low=95.0), confirmado en t=24
    opens[20], closes[20], highs[20], lows[20] = 102.0, 99.0, 103.0, 98.0
    opens[21], closes[21], highs[21], lows[21] = 99.0, 97.0, 100.0, 95.0  # SL=95
    opens[22], closes[22], highs[22], lows[22] = 97.0, 101.0, 102.0, 96.5
    opens[23], closes[23], highs[23], lows[23] = 101.0, 104.0, 105.0, 100.0
    opens[24], closes[24], highs[24], lows[24] = 104.0, 107.0, 108.0, 103.0  # confirma SL=95

    # Vela bajista en t=25 (será el Bullish OB!) seguida de Bullish BOS en t=26 (close=112 > 110)
    opens[25], closes[25], highs[25], lows[25] = 107.0, 105.0, 107.5, 104.5  # bajista: midpoint=106.0
    opens[26], closes[26], highs[26], lows[26] = 105.5, 112.0, 113.0, 105.0  # Bullish BOS!

    # Formamos Swing High #2 en p=28 (high=118), confirmado en t=31
    opens[27], closes[27], highs[27], lows[27] = 112.0, 114.0, 115.0, 111.0
    opens[28], closes[28], highs[28], lows[28] = 114.0, 116.0, 118.0, 113.0  # SH=118
    opens[29], closes[29], highs[29], lows[29] = 115.0, 110.0, 116.0, 109.0
    opens[30], closes[30], highs[30], lows[30] = 110.0, 106.0, 111.0, 105.0
    opens[31], closes[31], highs[31], lows[31] = 106.0, 103.0, 107.0, 102.0  # confirma SH=118

    # Vela alcista en t=32 (será Bearish OB!) seguida de Bearish CHoCH en t=33 (close=93 < SL=95)
    opens[32], closes[32], highs[32], lows[32] = 102.0, 104.0, 104.5, 101.5  # alcista: midpoint=103.0
    opens[33], closes[33], highs[33], lows[33] = 103.0, 93.0, 103.5, 92.0   # Bearish CHoCH!

    # Mantener estable 34..37 y luego romper por encima de SH=118 en t=38 -> Bullish CHoCH!
    for i in range(34, 38):
        opens[i], closes[i], highs[i], lows[i] = 95.0 + (i - 34) * 5.0, 98.0 + (i - 34) * 5.0, 99.0 + (i - 34) * 5.0, 94.0 + (i - 34) * 5.0
    opens[38], closes[38], highs[38], lows[38] = 115.0, 121.0, 122.0, 114.0  # Bullish CHoCH!

    for i in range(39, n):
        opens[i], closes[i], highs[i], lows[i] = 120.0, 120.5, 121.0, 119.5

    df = _make_ohlcv_from_arrays(opens, highs, lows, closes)
    ms = FeatureEngine(
        enabled_families=("market_structure",),
        swing_left=3,
        swing_right=3,
    ).build_features(df)

    # 1. Bullish BOS en t=26 y Bullish OB en k=25
    assert ms["ms_bullish_bos"].iloc[26] == 1
    assert ms["ms_bullish_choch"].iloc[26] == 0
    assert ms["ms_bull_ob_age"].iloc[26] == pytest.approx(1.0)
    assert ms["ms_bull_ob_age"].iloc[27] == pytest.approx(2.0)
    assert ms["ms_dist_bull_ob"].iloc[26] > 0.0  # close=112 > midpoint OB=106

    # 2. Bearish CHoCH en t=33 (estructura previa era alcista) y Bearish OB en k=32
    assert ms["ms_bearish_choch"].iloc[33] == 1
    assert ms["ms_bearish_bos"].iloc[33] == 0
    assert ms["ms_bear_ob_age"].iloc[33] == pytest.approx(1.0)
    assert ms["ms_bear_ob_age"].iloc[34] == pytest.approx(2.0)
    assert ms["ms_dist_bear_ob"].iloc[33] < 0.0  # close=93 < midpoint OB=103

    # 3. Bullish CHoCH en t=38 (estructura previa era bajista)
    assert ms["ms_bullish_choch"].iloc[38] == 1
    assert ms["ms_bullish_bos"].iloc[38] == 0


def test_market_structure_equal_highs_lows_sweeps_fvg_and_premium_discount():
    """
    Escenario controlado que verifica:
    - Equal Highs y Equal Lows dentro de `equal_tol_atr * atr_14`.
    - Liquidity Sweeps alcista (perfora swing low y cierra arriba) y bajista (perfora swing high y cierra abajo).
    - Fair Value Gaps alcista (`low[t] > high[t-2]`) y bajista (`high[t] < low[t-2]`) con `ms_fvg_size_atr > 0`.
    - Premium / Discount en [0, 1] y clipping a [-0.5, 1.5].
    """
    n = 60
    # Oscilación limpia con dos techos iguales (~110.0 y ~110.1) y dos suelos iguales (~90.0 y ~90.1)
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 102.0)
    lows = np.full(n, 98.0)

    # Warmup ATR ~4.0 en 0..14
    for i in range(15):
        base = 100.0 + 0.1 * i
        opens[i], closes[i], highs[i], lows[i] = base, base, base + 2.0, base - 2.0

    # Primer Swing High en p=17 (high=110.0), confirmado en t=20
    highs[15:21] = [104.0, 107.0, 110.0, 107.0, 104.0, 102.0]
    lows[15:21] = [100.0, 103.0, 106.0, 103.0, 100.0, 98.0]
    opens[15:21] = [102.0, 105.0, 108.0, 105.0, 102.0, 100.0]
    closes[15:21] = [103.0, 106.0, 108.0, 104.0, 101.0, 99.0]

    # Primer Swing Low en p=23 (low=90.0), confirmado en t=26
    highs[21:27] = [100.0, 97.0, 94.0, 97.0, 100.0, 103.0]
    lows[21:27] = [96.0, 93.0, 90.0, 93.0, 96.0, 99.0]
    opens[21:27] = [98.0, 95.0, 92.0, 95.0, 98.0, 101.0]
    closes[21:27] = [97.0, 94.0, 92.0, 96.0, 99.0, 102.0]

    # Segundo Swing High en p=29 (high=110.1 -> |110.1 - 110.0| = 0.1 < 0.15 * ATR), confirmado en t=32
    highs[27:33] = [105.0, 108.0, 110.1, 108.0, 105.0, 102.0]
    lows[27:33] = [101.0, 104.0, 106.0, 104.0, 101.0, 98.0]
    opens[27:33] = [103.0, 106.0, 108.0, 106.0, 103.0, 100.0]
    closes[27:33] = [104.0, 107.0, 108.0, 105.0, 102.0, 100.0]

    # Segundo Swing Low en p=35 (low=90.1 -> |90.1 - 90.0| = 0.1 < 0.15 * ATR), confirmado en t=38
    highs[33:39] = [100.0, 97.0, 94.0, 97.0, 100.0, 102.0]
    lows[33:39] = [96.0, 93.0, 90.1, 93.0, 96.0, 98.0]
    opens[33:39] = [98.0, 95.0, 92.0, 95.0, 98.0, 100.0]
    closes[33:39] = [97.0, 94.0, 92.0, 96.0, 99.0, 100.1]  # close=100.1 -> midpoint exacto de [90.1, 110.1]

    # En t=39: Bullish Liquidity Sweep (low=88.0 < 90.1, close=95.0 > 90.1)
    opens[39], highs[39], lows[39], closes[39] = 96.0, 97.0, 88.0, 95.0

    # En t=40: Bearish Liquidity Sweep (high=112.5 > 110.1, close=108.0 < 110.1)
    opens[40], highs[40], lows[40], closes[40] = 107.0, 112.5, 106.0, 108.0

    # Bullish FVG en t=43: low[43] > high[41]
    opens[41], highs[41], lows[41], closes[41] = 100.0, 101.0, 99.0, 100.5
    opens[42], highs[42], lows[42], closes[42] = 101.0, 106.0, 100.5, 105.5
    opens[43], highs[43], lows[43], closes[43] = 105.5, 109.0, 105.0, 108.0  # low=105 > high[41]=101!

    # Bearish FVG en t=46: high[46] < low[44]
    opens[44], highs[44], lows[44], closes[44] = 108.0, 109.0, 106.0, 107.0
    opens[45], highs[45], lows[45], closes[45] = 106.5, 107.0, 100.0, 101.0
    opens[46], highs[46], lows[46], closes[46] = 101.0, 102.0, 98.0, 99.0  # high=102 < low[44]=106!

    # Test de clipping de Premium/Discount: precio muy por encima de last_sh=110.1 en t=47
    opens[47], highs[47], lows[47], closes[47] = 140.0, 142.0, 139.0, 141.0

    df = _make_ohlcv_from_arrays(opens, highs, lows, closes)
    ms = FeatureEngine(
        enabled_families=("market_structure",),
        swing_left=3,
        swing_right=3,
        equal_tol_atr=0.15,
    ).build_features(df)

    # Equal Highs confirmado en t=32
    assert ms["ms_equal_highs"].iloc[32] == 1
    # Equal Lows confirmado en t=38
    assert ms["ms_equal_lows"].iloc[38] == 1
    # Premium/Discount en midpoint exacto (100.1 dentro de [90.1, 110.1]) -> 0.5
    assert ms["ms_premium_discount_pos"].iloc[38] == pytest.approx(0.5, abs=1e-6)
    # Clipping superior a 1.5
    assert ms["ms_premium_discount_pos"].iloc[47] == pytest.approx(1.5, abs=1e-12)

    # Liquidity Sweeps
    assert ms["ms_bullish_sweep_strength"].iloc[39] > 0.0
    assert ms["ms_bearish_sweep_strength"].iloc[39] == 0.0
    assert ms["ms_bearish_sweep_strength"].iloc[40] > 0.0
    assert ms["ms_bullish_sweep_strength"].iloc[40] == 0.0

    # Fair Value Gaps
    assert ms["ms_bullish_fvg"].iloc[43] == 1
    assert ms["ms_bearish_fvg"].iloc[43] == 0
    assert ms["ms_fvg_size_atr"].iloc[43] > 0.0

    assert ms["ms_bearish_fvg"].iloc[46] == 1
    assert ms["ms_bullish_fvg"].iloc[46] == 0
    assert ms["ms_fvg_size_atr"].iloc[46] > 0.0


def test_optional_df_derivatives_causal_no_bfill(sample_ohlcv_and_funding):
    """
    Verifica Fase 3.4 (`df_derivatives` opcional):
    - Sin `df_derivatives`: no añade columnas extra ni falla.
    - Con `df_derivatives` empezando tarde: NaN estricto antes del primer dato (SIN `.bfill()`),
      y cálculo causal de `deriv_oi_change_24h`, `deriv_oi_zscore`, `deriv_basis`, `deriv_premium`, `deriv_liquidations`.
    """
    df_btc, _, df_funding = sample_ohlcv_and_funding

    # Sin df_derivatives
    base_deriv = FeatureEngine(enabled_families=("derivatives",)).build_features(
        df_btc, df_funding=df_funding
    )
    assert set(base_deriv.columns) == EXPECTED_DERIV_COLS

    # Con df_derivatives comenzando en la vela 60 (frecuencia horaria)
    late_start = df_btc.index[60]
    deriv_idx = pd.date_range(late_start, df_btc.index[-1], freq="1h", tz="UTC")
    n_d = len(deriv_idx)
    df_derivatives = pd.DataFrame(
        {
            "open_interest": np.linspace(10000.0, 12000.0, n_d),
            "basis": np.linspace(0.0005, 0.0015, n_d),
            "premium": np.linspace(-0.0002, 0.0004, n_d),
            "liquidations": np.linspace(50.0, 200.0, n_d),
        },
        index=deriv_idx,
    )

    ext_deriv = FeatureEngine(enabled_families=("derivatives",)).build_features(
        df_btc,
        df_funding=df_funding,
        df_derivatives=df_derivatives,
    )

    expected_ext = EXPECTED_DERIV_COLS | {
        "deriv_oi_change_24h",
        "deriv_oi_zscore",
        "deriv_basis",
        "deriv_premium",
        "deriv_liquidations",
    }
    assert set(ext_deriv.columns) == expected_ext

    # CERO .bfill(): antes de late_start todo debe ser NaN
    before = ext_deriv.index < late_start
    assert before.sum() == 60
    for col in ("deriv_oi_change_24h", "deriv_oi_zscore", "deriv_basis", "deriv_premium", "deriv_liquidations"):
        assert ext_deriv.loc[before, col].isna().all()

    # Después de late_start propagación causal ffill activa
    after = ext_deriv.index >= late_start
    assert ext_deriv.loc[after, "deriv_basis"].notna().all()
    assert ext_deriv.loc[after, "deriv_premium"].notna().all()
    assert ext_deriv.loc[after, "deriv_liquidations"].notna().all()
    # OI change 24h y zscore disponibles tras warmup de 96 velas desde late_start
    assert ext_deriv.iloc[60 + 96:]["deriv_oi_change_24h"].notna().all()
    assert ext_deriv.iloc[60 + 96:]["deriv_oi_zscore"].notna().all()


def test_context_engine_six_regimes_and_no_bfill(sample_ohlcv_and_funding):
    """
    Verifica los 6 regímenes causales completos de `ContextEngine` (Fase 4):
    1. Market Regime (`ctx_market_regime` in {-1, 0, 1})
    2. Volatility Regime (`ctx_vol_regime_pct`, `ctx_vol_regime_state` in {0, 1, 2})
    3. Liquidity Regime (`ctx_liquidity_regime_zscore`)
    4. BTC Regime (`ctx_btc_bull_regime`, `ctx_btc_momentum_24h`)
    5. Altcoin Regime (`ctx_eth_btc_rel_24h`, `ctx_alt_strength_regime`, sin `.bfill()`)
    6. Derivatives Regime (`ctx_deriv_funding_regime` in {-1, 0, 1}, sin `.bfill()`)
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding

    # Construimos un df_funding controlado con valores < -0.0001, neutral y > 0.0003 empezando en vela 40
    start_f = df_btc.index[40]
    f_idx = pd.date_range(start_f, df_btc.index[-1], freq="8h", tz="UTC")
    f_vals = np.zeros(len(f_idx), dtype=float)
    f_vals[0] = -0.0005  # -> -1
    if len(f_idx) > 1:
        f_vals[1] = 0.0001  # -> 0 (neutral)
    if len(f_idx) > 2:
        f_vals[2:] = 0.0006  # -> +1
    df_funding_ctrl = pd.DataFrame({"funding_rate": f_vals}, index=f_idx)

    # ETH empezando tarde en vela 60 para verificar cero .bfill() en ctx_alt_strength_regime
    df_eth_late = df_eth.iloc[60:].copy()

    ctx = ContextEngine().build_context(
        df_btc=df_btc,
        df_eth=df_eth_late,
        df_funding=df_funding_ctrl,
    )

    expected_ctx_cols = {
        "ctx_btc_bull_regime",
        "ctx_market_regime",
        "ctx_btc_momentum_24h",
        "ctx_vol_regime_pct",
        "ctx_vol_regime_state",
        "ctx_liquidity_regime_zscore",
        "ctx_eth_btc_rel_24h",
        "ctx_alt_strength_regime",
        "ctx_deriv_funding_regime",
        "ctx_fomc_event_24h",
    }
    assert expected_ctx_cols.issubset(set(ctx.columns))

    # 1. Market Regime valores válidos
    assert set(ctx["ctx_market_regime"].unique()).issubset({-1, 0, 1})

    # 2. Volatility Regime state coherente con pct
    valid_v = ctx["ctx_vol_regime_pct"].notna()
    assert valid_v.any()
    assert set(ctx.loc[valid_v, "ctx_vol_regime_state"].unique()).issubset({0.0, 1.0, 2.0})
    assert ctx.loc[~valid_v, "ctx_vol_regime_state"].isna().all()

    # 3. Liquidity Regime zscore finito tras warmup
    assert ctx["ctx_liquidity_regime_zscore"].iloc[150:].notna().all()

    # 4. BTC momentum 24h finito tras 96 velas
    assert ctx["ctx_btc_momentum_24h"].iloc[96:].notna().all()

    # 5. Altcoin Regime sin .bfill()
    assert ctx["ctx_alt_strength_regime"].iloc[:60].isna().all()
    assert set(ctx["ctx_alt_strength_regime"].iloc[60:].unique()).issubset({0.0, 1.0})

    # 6. Derivatives Regime sin .bfill() y umbrales exactos
    assert ctx["ctx_deriv_funding_regime"].iloc[:40].isna().all()
    assert ctx.loc[f_idx[0], "ctx_deriv_funding_regime"] == -1.0
    if len(f_idx) > 1:
        assert ctx.loc[f_idx[1], "ctx_deriv_funding_regime"] == 0.0
    if len(f_idx) > 2:
        assert ctx.loc[f_idx[2], "ctx_deriv_funding_regime"] == 1.0


def test_context_market_regime_sideways_bull_bear():
    """Verifica explícitamente los estados -1 (bearish), 0 (sideways < 0.003) y +1 (bullish) de ctx_market_regime."""
    n = 600
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="15min", tz="UTC")

    # Mercado plano -> sideways (0)
    flat_close = np.full(n, 50000.0)
    df_flat = _make_ohlcv_from_arrays(flat_close, flat_close + 50.0, flat_close - 50.0, flat_close)
    ctx_flat = ContextEngine().build_context(df_flat)
    assert (ctx_flat["ctx_market_regime"] == 0).all()

    # Tendencia alcista clara -> +1 al final
    bull_close = 50000.0 * np.exp(np.linspace(0.0, 0.20, n))
    df_bull = _make_ohlcv_from_arrays(bull_close, bull_close * 1.002, bull_close * 0.998, bull_close)
    ctx_bull = ContextEngine().build_context(df_bull)
    assert ctx_bull["ctx_market_regime"].iloc[-1] == 1

    # Tendencia bajista clara -> -1 al final
    bear_close = 50000.0 * np.exp(np.linspace(0.0, -0.20, n))
    df_bear = _make_ohlcv_from_arrays(bear_close, bear_close * 1.002, bear_close * 0.998, bear_close)
    ctx_bear = ContextEngine().build_context(df_bear)
    assert ctx_bear["ctx_market_regime"].iloc[-1] == -1


def test_no_bfill_when_funding_starts_later_than_btc(sample_ohlcv_and_funding):
    """
    Si `df_funding` comienza más tarde que `df_btc`, las barras anteriores a la primera
    observación de funding DEBEN permanecer como NaN (prohibido `.bfill()`).
    """
    df_btc, _, df_funding = sample_ohlcv_and_funding

    # Recortamos df_funding para que empiece 50 velas después del inicio de df_btc
    cutoff_ts = df_btc.index[50]
    df_funding_late = df_funding.loc[df_funding.index >= cutoff_ts].copy()
    assert not df_funding_late.empty
    first_funding_ts = df_funding_late.index[0]

    feat = FeatureEngine(enabled_families=("derivatives",)).build_features(df_btc, df_funding_late)

    before_mask = feat.index < first_funding_ts
    after_mask = feat.index >= first_funding_ts

    assert before_mask.sum() >= 50
    # Cero look-ahead: antes del primer funding todo debe ser NaN
    assert feat.loc[before_mask, "deriv_funding_rate"].isna().all()
    assert feat.loc[before_mask, "deriv_funding_zscore"].isna().all()
    assert feat.loc[before_mask, "deriv_funding_change_24h"].isna().all()

    # Desde el primer funding en adelante debe propagarse causalmente con ffill
    assert feat.loc[after_mask, "deriv_funding_rate"].notna().all()


def test_no_bfill_when_eth_starts_later_than_btc(sample_ohlcv_and_funding):
    """
    Si `df_eth` comienza más tarde que `df_btc`, `ctx_eth_btc_rel_24h` DEBE permanecer como NaN
    antes del primer dato de ETH (sin `.bfill()`).
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding

    # Recortamos df_eth para que empiece en la vela 80
    df_eth_late = df_eth.iloc[80:].copy()
    first_eth_ts = df_eth_late.index[0]

    ctx = ContextEngine().build_context(df_btc=df_btc, df_eth=df_eth_late)

    before_mask = ctx.index < first_eth_ts
    assert before_mask.sum() == 80
    # Antes de la primera vela de ETH (+ 4 velas de pct_change(4)) no puede haber datos rellenados
    assert ctx.loc[before_mask, "ctx_eth_btc_rel_24h"].isna().all()
    assert ctx.loc[before_mask, "ctx_alt_strength_regime"].isna().all()
    # A partir de vela 80 + 4 (y con ATR listo desde vela 13) debe haber valores válidos
    assert ctx.iloc[85:]["ctx_eth_btc_rel_24h"].notna().all()


def test_strict_causality_future_perturbation_invariance(sample_ohlcv_and_funding):
    """
    Modificar precios, volúmenes y funding a partir del índice T+1 NO debe alterar
    ninguna feature (incluyendo `ms_*` y `deriv_*`) ni columna de contexto en índices <= T.
    """
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    split_idx = 220
    split_ts = df_btc.index[split_idx]

    pipe = QuantFeaturePipeline()
    base = pipe.build_features(df_btc, df_eth, df_funding)

    df_btc_mod = df_btc.copy()
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("close")] *= 1.35
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("high")] *= 1.40
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("low")] *= 0.70
    df_btc_mod.iloc[split_idx + 1:, df_btc_mod.columns.get_loc("volume")] *= 5.0

    df_eth_mod = df_eth.copy()
    df_eth_mod.iloc[split_idx + 1:, df_eth_mod.columns.get_loc("close")] *= 0.50

    df_funding_mod = df_funding.copy()
    df_funding_mod.loc[df_funding_mod.index > split_ts, "funding_rate"] = 0.05

    perturbed = pipe.build_features(df_btc_mod, df_eth_mod, df_funding_mod)

    pd.testing.assert_frame_equal(
        base.iloc[: split_idx + 1],
        perturbed.iloc[: split_idx + 1],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_context_engine_available_at_temporal_control(sample_ohlcv_and_funding):
    """
    Verifica el Control Temporal Cuádruple (`available_at <= bar_timestamp`):
    1. Un evento programado con `available_at` anterior a la ventana se activa en `[event-24h, event+2h]`.
    2. Un evento publicado tarde (`available_at` en mitad de la ventana o tras el evento) NO se activa
       en ninguna vela con `bar_timestamp < available_at`.
    3. Soporta eventos macro genéricos (`category="CPI"`, etc.).
    """
    df_btc, df_eth, _ = sample_ohlcv_and_funding
    # df_btc cubre 2026-01-01 00:00 UTC .. 2026-01-05 03:45 UTC (400 velas de 15m)

    # Evento A: FOMC programado el 2026-01-02 18:00 UTC, disponible desde 30 días antes
    ev_scheduled = ExternalEvent(
        name="FOMC_SCHEDULED",
        event_time="2026-01-02 18:00:00",
        publication_time="2025-12-03 18:00:00",
        effective_time="2026-01-02 18:00:00",
        available_at="2025-12-03 18:00:00",
        category="FOMC",
    )

    # Evento B: Evento macro genérico (CPI) el 2026-01-04 12:00 UTC, solo disponible desde 2026-01-04 11:00 UTC
    ev_late_pub = ExternalEvent(
        name="CPI_RELEASE",
        event_time="2026-01-04 12:00:00",
        publication_time="2026-01-04 11:00:00",
        effective_time="2026-01-04 12:00:00",
        available_at="2026-01-04 11:00:00",
        category="CPI",
    )

    # Evento C: Evento publicado después del cierre de la ventana -> nunca debe activarse
    ev_retroactive = ExternalEvent(
        name="RETROACTIVE_LEAK",
        event_time="2026-01-03 12:00:00",
        publication_time="2026-01-03 16:00:00",
        effective_time="2026-01-03 12:00:00",
        available_at="2026-01-03 16:00:00",
    )

    engine = ContextEngine(events=[ev_scheduled, ev_late_pub, ev_retroactive])
    ctx = engine.build_context(df_btc=df_btc, df_eth=df_eth)
    flag = ctx["ctx_fomc_event_24h"]

    # Evento A: activo en 2026-01-01 18:00..2026-01-02 20:00
    assert flag.loc[pd.Timestamp("2026-01-01 17:45:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-01 18:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 18:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 20:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-02 20:15:00", tz="UTC")] == 0

    # Evento C (retroactivo): nunca se activa en su ventana
    assert flag.loc[pd.Timestamp("2026-01-03 12:00:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-03 14:00:00", tz="UTC")] == 0

    # Evento B (publicación tardía a las 11:00): 0 antes de las 11:00, 1 desde las 11:00 hasta las 14:00
    assert flag.loc[pd.Timestamp("2026-01-04 10:45:00", tz="UTC")] == 0
    assert flag.loc[pd.Timestamp("2026-01-04 11:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 12:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 14:00:00", tz="UTC")] == 1
    assert flag.loc[pd.Timestamp("2026-01-04 14:15:00", tz="UTC")] == 0

