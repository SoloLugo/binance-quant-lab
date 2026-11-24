import hashlib
import json
import os
import numpy as np
import pandas as pd
import pytest

from data_loader import HistoricalDataLoader
from data_quality import DataQualityGate, timeframe_to_timedelta
from download_data import (
    compute_file_sha256,
    download_and_extract_zip,
    is_valid_csv_file,
    write_download_manifest,
)


def _make_clean_klines(n: int = 100, freq: str = "15min", base_price: float = 70000.0) -> pd.DataFrame:
    """Genera un DataFrame OHLCV limpio y determinista en UTC."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq=freq, tz="UTC")
    rets = rng.normal(0.0, 0.002, size=n)
    closes = base_price * np.exp(np.cumsum(rets))
    opens = np.empty(n, dtype=float)
    opens[0] = base_price
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(20.0, 5.0, size=n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(20.0, 5.0, size=n))
    volume = rng.uniform(10.0, 250.0, size=n)
    quote_volume = volume * closes
    count = rng.integers(100, 1000, size=n).astype(float)
    taker_buy_volume = volume * 0.52

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
        index=idx,
    )


def _write_kline_csv(
    path: str,
    df: pd.DataFrame,
    unit: str = "ms",
    include_header: bool = True,
) -> None:
    """Escribe un CSV en formato Binance Vision (12 columnas) en ms o us, con o sin cabecera."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    open_ms = ((df.index - epoch) // pd.Timedelta("1ms")).to_numpy(dtype=np.int64)
    close_ms = open_ms + 15 * 60 * 1000 - 1
    if unit == "us":
        open_ts = open_ms * 1000
        close_ts = close_ms * 1000
    else:
        open_ts = open_ms
        close_ts = close_ms

    raw = pd.DataFrame(
        {
            "open_time": open_ts,
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
            "volume": df["volume"].to_numpy(),
            "close_time": close_ts,
            "quote_volume": df["quote_volume"].to_numpy(),
            "count": df["count"].to_numpy(),
            "taker_buy_volume": df["taker_buy_volume"].to_numpy(),
            "taker_buy_quote_volume": (df["taker_buy_volume"] * df["close"]).to_numpy(),
            "ignore": np.zeros(len(df), dtype=float),
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path, index=False, header=include_header)


def test_timeframe_to_timedelta_supported_and_invalid():
    assert timeframe_to_timedelta("1m") == pd.Timedelta(minutes=1)
    assert timeframe_to_timedelta("5m") == pd.Timedelta(minutes=5)
    assert timeframe_to_timedelta("15m") == pd.Timedelta(minutes=15)
    assert timeframe_to_timedelta("30m") == pd.Timedelta(minutes=30)
    assert timeframe_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert timeframe_to_timedelta("4h") == pd.Timedelta(hours=4)
    assert timeframe_to_timedelta("1d") == pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="Timeframe no soportado"):
        timeframe_to_timedelta("2h")


def test_clean_dataset_passes_data_quality_gate():
    df = _make_clean_klines(n=120)
    gate = DataQualityGate()
    report = gate.validate_klines(df, symbol="BTCUSDT", timeframe="15m")

    assert report.passed is True
    assert report.row_count == 120
    assert report.duplicate_count == 0
    assert report.missing_bars_count == 0
    assert report.invalid_ohlc_count == 0
    assert report.negative_volume_count == 0
    assert report.invalid_price_count == 0
    assert report.out_of_order_count == 0
    assert report.extreme_outlier_count == 0
    assert report.errors == []
    gate.assert_valid(report)


def test_empty_and_naive_timezone_blocked():
    gate = DataQualityGate()

    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty_report = gate.validate_klines(empty_df, symbol="BTCUSDT", timeframe="15m")
    assert empty_report.passed is False
    with pytest.raises(ValueError, match="DataQualityGate BLOQUEADO"):
        gate.assert_valid(empty_report)

    naive_df = _make_clean_klines(n=20)
    naive_df.index = naive_df.index.tz_localize(None)
    naive_report = gate.validate_klines(naive_df, symbol="BTCUSDT", timeframe="15m")
    assert naive_report.passed is False
    assert any("naive" in e.lower() for e in naive_report.errors)
    with pytest.raises(ValueError):
        gate.assert_valid(naive_report)


def test_duplicates_detected_and_blocked():
    df = _make_clean_klines(n=50)
    df_dup = pd.concat([df, df.iloc[[10, 20]]])
    df_dup = df_dup.sort_index()

    gate = DataQualityGate()
    report = gate.validate_klines(df_dup, symbol="BTCUSDT", timeframe="15m")

    assert report.duplicate_count == 2
    assert report.passed is False
    with pytest.raises(ValueError, match="duplicates=2"):
        gate.assert_valid(report)


def test_gaps_and_missing_bars_detected_and_blocked():
    df = _make_clean_klines(n=100)
    # Eliminamos 5 velas consecutivas (5 / 100 = 5% > 1% max_missing_ratio)
    df_gap = df.drop(df.index[30:35])

    gate = DataQualityGate()
    report = gate.validate_klines(df_gap, symbol="BTCUSDT", timeframe="15m", max_missing_ratio=0.01)

    assert report.missing_bars_count == 5
    assert len(report.gap_details) == 1
    assert report.gap_details[0]["missing_bars"] == 5
    assert report.passed is False
    with pytest.raises(ValueError, match="missing_bars=5"):
        gate.assert_valid(report)


def test_inconsistent_ohlc_detected_and_blocked():
    df = _make_clean_klines(n=40)
    # Caso 1: low > min(open, close)
    df.iloc[5, df.columns.get_loc("low")] = max(df.iloc[5]["open"], df.iloc[5]["close"]) + 5.0
    # Caso 2: high < max(open, close)
    df.iloc[10, df.columns.get_loc("high")] = min(df.iloc[10]["open"], df.iloc[10]["close"]) - 5.0
    # Caso 3: low > high
    df.iloc[15, df.columns.get_loc("low")] = df.iloc[15]["high"] + 50.0

    gate = DataQualityGate()
    report = gate.validate_klines(df, symbol="BTCUSDT", timeframe="15m")

    assert report.invalid_ohlc_count == 3
    assert report.passed is False
    with pytest.raises(ValueError, match="invalid_ohlc=3"):
        gate.assert_valid(report)


def test_negative_volume_detected_and_blocked():
    df = _make_clean_klines(n=40)
    df.iloc[7, df.columns.get_loc("volume")] = -12.5
    df.iloc[12, df.columns.get_loc("volume")] = -0.001

    gate = DataQualityGate()
    report = gate.validate_klines(df, symbol="BTCUSDT", timeframe="15m")

    assert report.negative_volume_count == 2
    assert report.passed is False
    with pytest.raises(ValueError, match="negative_volume=2"):
        gate.assert_valid(report)


def test_invalid_prices_zero_negative_nan_inf_detected_and_blocked():
    df = _make_clean_klines(n=40)
    df.iloc[2, df.columns.get_loc("close")] = 0.0
    df.iloc[4, df.columns.get_loc("low")] = -100.0
    df.iloc[6, df.columns.get_loc("open")] = np.nan
    df.iloc[8, df.columns.get_loc("high")] = np.inf

    gate = DataQualityGate()
    report = gate.validate_klines(df, symbol="BTCUSDT", timeframe="15m")

    assert report.invalid_price_count == 4
    assert report.passed is False
    with pytest.raises(ValueError, match="invalid_price=4"):
        gate.assert_valid(report)


def test_out_of_order_timestamps_detected_and_blocked():
    df = _make_clean_klines(n=30)
    # Intercambiamos dos filas para romper el orden cronológico ascendente
    order = list(range(30))
    order[10], order[11] = order[11], order[10]
    df_unordered = df.iloc[order]

    gate = DataQualityGate()
    report = gate.validate_klines(df_unordered, symbol="BTCUSDT", timeframe="15m")

    assert report.out_of_order_count >= 1
    assert report.passed is False
    with pytest.raises(ValueError, match="out_of_order="):
        gate.assert_valid(report)


def test_extreme_outliers_flagged_as_warning_without_blocking():
    df = _make_clean_klines(n=40)
    # Simulamos un salto del +45% en una vela pero con OHLC internamente consistente
    prev_close = df.iloc[19]["close"]
    spike_close = prev_close * 1.45
    df.iloc[20, df.columns.get_loc("open")] = prev_close
    df.iloc[20, df.columns.get_loc("close")] = spike_close
    df.iloc[20, df.columns.get_loc("high")] = spike_close + 10.0
    df.iloc[20, df.columns.get_loc("low")] = prev_close - 10.0
    # Mantenemos consistencia en la vela siguiente para que vuelva suavemente o quede arriba
    df.iloc[21:, df.columns.get_loc("open")] *= 1.45
    df.iloc[21:, df.columns.get_loc("high")] *= 1.45
    df.iloc[21:, df.columns.get_loc("low")] *= 1.45
    df.iloc[21:, df.columns.get_loc("close")] *= 1.45

    gate = DataQualityGate()
    report = gate.validate_klines(df, symbol="BTCUSDT", timeframe="15m", extreme_return_threshold=0.30)

    assert report.extreme_outlier_count >= 1
    assert len(report.warnings) >= 1
    assert report.passed is True
    gate.assert_valid(report)


def test_validate_alignment_aligned_and_misaligned(sample_ohlcv_and_funding):
    df_btc, df_eth, df_funding = sample_ohlcv_and_funding
    gate = DataQualityGate()

    aligned = gate.validate_alignment(df_btc, df_eth, df_funding)
    assert aligned["passed"] is True
    assert aligned["overlap_ratio"] == pytest.approx(1.0)
    assert aligned["primary_missing_in_context_count"] == 0
    assert aligned["funding_covers_primary"] is True

    # Desalineación completa: desplazamos ETH 30 días hacia adelante
    df_eth_shifted = df_eth.copy()
    df_eth_shifted.index = df_eth_shifted.index + pd.Timedelta(days=30)
    misaligned = gate.validate_alignment(df_btc, df_eth_shifted, df_funding)
    assert misaligned["passed"] is False
    assert misaligned["overlap_bars"] == 0
    assert len(misaligned["errors"]) >= 1


def test_loader_load_and_validate_mixed_ms_us_headers_reproducible_and_immutable(tmp_path):
    raw_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "manifests"
    symbol_dir = raw_dir / "symbol=BTCUSDT"

    df_all = _make_clean_klines(n=80)
    part1 = df_all.iloc[:40]
    part2 = df_all.iloc[40:]

    # Archivo 1: timestamps en milisegundos CON cabecera
    csv1 = str(symbol_dir / "BTCUSDT-15m-2026-01.csv")
    _write_kline_csv(csv1, part1, unit="ms", include_header=True)

    # Archivo 2: timestamps en microsegundos SIN cabecera
    csv2 = str(symbol_dir / "BTCUSDT-15m-2026-02.csv")
    _write_kline_csv(csv2, part2, unit="us", include_header=False)

    loader = HistoricalDataLoader(raw_dir=str(raw_dir), manifests_dir=str(manifests_dir))

    # Ejecución 1 con save_manifest=True
    df1, manifest1 = loader.load_and_validate_klines(
        symbol="BTCUSDT",
        interval="15m",
        dataset_version="BTCUSDT_15m_dataset_v1.0",
        save_manifest=True,
    )
    assert len(df1) == 80
    assert str(df1.index.tz) == "UTC"
    assert manifest1.quality_passed is True
    assert manifest1.dataset_version == "BTCUSDT_15m_dataset_v1.0"
    assert manifest1.row_count == 80
    assert len(manifest1.dataset_hash) == 64
    assert set(manifest1.file_hashes.keys()) == {
        "BTCUSDT-15m-2026-01.csv",
        "BTCUSDT-15m-2026-02.csv",
    }
    manifest_file = manifests_dir / "BTCUSDT_15m_dataset_v1.0.json"
    assert manifest_file.exists()

    # Ejecución 2 consecutiva sobre los mismos datos -> hash idéntico y sin error
    df2, manifest2 = loader.load_and_validate_klines(
        symbol="BTCUSDT",
        interval="15m",
        dataset_version="BTCUSDT_15m_dataset_v1.0",
        save_manifest=True,
    )
    assert manifest1.dataset_hash == manifest2.dataset_hash
    pd.testing.assert_frame_equal(df1, df2)

    # Modificamos los datos fuente (nuevo close) y tratamos de sobrescribir la misma versión
    part2_modified = part2.copy()
    part2_modified.iloc[0, part2_modified.columns.get_loc("close")] += 1.0
    part2_modified.iloc[0, part2_modified.columns.get_loc("high")] = max(
        part2_modified.iloc[0]["high"],
        part2_modified.iloc[0]["close"] + 1.0,
    )
    _write_kline_csv(csv2, part2_modified, unit="us", include_header=False)

    with pytest.raises(FileExistsError, match="Conflicto de inmutabilidad"):
        loader.load_and_validate_klines(
            symbol="BTCUSDT",
            interval="15m",
            dataset_version="BTCUSDT_15m_dataset_v1.0",
            save_manifest=True,
        )

    # Con nueva versión sí debe permitir guardar
    _, manifest_v1_1 = loader.load_and_validate_klines(
        symbol="BTCUSDT",
        interval="15m",
        dataset_version="BTCUSDT_15m_dataset_v1.1",
        save_manifest=True,
    )
    assert manifest_v1_1.dataset_hash != manifest1.dataset_hash
    assert (manifests_dir / "BTCUSDT_15m_dataset_v1.1.json").exists()


def test_loader_validate_flag_blocks_corrupted_ohlc_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    symbol_dir = raw_dir / "symbol=BTCUSDT"
    df_bad = _make_clean_klines(n=30)
    df_bad.iloc[5, df_bad.columns.get_loc("low")] = df_bad.iloc[5]["high"] + 100.0
    _write_kline_csv(str(symbol_dir / "BTCUSDT-15m-2026-01.csv"), df_bad, unit="ms", include_header=True)

    loader = HistoricalDataLoader(raw_dir=str(raw_dir), manifests_dir=str(tmp_path / "manifests"))

    # Por defecto validate=False mantiene compatibilidad hacia atrás
    df_unvalidated = loader.load_klines("BTCUSDT", interval="15m")
    assert len(df_unvalidated) == 30

    # Con validate=True debe bloquear por OHLC inconsistente
    with pytest.raises(ValueError, match="invalid_ohlc=1"):
        loader.load_klines("BTCUSDT", interval="15m", validate=True)


def test_download_integrity_helpers_sha256_corruption_and_no_overwrite(tmp_path):
    valid_csv = tmp_path / "BTCUSDT-15m-2026-01.csv"
    content = b"open_time,open,high,low,close,volume\n1704067200000,70000,70100,69900,70050,12.5\n"
    valid_csv.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(str(valid_csv)) == expected_sha
    assert is_valid_csv_file(str(valid_csv)) is True

    # Archivo vacío y archivo binario corrupto deben ser detectados
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_bytes(b"")
    assert is_valid_csv_file(str(empty_csv)) is False

    corrupt_csv = tmp_path / "corrupt.csv"
    corrupt_csv.write_bytes(b"\x00\x01\x02\xffnot_a_csv")
    assert is_valid_csv_file(str(corrupt_csv)) is False

    # overwrite=False no sobrescribe un archivo válido existente ni realiza peticiones de red
    res = download_and_extract_zip(
        url="https://data.binance.vision/nonexistent.zip",
        dest_folder=str(tmp_path),
        expected_csv_name="BTCUSDT-15m-2026-01.csv",
        overwrite=False,
    )
    assert res["success"] is True
    assert res["status"] == "skipped_existing"
    assert res["sha256"] == expected_sha
    assert valid_csv.read_bytes() == content

    # Manifiesto de descarga en JSON
    manifest_path = write_download_manifest([res], manifests_dir=str(tmp_path / "manifests"))
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["summary"]["total"] == 1
    assert manifest_data["summary"]["skipped_existing"] == 1
    assert manifest_data["files"][0]["sha256"] == expected_sha
