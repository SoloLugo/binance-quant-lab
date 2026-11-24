import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import settings
from contracts import DataManifest
from data_quality import DataQualityGate
from download_data import compute_file_sha256


class HistoricalDataLoader:
    """
    Lee y consolida los archivos CSV mensuales de Binance Vision
    almacenados en data/raw/symbol=*/
    """

    KLINE_RAW_COLS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    KLINE_NUMERIC_COLS = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume",
    ]

    def __init__(self, raw_dir: str = "data/raw", manifests_dir: Optional[str] = None):
        self.raw_dir = raw_dir
        self.manifests_dir = manifests_dir or settings.manifests_dir

    @staticmethod
    def _normalize_epoch_ms(series: pd.Series) -> pd.Series:
        """
        Normaliza timestamps fila por fila a milisegundos (ms), evitando fallos
        cuando Binance Vision mezcla archivos mensuales en milisegundos (>1e11)
        con archivos en microsegundos (>1e14).
        """
        vals = pd.to_numeric(series, errors="coerce")
        # Si un valor está en microsegundos (> 1e14), lo convertimos a milisegundos
        vals_ms = np.where(vals > 1e14, np.floor(vals / 1000.0), vals)
        return pd.Series(vals_ms, index=series.index)

    @staticmethod
    def compute_dataframe_sha256(df: pd.DataFrame) -> str:
        """
        Calcula un hash SHA256 determinista sobre el DataFrame normalizado
        (índice UTC en milisegundos + nombres de columnas + valores float64 en orden canónico).
        """
        hasher = hashlib.sha256()
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        idx_ms = ((df.index - epoch) // pd.Timedelta("1ms")).to_numpy(dtype="<i8")
        hasher.update(b"index_ms:")
        hasher.update(np.ascontiguousarray(idx_ms).tobytes())

        for col in df.columns:
            hasher.update(f"|col:{col}:".encode("utf-8"))
            arr = np.ascontiguousarray(df[col].to_numpy(dtype="<f8"))
            hasher.update(arr.tobytes())

        return hasher.hexdigest()

    def _read_kline_files(self, symbol: str, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        pattern = os.path.join(self.raw_dir, f"symbol={symbol}", f"{symbol}-{interval}-*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para {pattern}")

        df_list = []
        for f in files:
            # Detectamos si el CSV tiene cabecera o empieza directo con números
            first_line = pd.read_csv(f, nrows=1, header=None)
            has_header = pd.isna(pd.to_numeric(first_line.iloc[0, 0], errors="coerce"))

            if has_header:
                df_part = pd.read_csv(f)
                # Estandarizamos nombres si vienen con cabecera
                df_part.columns = self.KLINE_RAW_COLS[:len(df_part.columns)]
            else:
                df_part = pd.read_csv(f, header=None, names=self.KLINE_RAW_COLS)
            df_list.append(df_part)

        df = pd.concat(df_list, ignore_index=True)

        # Normalizamos fila por fila a milisegundos antes de convertir a UTC datetime
        df["open_time"] = self._normalize_epoch_ms(df["open_time"])
        df = df.dropna(subset=["open_time"])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        for c in self.KLINE_NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df[self.KLINE_NUMERIC_COLS], files

    def load_klines(
        self,
        symbol: str,
        interval: str = "15m",
        validate: bool = False,
        max_missing_ratio: float = 0.01,
        extreme_return_threshold: float = 0.30,
    ) -> pd.DataFrame:
        df, _ = self._read_kline_files(symbol=symbol, interval=interval)
        if validate:
            gate = DataQualityGate()
            report = gate.validate_klines(
                df,
                symbol=symbol,
                timeframe=interval,
                max_missing_ratio=max_missing_ratio,
                extreme_return_threshold=extreme_return_threshold,
            )
            gate.assert_valid(report)
            return df
        return df.dropna()

    def _save_manifest(self, manifest: DataManifest) -> str:
        os.makedirs(self.manifests_dir, exist_ok=True)
        manifest_path = os.path.join(self.manifests_dir, f"{manifest.dataset_version}.json")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_hash = existing.get("dataset_hash")
            if existing_hash != manifest.dataset_hash:
                raise FileExistsError(
                    f"Conflicto de inmutabilidad de dataset: ya existe un manifiesto para "
                    f"'{manifest.dataset_version}' en '{manifest_path}' con dataset_hash="
                    f"{existing_hash}, distinto del nuevo dataset_hash={manifest.dataset_hash}. "
                    f"Incrementa dataset_version en lugar de sobrescribir silenciosamente."
                )
            return manifest_path

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        return manifest_path

    def load_and_validate_klines(
        self,
        symbol: str,
        interval: str = "15m",
        dataset_version: Optional[str] = None,
        save_manifest: bool = False,
        max_missing_ratio: float = 0.01,
        extreme_return_threshold: float = 0.30,
        source: str = "binance_vision_um_monthly",
        schema_version: str = "1.0",
    ) -> Tuple[pd.DataFrame, DataManifest]:
        """
        Carga klines desde disco, ejecuta DataQualityGate (lanzando ValueError si falla),
        calcula el hash SHA256 determinista del DataFrame normalizado y devuelve (df, DataManifest).
        Si save_manifest=True, persiste el manifiesto en manifests_dir sin sobrescribir
        silenciosamente versiones existentes con distinto dataset_hash (lanza FileExistsError).
        """
        df, files = self._read_kline_files(symbol=symbol, interval=interval)

        gate = DataQualityGate()
        report = gate.validate_klines(
            df,
            symbol=symbol,
            timeframe=interval,
            max_missing_ratio=max_missing_ratio,
            extreme_return_threshold=extreme_return_threshold,
        )
        gate.assert_valid(report)

        version_id = dataset_version or f"{symbol}_{interval}_dataset_v1.0"
        file_hashes: Dict[str, str] = {
            os.path.basename(f): compute_file_sha256(f) for f in files
        }
        dataset_hash = self.compute_dataframe_sha256(df)

        manifest = DataManifest(
            dataset_version=version_id,
            symbol=symbol,
            timeframe=interval,
            source=source,
            start_timestamp=df.index[0].isoformat(),
            end_timestamp=df.index[-1].isoformat(),
            row_count=int(len(df)),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            schema_version=schema_version,
            dataset_hash=dataset_hash,
            file_hashes=file_hashes,
            transformations=[
                "normalize_epoch_ms_us_to_utc",
                "sort_index_utc",
                "deduplicate_timestamp_keep_last",
                "cast_numeric_ohlcv",
                "data_quality_gate_validated",
            ],
            quality_passed=report.passed,
        )

        if save_manifest:
            self._save_manifest(manifest)

        return df, manifest

    def load_funding_rate(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        pattern = os.path.join(self.raw_dir, f"symbol={symbol}", f"{symbol}-fundingRate-*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return pd.DataFrame()

        df_list = [pd.read_csv(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)

        # Normalizamos las columnas de funding rate de Binance Vision
        time_col = "calc_time" if "calc_time" in df.columns else df.columns[0]
        rate_col = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]

        df[time_col] = self._normalize_epoch_ms(df[time_col])
        df = df.dropna(subset=[time_col])

        # Redondeamos hacia abajo al minuto más cercano (00:00, 08:00, 16:00 UTC)
        # porque Binance a menudo registra el funding con 1-15ms de desfase (ej. 00:00:00.002)
        df["timestamp"] = pd.to_datetime(df[time_col], unit="ms", utc=True).dt.floor("1min")
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")

        return df[["funding_rate"]].dropna()