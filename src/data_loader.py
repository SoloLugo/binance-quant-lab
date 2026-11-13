import glob
import os
import numpy as np
import pandas as pd


class HistoricalDataLoader:
    """
    Lee y consolida los archivos CSV mensuales de Binance Vision
    almacenados en data/raw/symbol=*/
    """
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir

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

    def load_klines(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        pattern = os.path.join(self.raw_dir, f"symbol={symbol}", f"{symbol}-{interval}-*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para {pattern}")

        # Columnas estándar de Binance Vision para Klines
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]

        df_list = []
        for f in files:
            # Detectamos si el CSV tiene cabecera o empieza directo con números
            first_line = pd.read_csv(f, nrows=1, header=None)
            has_header = isinstance(first_line.iloc[0, 0], str)

            if has_header:
                df_part = pd.read_csv(f)
                # Estandarizamos nombres si vienen con cabecera
                df_part.columns = cols[:len(df_part.columns)]
            else:
                df_part = pd.read_csv(f, header=None, names=cols)
            df_list.append(df_part)

        df = pd.concat(df_list, ignore_index=True)

        # Normalizamos fila por fila a milisegundos antes de convertir a UTC datetime
        df["open_time"] = self._normalize_epoch_ms(df["open_time"])
        df = df.dropna(subset=["open_time"])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df[numeric_cols].dropna()

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