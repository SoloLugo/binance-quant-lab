"""
Data Quality Gate para Binance Quant Lab (Fase 1).
Valida integridad temporal, consistencia OHLCV, valores inválidos y alineación
entre fuentes antes de permitir el entrenamiento o la evaluación.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from contracts import DataQualityReport


SUPPORTED_TIMEFRAMES: Dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """
    Convierte un string de timeframe estándar de Binance a pd.Timedelta.
    Soporta: 1m, 5m, 15m, 30m, 1h, 4h, 1d.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Timeframe no soportado: '{timeframe}'. "
            f"Valores permitidos: {list(SUPPORTED_TIMEFRAMES.keys())}"
        )
    return SUPPORTED_TIMEFRAMES[timeframe]


class DataQualityGate:
    """
    Puerta de control de calidad de datos.
    Detecta errores críticos (que bloquean el pipeline) y advertencias
    (ej. retornos extremos sospechosos).
    """

    def validate_klines(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "15m",
        max_missing_ratio: float = 0.01,
        extreme_return_threshold: float = 0.30,
    ) -> DataQualityReport:
        """
        Ejecuta la batería completa de validaciones sobre un DataFrame de klines.
        """
        expected_delta = timeframe_to_timedelta(timeframe)
        errors: List[str] = []
        warnings: List[str] = []
        gap_details: List[Dict[str, Any]] = []

        row_count = int(len(df))
        if row_count == 0:
            errors.append("El DataFrame está vacío (row_count == 0).")
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                row_count=0,
                duplicate_count=0,
                missing_bars_count=0,
                invalid_ohlc_count=0,
                negative_volume_count=0,
                invalid_price_count=0,
                out_of_order_count=0,
                extreme_outlier_count=0,
                passed=False,
                errors=errors,
                warnings=warnings,
                gap_details=gap_details,
            )

        # 1. Validación de índice temporal UTC
        is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        is_utc = False
        if not is_datetime_index:
            errors.append(f"El índice debe ser pd.DatetimeIndex UTC, recibido: {type(df.index).__name__}.")
        else:
            if df.index.tz is None:
                errors.append("El índice DatetimeIndex es naive (sin zona horaria UTC).")
            elif str(df.index.tz) not in ("UTC", "utc", "datetime.timezone.utc"):
                errors.append(f"El índice DatetimeIndex debe estar en UTC, recibido: {df.index.tz}.")
            else:
                is_utc = True

        # 2. Orden temporal y duplicados
        out_of_order_count = 0
        duplicate_count = int(df.index.duplicated().sum())
        if duplicate_count > 0:
            errors.append(f"Se detectaron {duplicate_count} timestamps duplicados.")

        missing_bars_count = 0
        missing_ratio = 0.0

        if is_datetime_index:
            if row_count > 1:
                idx_series = pd.Series(df.index)
                raw_diffs = idx_series.diff().iloc[1:]
                out_of_order_count = int((raw_diffs < pd.Timedelta(0)).sum())
                if out_of_order_count > 0:
                    errors.append(
                        f"El índice temporal no está ordenado ascendentemente "
                        f"({out_of_order_count} saltos hacia atrás)."
                    )

            # 3. Detección de velas faltantes y gaps sobre índice único y ordenado
            unique_sorted_idx = df.index.unique().sort_values()
            if len(unique_sorted_idx) > 1:
                sorted_diffs = pd.Series(unique_sorted_idx).diff().iloc[1:]
                gap_mask = sorted_diffs > expected_delta
                if gap_mask.any():
                    gap_positions = np.where(gap_mask.to_numpy())[0]
                    for pos in gap_positions:
                        prev_ts = unique_sorted_idx[pos]
                        curr_ts = unique_sorted_idx[pos + 1]
                        diff = curr_ts - prev_ts
                        missing_in_gap = max(1, int(np.ceil(diff / expected_delta)) - 1)
                        missing_bars_count += missing_in_gap
                        gap_details.append(
                            {
                                "start": prev_ts.isoformat(),
                                "end": curr_ts.isoformat(),
                                "missing_bars": int(missing_in_gap),
                                "gap_seconds": float(diff.total_seconds()),
                            }
                        )

                total_expected = len(unique_sorted_idx) + missing_bars_count
                missing_ratio = (missing_bars_count / total_expected) if total_expected > 0 else 0.0
                if missing_ratio > max_missing_ratio:
                    errors.append(
                        f"Ratio de velas faltantes ({missing_ratio:.4%}) excede el máximo permitido "
                        f"({max_missing_ratio:.4%}); missing_bars_count={missing_bars_count}."
                    )
                elif missing_bars_count > 0:
                    warnings.append(
                        f"Se detectaron {missing_bars_count} velas faltantes ({missing_ratio:.4%}), "
                        f"dentro del umbral permitido ({max_missing_ratio:.4%})."
                    )

        # 4. Validación de columnas OHLCV requeridas
        missing_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Faltan columnas OHLCV obligatorias: {missing_cols}.")
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                row_count=row_count,
                duplicate_count=duplicate_count,
                missing_bars_count=missing_bars_count,
                invalid_ohlc_count=0,
                negative_volume_count=0,
                invalid_price_count=0,
                out_of_order_count=out_of_order_count,
                extreme_outlier_count=0,
                passed=False,
                errors=errors,
                warnings=warnings,
                gap_details=gap_details,
            )

        open_s = pd.to_numeric(df["open"], errors="coerce")
        high_s = pd.to_numeric(df["high"], errors="coerce")
        low_s = pd.to_numeric(df["low"], errors="coerce")
        close_s = pd.to_numeric(df["close"], errors="coerce")
        vol_s = pd.to_numeric(df["volume"], errors="coerce")

        # 5. Precios inválidos (<= 0, NaN o inf)
        price_frame = pd.DataFrame(
            {"open": open_s, "high": high_s, "low": low_s, "close": close_s},
            index=df.index,
        )
        invalid_price_mask = (~np.isfinite(price_frame)) | (price_frame <= 0)
        invalid_price_count = int(invalid_price_mask.any(axis=1).sum())
        if invalid_price_count > 0:
            errors.append(
                f"Se detectaron {invalid_price_count} filas con precios inválidos (<= 0, NaN o inf)."
            )

        # 6. Consistencia OHLC: low <= min(open, close), high >= max(open, close), low <= high
        finite_ohlc = np.isfinite(open_s) & np.isfinite(high_s) & np.isfinite(low_s) & np.isfinite(close_s)
        min_oc = np.minimum(open_s, close_s)
        max_oc = np.maximum(open_s, close_s)
        invalid_ohlc_mask = finite_ohlc & ((low_s > min_oc) | (high_s < max_oc) | (low_s > high_s))
        invalid_ohlc_count = int(invalid_ohlc_mask.sum())
        if invalid_ohlc_count > 0:
            errors.append(
                f"Se detectaron {invalid_ohlc_count} filas con relaciones OHLC inconsistentes "
                f"(low > min(open,close), high < max(open,close) o low > high)."
            )

        # 7. Volúmenes negativos (o no finitos)
        vol_cols = [c for c in ("volume", "quote_volume", "taker_buy_volume") if c in df.columns]
        neg_vol_mask = pd.Series(False, index=df.index)
        for vc in vol_cols:
            s_vc = pd.to_numeric(df[vc], errors="coerce")
            neg_vol_mask = neg_vol_mask | (s_vc < 0) | (~np.isfinite(s_vc))
        negative_volume_count = int(neg_vol_mask.sum())
        if negative_volume_count > 0:
            errors.append(
                f"Se detectaron {negative_volume_count} filas con volumen negativo o no finito."
            )

        # 8. Valores extremos sospechosos (warning, no bloquea por sí solo)
        valid_close = close_s[np.isfinite(close_s) & (close_s > 0)]
        if is_datetime_index:
            valid_close = valid_close.sort_index()
        extreme_outlier_count = 0
        if len(valid_close) > 1:
            abs_returns = valid_close.pct_change().abs()
            extreme_outlier_count = int((abs_returns > extreme_return_threshold).sum())
            if extreme_outlier_count > 0:
                warnings.append(
                    f"Se detectaron {extreme_outlier_count} retornos extremos "
                    f"(|ret| > {extreme_return_threshold:.2%}). Revisar posibles anomalías de mercado."
                )

        passed = (
            row_count > 0
            and is_utc
            and out_of_order_count == 0
            and duplicate_count == 0
            and invalid_ohlc_count == 0
            and negative_volume_count == 0
            and invalid_price_count == 0
            and missing_ratio <= max_missing_ratio
        )

        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            row_count=row_count,
            duplicate_count=duplicate_count,
            missing_bars_count=missing_bars_count,
            invalid_ohlc_count=invalid_ohlc_count,
            negative_volume_count=negative_volume_count,
            invalid_price_count=invalid_price_count,
            out_of_order_count=out_of_order_count,
            extreme_outlier_count=extreme_outlier_count,
            passed=passed,
            errors=errors,
            warnings=warnings,
            gap_details=gap_details,
        )

    def validate_alignment(
        self,
        df_primary: pd.DataFrame,
        df_context: pd.DataFrame,
        df_funding: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Verifica el solapamiento temporal y las desalineaciones entre el activo principal
        (ej. BTCUSDT), el activo de contexto (ej. ETHUSDT) y la serie de Funding Rate (8h).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df_primary.empty:
            errors.append("df_primary está vacío.")
        if df_context.empty:
            errors.append("df_context está vacío.")

        if errors:
            return {
                "passed": False,
                "primary_rows": int(len(df_primary)),
                "context_rows": int(len(df_context)),
                "funding_rows": int(len(df_funding)),
                "overlap_bars": 0,
                "overlap_ratio": 0.0,
                "primary_missing_in_context_count": int(len(df_primary)),
                "context_missing_in_primary_count": int(len(df_context)),
                "funding_covers_primary": False,
                "errors": errors,
                "warnings": warnings,
            }

        p_idx = df_primary.index.unique().sort_values()
        c_idx = df_context.index.unique().sort_values()

        overlap_idx = p_idx.intersection(c_idx)
        primary_missing_in_context = p_idx.difference(c_idx)
        context_missing_in_primary = c_idx.difference(p_idx)

        overlap_bars = int(len(overlap_idx))
        overlap_ratio = overlap_bars / len(p_idx) if len(p_idx) > 0 else 0.0

        if overlap_bars == 0:
            errors.append("No existe solapamiento temporal entre df_primary y df_context.")
        elif len(primary_missing_in_context) > 0 or len(context_missing_in_primary) > 0:
            warnings.append(
                f"Desalineación parcial entre primary y context: "
                f"{len(primary_missing_in_context)} velas de primary sin context, "
                f"{len(context_missing_in_primary)} velas de context sin primary."
            )

        funding_start = None
        funding_end = None
        funding_covers_primary = False
        if df_funding is None or df_funding.empty:
            warnings.append("df_funding está vacío; las features de funding dependerán de fallback.")
        else:
            f_idx = df_funding.index.unique().sort_values()
            funding_start = f_idx[0].isoformat()
            funding_end = f_idx[-1].isoformat()
            # Funding se publica cada 8h; comprobamos solapamiento de rango con tolerancia de 8h
            has_funding_overlap = (f_idx[0] <= p_idx[-1]) and (f_idx[-1] >= p_idx[0] - pd.Timedelta(hours=8))
            funding_covers_primary = bool(has_funding_overlap)
            if not funding_covers_primary:
                errors.append(
                    f"df_funding ({funding_start} -> {funding_end}) no cubre el rango temporal "
                    f"de df_primary ({p_idx[0].isoformat()} -> {p_idx[-1].isoformat()})."
                )

        passed = len(errors) == 0
        return {
            "passed": passed,
            "primary_start": p_idx[0].isoformat(),
            "primary_end": p_idx[-1].isoformat(),
            "primary_rows": int(len(df_primary)),
            "context_start": c_idx[0].isoformat(),
            "context_end": c_idx[-1].isoformat(),
            "context_rows": int(len(df_context)),
            "funding_start": funding_start,
            "funding_end": funding_end,
            "funding_rows": int(len(df_funding)) if df_funding is not None else 0,
            "overlap_start": overlap_idx[0].isoformat() if overlap_bars > 0 else None,
            "overlap_end": overlap_idx[-1].isoformat() if overlap_bars > 0 else None,
            "overlap_bars": overlap_bars,
            "overlap_ratio": float(overlap_ratio),
            "primary_missing_in_context_count": int(len(primary_missing_in_context)),
            "context_missing_in_primary_count": int(len(context_missing_in_primary)),
            "funding_covers_primary": funding_covers_primary,
            "errors": errors,
            "warnings": warnings,
        }

    def assert_valid(self, report: DataQualityReport) -> None:
        """
        Lanza ValueError detallado si el reporte no supera los controles críticos.
        """
        if not report.passed:
            error_lines = "\n  - ".join(report.errors) if report.errors else "Fallo de calidad no especificado."
            raise ValueError(
                f"DataQualityGate BLOQUEADO para {report.symbol} ({report.timeframe}): "
                f"rows={report.row_count}, duplicates={report.duplicate_count}, "
                f"missing_bars={report.missing_bars_count}, invalid_ohlc={report.invalid_ohlc_count}, "
                f"negative_volume={report.negative_volume_count}, invalid_price={report.invalid_price_count}, "
                f"out_of_order={report.out_of_order_count}.\n"
                f"Errores críticos:\n  - {error_lines}"
            )
