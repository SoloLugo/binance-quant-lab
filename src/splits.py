"""
Particionamiento temporal estricto con purging, embargo y Test Final Sellado.

Garantiza que no exista fuga temporal (temporal leakage) entre entrenamiento,
calibración, validación y el bloque de Test Final Sellado.
"""
from typing import List, Optional, Tuple
import pandas as pd

from contracts import SplitSpec


def verify_no_temporal_overlap(
    idx_a: pd.DatetimeIndex,
    idx_b: pd.DatetimeIndex,
    purge_bars: int,
    embargo_bars: int,
    bar_delta: pd.Timedelta = pd.Timedelta(minutes=15),
) -> None:
    """
    Verifica estrictamente que `idx_a` e `idx_b` estén ordenados cronológicamente,
    sean disjuntos y mantengan una brecha temporal mínima de
    `(purge_bars + embargo_bars) * bar_delta`.

    Lanza `ValueError` si alguna condición se incumple.
    """
    if not isinstance(idx_a, pd.DatetimeIndex) or not isinstance(idx_b, pd.DatetimeIndex):
        raise ValueError("idx_a e idx_b deben ser instancias de pd.DatetimeIndex.")
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError("Los índices temporales idx_a e idx_b no pueden estar vacíos.")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError(
            f"purge_bars ({purge_bars}) y embargo_bars ({embargo_bars}) deben ser >= 0."
        )
    if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
        raise ValueError("bar_delta debe ser un pd.Timedelta estrictamente positivo.")

    if not idx_a.is_monotonic_increasing:
        raise ValueError("idx_a no está ordenado de forma monótona creciente.")
    if not idx_b.is_monotonic_increasing:
        raise ValueError("idx_b no está ordenado de forma monótona creciente.")
    if not idx_a.is_unique:
        raise ValueError("idx_a contiene timestamps duplicados.")
    if not idx_b.is_unique:
        raise ValueError("idx_b contiene timestamps duplicados.")

    if (idx_a.tz is None) != (idx_b.tz is None):
        raise ValueError("idx_a e idx_b deben compartir compatibilidad de zona horaria.")

    overlap = idx_a.intersection(idx_b)
    if len(overlap) > 0:
        raise ValueError(
            f"Solapamiento temporal detectado: {len(overlap)} timestamps compartidos "
            f"(ej. {overlap[0]})."
        )

    a_max = idx_a.max()
    b_min = idx_b.min()
    if b_min <= a_max:
        raise ValueError(
            f"Orden temporal violado o solapamiento: idx_b.min() ({b_min}) <= idx_a.max() ({a_max})."
        )

    required_gap = (purge_bars + embargo_bars) * bar_delta
    min_allowed_b_start = a_max + required_gap
    if b_min < min_allowed_b_start:
        actual_gap = b_min - a_max
        raise ValueError(
            f"Brecha temporal insuficiente entre splits: actual={actual_gap}, "
            f"requerida>={required_gap} (purge_bars={purge_bars}, "
            f"embargo_bars={embargo_bars}, bar_delta={bar_delta})."
        )


def _validate_utc_datetime_index(df: pd.DataFrame) -> None:
    """Valida que el DataFrame tenga un DatetimeIndex UTC ordenado y sin duplicados."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("El DataFrame de entrada debe ser un pd.DataFrame no vacío.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame debe estar indexado por un pd.DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("El DatetimeIndex debe tener zona horaria UTC (tz-aware).")
    if df.index[0].utcoffset() != pd.Timedelta(0):
        raise ValueError(
            f"El DatetimeIndex debe estar en UTC (offset 0), recibido tz={df.index.tz}."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("El DatetimeIndex del DataFrame debe estar ordenado cronológicamente.")
    if not df.index.is_unique:
        raise ValueError("El DatetimeIndex del DataFrame no puede contener duplicados.")


class TemporalDatasetPartitioner:
    """
    Particionador temporal que aísla primero el Test Final Sellado y expone
    splits purgados y con embargo exclusivamente sobre el bloque de desarrollo.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        purge_bars: int = 24,
        embargo_bars: int = 24,
        sealed_test_ratio: float = 0.15,
        sealed_test_start: Optional[pd.Timestamp] = None,
        bar_delta: pd.Timedelta = pd.Timedelta(minutes=15),
        horizon_bars: Optional[int] = None,
    ) -> None:
        _validate_utc_datetime_index(df)

        if purge_bars < 0 or embargo_bars < 0:
            raise ValueError("purge_bars y embargo_bars deben ser enteros >= 0.")
        if horizon_bars is not None:
            if purge_bars < horizon_bars or embargo_bars < horizon_bars:
                raise ValueError(
                    f"purge_bars ({purge_bars}) y embargo_bars ({embargo_bars}) deben ser "
                    f">= horizon_bars ({horizon_bars})."
                )
        if not isinstance(bar_delta, pd.Timedelta) or bar_delta <= pd.Timedelta(0):
            raise ValueError("bar_delta debe ser un pd.Timedelta estrictamente positivo.")

        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)
        self.bar_delta = bar_delta
        self._gap_bars = self.purge_bars + self.embargo_bars
        self._sealed_access_log: List[str] = []

        df_copy = df.copy()

        # 1. Separar PRIMERO el bloque de Test Final Sellado
        if sealed_test_start is not None:
            ts_start = pd.Timestamp(sealed_test_start)
            if ts_start.tzinfo is None:
                ts_start = ts_start.tz_localize("UTC")
            else:
                ts_start = ts_start.tz_convert("UTC")

            sealed_df = df_copy.loc[df_copy.index >= ts_start].copy()
            if sealed_df.empty:
                raise ValueError(
                    f"sealed_test_start ({ts_start}) no dejó filas para el Test Final Sellado."
                )

            if self._gap_bars > 0:
                dev_cutoff = sealed_df.index[0] - self._gap_bars * self.bar_delta
                dev_df = df_copy.loc[df_copy.index < dev_cutoff].copy()
            else:
                dev_df = df_copy.loc[df_copy.index < sealed_df.index[0]].copy()
        else:
            if not (0.0 < sealed_test_ratio < 1.0):
                raise ValueError(
                    f"sealed_test_ratio debe estar en (0, 1), recibido {sealed_test_ratio}."
                )
            n = len(df_copy)
            sealed_size = max(1, int(round(n * sealed_test_ratio)))
            sealed_start_idx = n - sealed_size
            dev_end_idx = sealed_start_idx - self._gap_bars

            if dev_end_idx <= 0:
                raise ValueError(
                    f"Dataset demasiado pequeño (n={n}) para separar sealed_test_ratio="
                    f"{sealed_test_ratio} con gap={self._gap_bars} barras."
                )

            dev_df = df_copy.iloc[:dev_end_idx].copy()
            sealed_df = df_copy.iloc[sealed_start_idx:].copy()

        if dev_df.empty or sealed_df.empty:
            raise ValueError(
                "No fue posible construir bloques no vacíos de desarrollo y Test Final Sellado."
            )

        verify_no_temporal_overlap(
            dev_df.index,
            sealed_df.index,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            bar_delta=self.bar_delta,
        )

        self._dev_df = dev_df
        self._sealed_test_df = sealed_df

    @property
    def dev_size(self) -> int:
        return len(self._dev_df)

    @property
    def sealed_test_size(self) -> int:
        return len(self._sealed_test_df)

    def get_sealed_test(
        self,
        unlock_sealed_test: bool = False,
        reason: str = "",
    ) -> pd.DataFrame:
        """
        Devuelve una copia del Test Final Sellado únicamente si se desbloquea
        explícitamente con `unlock_sealed_test=True` y un `reason` no vacío.
        """
        if not unlock_sealed_test or not isinstance(reason, str) or not reason.strip():
            raise PermissionError(
                "ACCESO DENEGADO AL TEST FINAL SELLADO: requiere unlock_sealed_test=True "
                "y una justificación explícita no vacía en 'reason'. "
                "El Test Final Sellado nunca debe usarse para selección de modelos, features o umbrales."
            )
        self._sealed_access_log.append(reason.strip())
        return self._sealed_test_df.copy()

    def get_dev_split(
        self,
        train_ratio: float = 0.65,
        calib_ratio: float = 0.175,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide el bloque de desarrollo en `(df_train, df_calib, df_val)` respetando
        `purge_bars` y `embargo_bars` estrictos entre `train -> calib` y `calib -> val`.
        """
        if train_ratio <= 0.0 or calib_ratio <= 0.0 or (train_ratio + calib_ratio) >= 1.0:
            raise ValueError(
                f"Ratios inválidos: train_ratio={train_ratio}, calib_ratio={calib_ratio}. "
                "Deben ser positivos y sumar < 1.0."
            )

        n_dev = len(self._dev_df)
        usable = n_dev - 2 * self._gap_bars
        if usable < 3:
            raise ValueError(
                f"Bloque de desarrollo insuficiente (n_dev={n_dev}) para 3 splits "
                f"con gap={self._gap_bars} barras entre cada uno."
            )

        n_train = int(usable * train_ratio)
        n_calib = int(usable * calib_ratio)
        n_val = usable - n_train - n_calib

        if n_train < 1 or n_calib < 1 or n_val < 1:
            raise ValueError(
                f"Tamaño insuficiente en alguno de los splits de desarrollo: "
                f"n_train={n_train}, n_calib={n_calib}, n_val={n_val}."
            )

        train_end = n_train
        calib_start = train_end + self._gap_bars
        calib_end = calib_start + n_calib
        val_start = calib_end + self._gap_bars

        df_train = self._dev_df.iloc[:train_end].copy()
        df_calib = self._dev_df.iloc[calib_start:calib_end].copy()
        df_val = self._dev_df.iloc[val_start:].copy()

        verify_no_temporal_overlap(
            df_train.index,
            df_calib.index,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            bar_delta=self.bar_delta,
        )
        verify_no_temporal_overlap(
            df_calib.index,
            df_val.index,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            bar_delta=self.bar_delta,
        )
        verify_no_temporal_overlap(
            df_val.index,
            self._sealed_test_df.index,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            bar_delta=self.bar_delta,
        )

        return df_train, df_calib, df_val

    def get_walk_forward_folds(
        self,
        n_splits: int = 4,
        min_train_size: Optional[int] = None,
        calib_ratio_of_fold: float = 0.35,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Genera folds expansivos `(df_train, df_calib, df_val)` exclusivamente sobre
        el bloque de desarrollo (jamás tocando el Test Final Sellado), con purga y
        embargo verificados en cada fold.
        """
        if n_splits < 1:
            raise ValueError(f"n_splits debe ser >= 1, recibido {n_splits}.")
        if not (0.0 < calib_ratio_of_fold < 1.0):
            raise ValueError(
                f"calib_ratio_of_fold debe estar en (0, 1), recibido {calib_ratio_of_fold}."
            )

        n_dev = len(self._dev_df)
        min_step_per_fold = 2 * self._gap_bars + 2

        if min_train_size is None:
            initial_train = max(1, int(n_dev * 0.35))
            max_initial_train = n_dev - n_splits * min_step_per_fold
            if initial_train > max_initial_train:
                initial_train = max_initial_train
        else:
            initial_train = int(min_train_size)

        if initial_train < 1:
            raise ValueError(
                f"Bloque de desarrollo insuficiente (n_dev={n_dev}) para {n_splits} folds "
                f"walk-forward con gap={self._gap_bars} barras."
            )

        remaining = n_dev - initial_train
        step = remaining // n_splits
        if step < min_step_per_fold:
            raise ValueError(
                f"Tamaño de paso insuficiente por fold (step={step} < requerido={min_step_per_fold}) "
                f"para n_dev={n_dev}, min_train_size={initial_train}, n_splits={n_splits}, "
                f"gap={self._gap_bars}."
            )

        folds: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
        for k in range(n_splits):
            train_end = initial_train + k * step
            calib_start = train_end + self._gap_bars
            fold_end = initial_train + (k + 1) * step if k < n_splits - 1 else n_dev

            available_eval = (fold_end - calib_start) - self._gap_bars
            if available_eval < 2:
                raise ValueError(
                    f"Fold {k}: filas insuficientes para calibración y validación "
                    f"(available_eval={available_eval})."
                )

            n_calib = max(1, int(available_eval * calib_ratio_of_fold))
            if n_calib >= available_eval:
                n_calib = available_eval - 1

            calib_end = calib_start + n_calib
            val_start = calib_end + self._gap_bars
            val_end = fold_end

            df_train = self._dev_df.iloc[:train_end].copy()
            df_calib = self._dev_df.iloc[calib_start:calib_end].copy()
            df_val = self._dev_df.iloc[val_start:val_end].copy()

            verify_no_temporal_overlap(
                df_train.index,
                df_calib.index,
                purge_bars=self.purge_bars,
                embargo_bars=self.embargo_bars,
                bar_delta=self.bar_delta,
            )
            verify_no_temporal_overlap(
                df_calib.index,
                df_val.index,
                purge_bars=self.purge_bars,
                embargo_bars=self.embargo_bars,
                bar_delta=self.bar_delta,
            )
            verify_no_temporal_overlap(
                df_val.index,
                self._sealed_test_df.index,
                purge_bars=self.purge_bars,
                embargo_bars=self.embargo_bars,
                bar_delta=self.bar_delta,
            )

            folds.append((df_train, df_calib, df_val))

        return folds

    def describe_split(
        self,
        train_ratio: float = 0.65,
        calib_ratio: float = 0.175,
    ) -> SplitSpec:
        """
        Devuelve un `SplitSpec` inmutable con los límites temporales del split de
        desarrollo y del Test Final Sellado sin exponer los datos sellados.
        """
        df_train, df_calib, df_val = self.get_dev_split(
            train_ratio=train_ratio,
            calib_ratio=calib_ratio,
        )
        return SplitSpec(
            train_start=df_train.index[0].isoformat(),
            train_end=df_train.index[-1].isoformat(),
            calib_start=df_calib.index[0].isoformat(),
            calib_end=df_calib.index[-1].isoformat(),
            val_start=df_val.index[0].isoformat(),
            val_end=df_val.index[-1].isoformat(),
            sealed_test_start=self._sealed_test_df.index[0].isoformat(),
            sealed_test_end=self._sealed_test_df.index[-1].isoformat(),
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
        )
