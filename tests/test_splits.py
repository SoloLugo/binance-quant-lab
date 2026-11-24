import numpy as np
import pandas as pd
import pytest

from contracts import SplitSpec
from splits import TemporalDatasetPartitioner, verify_no_temporal_overlap


def _make_utc_dataset(n_bars: int = 2000, freq: str = "15min") -> pd.DataFrame:
    """Crea un DataFrame determinista indexado por UTC DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01 00:00:00", periods=n_bars, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "close": 50000.0 + np.cumsum(rng.normal(0.0, 20.0, size=n_bars)),
            "feature_a": rng.normal(0.0, 1.0, size=n_bars),
            "target": rng.integers(0, 2, size=n_bars),
        },
        index=idx,
    )


class TestVerifyNoTemporalOverlap:
    def test_accepts_exact_required_gap(self):
        bar_delta = pd.Timedelta(minutes=15)
        idx_a = pd.date_range("2025-01-01 00:00:00", periods=50, freq="15min", tz="UTC")
        purge_bars = 24
        embargo_bars = 24
        b_start = idx_a.max() + (purge_bars + embargo_bars) * bar_delta
        idx_b = pd.date_range(b_start, periods=30, freq="15min", tz="UTC")

        # No debe lanzar excepción
        verify_no_temporal_overlap(idx_a, idx_b, purge_bars, embargo_bars, bar_delta)

    def test_rejects_overlap_shared_timestamps(self):
        idx_a = pd.date_range("2025-01-01 00:00:00", periods=50, freq="15min", tz="UTC")
        idx_b = pd.date_range(idx_a[-5], periods=30, freq="15min", tz="UTC")

        with pytest.raises(ValueError, match="Solapamiento temporal"):
            verify_no_temporal_overlap(idx_a, idx_b, purge_bars=24, embargo_bars=24)

    def test_rejects_gap_smaller_than_purge_plus_embargo(self):
        bar_delta = pd.Timedelta(minutes=15)
        idx_a = pd.date_range("2025-01-01 00:00:00", periods=50, freq="15min", tz="UTC")
        purge_bars = 24
        embargo_bars = 24
        # 1 barra menos que la brecha requerida
        b_start = idx_a.max() + (purge_bars + embargo_bars - 1) * bar_delta
        idx_b = pd.date_range(b_start, periods=30, freq="15min", tz="UTC")

        with pytest.raises(ValueError, match="Brecha temporal insuficiente"):
            verify_no_temporal_overlap(idx_a, idx_b, purge_bars, embargo_bars, bar_delta)

    def test_rejects_reversed_order(self):
        idx_a = pd.date_range("2025-02-01 00:00:00", periods=20, freq="15min", tz="UTC")
        idx_b = pd.date_range("2025-01-01 00:00:00", periods=20, freq="15min", tz="UTC")

        with pytest.raises(ValueError, match="Orden temporal violado"):
            verify_no_temporal_overlap(idx_a, idx_b, purge_bars=0, embargo_bars=0)

    def test_rejects_unsorted_or_duplicate_index(self):
        idx_a = pd.date_range("2025-01-01 00:00:00", periods=10, freq="15min", tz="UTC")
        idx_b = pd.date_range("2025-01-05 00:00:00", periods=10, freq="15min", tz="UTC")

        unsorted_a = idx_a[::-1]
        with pytest.raises(ValueError, match="no está ordenado"):
            verify_no_temporal_overlap(unsorted_a, idx_b, purge_bars=2, embargo_bars=2)

        dup_a = pd.DatetimeIndex(list(idx_a) + [idx_a[-1]])
        with pytest.raises(ValueError, match="duplicados"):
            verify_no_temporal_overlap(dup_a, idx_b, purge_bars=2, embargo_bars=2)

    def test_rejects_empty_index(self):
        idx_a = pd.DatetimeIndex([], tz="UTC")
        idx_b = pd.date_range("2025-01-05 00:00:00", periods=10, freq="15min", tz="UTC")

        with pytest.raises(ValueError, match="vacíos"):
            verify_no_temporal_overlap(idx_a, idx_b, purge_bars=2, embargo_bars=2)


class TestTemporalDatasetPartitioner:
    def test_sealed_test_requires_explicit_unlock_and_reason(self):
        df = _make_utc_dataset(n_bars=1200)
        partitioner = TemporalDatasetPartitioner(df, purge_bars=24, embargo_bars=24)

        with pytest.raises(PermissionError):
            partitioner.get_sealed_test()

        with pytest.raises(PermissionError):
            partitioner.get_sealed_test(unlock_sealed_test=True, reason="")

        with pytest.raises(PermissionError):
            partitioner.get_sealed_test(unlock_sealed_test=True, reason="   ")

        with pytest.raises(PermissionError):
            partitioner.get_sealed_test(
                unlock_sealed_test=False,
                reason="Evaluación final única",
            )

        sealed = partitioner.get_sealed_test(
            unlock_sealed_test=True,
            reason="Evaluación final única tras congelar arquitectura",
        )
        assert len(sealed) == partitioner.sealed_test_size
        assert len(sealed) > 0

    def test_dev_split_respects_purge_and_embargo_everywhere(self):
        df = _make_utc_dataset(n_bars=1600)
        purge_bars = 24
        embargo_bars = 24
        bar_delta = pd.Timedelta(minutes=15)

        partitioner = TemporalDatasetPartitioner(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            sealed_test_ratio=0.15,
            bar_delta=bar_delta,
        )
        df_train, df_calib, df_val = partitioner.get_dev_split(
            train_ratio=0.65,
            calib_ratio=0.175,
        )
        df_sealed = partitioner.get_sealed_test(
            unlock_sealed_test=True,
            reason="Verificación unitaria de separación temporal",
        )

        verify_no_temporal_overlap(df_train.index, df_calib.index, purge_bars, embargo_bars, bar_delta)
        verify_no_temporal_overlap(df_calib.index, df_val.index, purge_bars, embargo_bars, bar_delta)
        verify_no_temporal_overlap(df_val.index, df_sealed.index, purge_bars, embargo_bars, bar_delta)

        assert len(df_train.index.intersection(df_sealed.index)) == 0
        assert len(df_calib.index.intersection(df_sealed.index)) == 0
        assert len(df_val.index.intersection(df_sealed.index)) == 0

    def test_sealed_test_start_timestamp_mode(self):
        df = _make_utc_dataset(n_bars=1200)
        purge_bars = 24
        embargo_bars = 24
        bar_delta = pd.Timedelta(minutes=15)
        sealed_start = df.index[1000]

        partitioner = TemporalDatasetPartitioner(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            sealed_test_start=sealed_start,
            bar_delta=bar_delta,
        )
        df_train, df_calib, df_val = partitioner.get_dev_split()
        df_sealed = partitioner.get_sealed_test(
            unlock_sealed_test=True,
            reason="Verificación unitaria sealed_test_start",
        )

        assert df_sealed.index[0] == sealed_start
        verify_no_temporal_overlap(df_train.index, df_calib.index, purge_bars, embargo_bars, bar_delta)
        verify_no_temporal_overlap(df_calib.index, df_val.index, purge_bars, embargo_bars, bar_delta)
        verify_no_temporal_overlap(df_val.index, df_sealed.index, purge_bars, embargo_bars, bar_delta)

    def test_walk_forward_folds_never_touch_sealed_test_and_verify_gaps(self):
        df = _make_utc_dataset(n_bars=2000)
        purge_bars = 24
        embargo_bars = 24
        bar_delta = pd.Timedelta(minutes=15)

        partitioner = TemporalDatasetPartitioner(
            df,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            sealed_test_ratio=0.15,
            bar_delta=bar_delta,
        )
        folds = partitioner.get_walk_forward_folds(n_splits=4, calib_ratio_of_fold=0.35)
        df_sealed = partitioner.get_sealed_test(
            unlock_sealed_test=True,
            reason="Verificación de aislamiento de folds walk-forward",
        )
        sealed_ts = set(df_sealed.index)

        assert len(folds) == 4
        prev_train_len = 0
        prev_val_end = None

        for df_train, df_calib, df_val in folds:
            # Ventana de entrenamiento estrictamente expansiva
            assert len(df_train) > prev_train_len
            prev_train_len = len(df_train)

            # Ventanas de validación cronológicas y disjuntas
            if prev_val_end is not None:
                assert df_val.index.min() > prev_val_end
            prev_val_end = df_val.index.max()

            # Purging y embargo estrictos dentro de cada fold y contra sealed_test
            verify_no_temporal_overlap(df_train.index, df_calib.index, purge_bars, embargo_bars, bar_delta)
            verify_no_temporal_overlap(df_calib.index, df_val.index, purge_bars, embargo_bars, bar_delta)
            verify_no_temporal_overlap(df_val.index, df_sealed.index, purge_bars, embargo_bars, bar_delta)

            # Ningún timestamp del sealed_test aparece en ningún fold de desarrollo
            assert sealed_ts.isdisjoint(df_train.index)
            assert sealed_ts.isdisjoint(df_calib.index)
            assert sealed_ts.isdisjoint(df_val.index)

    def test_describe_split_returns_valid_split_spec(self):
        df = _make_utc_dataset(n_bars=1200)
        partitioner = TemporalDatasetPartitioner(df, purge_bars=24, embargo_bars=24)

        spec = partitioner.describe_split()
        df_train, df_calib, df_val = partitioner.get_dev_split()
        df_sealed = partitioner.get_sealed_test(
            unlock_sealed_test=True,
            reason="Verificación de describe_split",
        )

        assert isinstance(spec, SplitSpec)
        assert spec.train_start == df_train.index[0].isoformat()
        assert spec.train_end == df_train.index[-1].isoformat()
        assert spec.calib_start == df_calib.index[0].isoformat()
        assert spec.calib_end == df_calib.index[-1].isoformat()
        assert spec.val_start == df_val.index[0].isoformat()
        assert spec.val_end == df_val.index[-1].isoformat()
        assert spec.sealed_test_start == df_sealed.index[0].isoformat()
        assert spec.sealed_test_end == df_sealed.index[-1].isoformat()
        assert spec.purge_bars == 24
        assert spec.embargo_bars == 24

    def test_rejects_naive_or_non_utc_index_and_insufficient_horizon_purge(self):
        df_naive = _make_utc_dataset(n_bars=400)
        df_naive.index = df_naive.index.tz_localize(None)
        with pytest.raises(ValueError, match="UTC"):
            TemporalDatasetPartitioner(df_naive)

        df_ny = _make_utc_dataset(n_bars=400)
        df_ny.index = df_ny.index.tz_convert("America/New_York")
        with pytest.raises(ValueError, match="UTC"):
            TemporalDatasetPartitioner(df_ny)

        df_utc = _make_utc_dataset(n_bars=600)
        with pytest.raises(ValueError, match="horizon_bars"):
            TemporalDatasetPartitioner(df_utc, purge_bars=12, embargo_bars=24, horizon_bars=24)
