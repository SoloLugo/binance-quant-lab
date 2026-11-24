"""
Adaptador compatible de alto nivel que delega en:
- `FeatureEngine` (src/feature_engine.py)
- `ContextEngine` (src/context_engine.py)
- `LabelEngine`   (src/label_engine.py)

Mantiene compatibilidad con los scripts y tests existentes (`QuantFeaturePipeline`),
eliminando código duplicado y cualquier uso de `.bfill()`.
"""
from typing import Optional, Sequence
import pandas as pd

from context_engine import ContextEngine, ExternalEvent, FOMC_DATES
from contracts import LabelSpec
from feature_engine import FeatureEngine
from label_engine import LabelEngine


class QuantFeaturePipeline:
    """
    Adaptador compatible que orquesta las capas causales de features, contexto y etiquetado:
    1. Technical Analysis (`ta_*`) via `FeatureEngine`
    2. Smart Money / Microestructura (`sm_*`) via `FeatureEngine`
    3. Derivatives / Funding (`deriv_*`) via `FeatureEngine`
    4. Context Engine / Macro / Régimen (`ctx_*`) via `ContextEngine`
    5. Triple-Barrier ejecutable via `LabelEngine`
    """

    def __init__(
        self,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        horizon_bars: int = 24,
        entry_mode: str = "next_open",
        exclude_ambiguous: bool = True,
        enabled_families: Optional[Sequence[str]] = None,
        events: Optional[Sequence[ExternalEvent]] = None,
    ):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.horizon_bars = horizon_bars
        self.fomc_timestamps = pd.to_datetime(FOMC_DATES, utc=True)

        self.feature_engine = FeatureEngine(enabled_families=enabled_families)
        self.context_engine = ContextEngine(events=events)
        self.label_spec = LabelSpec(
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            horizon_bars=horizon_bars,
            entry_mode=entry_mode,
            exclude_ambiguous=exclude_ambiguous,
        )
        self.label_engine = LabelEngine(spec=self.label_spec)

    def build_features(
        self,
        df_btc: pd.DataFrame,
        df_eth: Optional[pd.DataFrame] = None,
        df_funding: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Delega la construcción causal de features y contexto a `FeatureEngine` y `ContextEngine`."""
        feat = self.feature_engine.build_features(df_btc=df_btc, df_funding=df_funding)
        atr_pct = feat["atr_pct"] if "atr_pct" in feat.columns else None
        ctx = self.context_engine.build_context(df_btc=df_btc, df_eth=df_eth, atr_pct=atr_pct)
        return pd.concat([feat, ctx], axis=1)

    def compute_triple_barrier_outcomes(
        self,
        df_btc: pd.DataFrame,
        atr_pct: pd.Series,
        direction: str = "LONG",
    ) -> pd.DataFrame:
        """Delega el etiquetado Triple-Barrier ejecutable a `LabelEngine`."""
        return self.label_engine.compute_outcomes(
            df_btc=df_btc,
            atr_pct=atr_pct,
            direction=direction,
        )

    def create_triple_barrier_labels(
        self,
        df_btc: pd.DataFrame,
        atr_pct: pd.Series,
        direction: str = "LONG",
    ) -> pd.Series:
        """Devuelve la serie `target_tp_before_sl` generada por `LabelEngine`."""
        outcomes = self.compute_triple_barrier_outcomes(df_btc, atr_pct, direction=direction)
        return outcomes["target_tp_before_sl"]
