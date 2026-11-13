from typing import Dict, List
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from config import settings


class BenterQuantBrain:
    """
    Motor de Probabilidad, Calibración Isotónica Regularizada en Cola,
    Valor Esperado (EV) y Dimensionamiento de Riesgo (Criterio de Kelly Fraccionado).
    """
    def __init__(
        self,
        feature_cols: List[str],
        tp_mult: float = settings.tp_multiple,
        sl_mult: float = settings.sl_multiple,
        roundtrip_fee: float = settings.roundtrip_fee,
        direction: str = "LONG"
    ):
        self.feature_cols = feature_cols
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.roundtrip_fee = roundtrip_fee
        self.direction = direction.upper()

        # Hiperparámetros regularizados para evitar sobreajuste en series ruidosas
        self.base_clf = LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.03,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=settings.n_jobs
        )
        # Calibrador isotónico monótono acotado en [0.01, 0.75] para evitar probabilidades irreales (0.999)
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.75, out_of_bounds="clip")
        self.raw_upper_cap: float = 1.0
        self.is_fitted = False

    def fit_and_calibrate(
        self,
        df_train: pd.DataFrame,
        y_train: pd.Series,
        df_calib: pd.DataFrame,
        y_calib: pd.Series,
        purge_bars: int = 24
    ) -> None:
        """
        Paso 1: Entrena LightGBM en el bloque histórico de entrenamiento (df_train).
        Paso 2: Predice probabilidades crudas en el bloque futuro separado (df_calib)
                con Winsorización de cola superior (mínimo 4 muestras) para evitar que
                un único outlier en validación genere un escalón artificial de p=0.999.
        """
        X_tr = df_train[self.feature_cols]
        X_ca = df_calib[self.feature_cols]

        # 1. Entrenamiento del clasificador base
        self.base_clf.fit(X_tr, y_train.astype(int))

        # 2. Obtención de probabilidades sin calibrar en el conjunto de validación temporal
        raw_calib_probs = self.base_clf.predict_proba(X_ca)[:, 1]
        if len(raw_calib_probs) >= 10:
            self.raw_upper_cap = float(np.sort(raw_calib_probs)[-4])
        else:
            self.raw_upper_cap = 1.0

        # 3. Ajuste del mapeo empírico de probabilidades reales P(TP before SL)
        clipped_calib = np.minimum(raw_calib_probs, self.raw_upper_cap)
        self.calibrator.fit(clipped_calib, y_calib.values.astype(float))
        self.is_fitted = True

    def predict_calibrated_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Retorna la probabilidad calibrada real P(TP before SL) lista para entrar
        en la ecuación de Valor Esperado de Bill Benter.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado y calibrado antes de predecir.")

        X = df_features[self.feature_cols]
        raw_probs = self.base_clf.predict_proba(X)[:, 1]
        cap = getattr(self, "raw_upper_cap", 1.0)
        return self.calibrator.predict(np.minimum(raw_probs, cap))

    def compute_ev_and_kelly(
        self,
        p_win: np.ndarray,
        atr_pct: np.ndarray,
        kelly_fraction: float = settings.kelly_fraction
    ) -> pd.DataFrame:
        """
        Aplica la fórmula de Bill Benter adaptada a futuros con comisiones:
        EV = p * b_net - (1 - p) * l_net
        """
        # Distancia porcentual de 1 unidad de riesgo (1R)
        r_dist_pct = np.maximum(atr_pct * self.sl_mult, 1e-4)

        # Impacto de las comisiones de apertura y cierre expresado en unidades de R
        fee_in_r = self.roundtrip_fee / r_dist_pct

        # Cuota neta de ganancia (b) y pérdida neta (l) en unidades R
        b_net = (self.tp_mult / self.sl_mult) - fee_in_r
        l_net = 1.0 + fee_in_r

        p_loss = 1.0 - p_win
        ev_r = (p_win * b_net) - (p_loss * l_net)

        # Fracción de Kelly: f* = (p*b_net - q*l_net) / (b_net * l_net)
        raw_kelly = np.where(b_net > 0, ev_r / (b_net * l_net), 0.0)
        safe_kelly = np.clip(raw_kelly * kelly_fraction, 0.0, settings.max_risk_per_trade)

        return pd.DataFrame({
            "p_tp_before_sl": p_win,
            "b_net_odds": b_net,
            "l_net_odds": l_net,
            "ev_in_r": ev_r,
            "risk_fraction_kelly": safe_kelly
        })

    def evaluate_model(
        self,
        df_test: pd.DataFrame,
        y_test: pd.Series,
        ev_min_threshold: float = 0.10,
        non_overlapping: bool = True
    ) -> Dict[str, float]:
        """
        Evalúa el modelo fuera de muestra tanto en señales brutas como respetando
        la restricción de una posición activa a la vez (`non_overlapping=True`)
        y el retorno exacto de las 3 barreras (TP, SL y Time-Stop).
        """
        p_win = self.predict_calibrated_proba(df_test)
        atr_pct = df_test["atr_pct"].values
        ev_df = self.compute_ev_and_kelly(p_win, atr_pct)

        ev_vals = ev_df["ev_in_r"].values
        b_net_vals = ev_df["b_net_odds"].values
        fee_in_r_vals = self.roundtrip_fee / np.maximum(atr_pct * self.sl_mult, 1e-4)
        y_vals = y_test.values

        has_exact_outcomes = ("realized_r_gross" in df_test.columns) and ("exit_bars" in df_test.columns)
        gross_r_vals = df_test["realized_r_gross"].values if has_exact_outcomes else None
        exit_bars_vals = df_test["exit_bars"].values.astype(int) if has_exact_outcomes else np.full(len(df_test), 24, dtype=int)

        taken_indices = []
        next_free_idx = 0

        for idx in range(len(df_test)):
            if non_overlapping and idx < next_free_idx:
                continue
            if ev_vals[idx] > ev_min_threshold:
                taken_indices.append(idx)
                if non_overlapping:
                    next_free_idx = idx + max(1, int(exit_bars_vals[idx]))

        n_trades = len(taken_indices)
        if n_trades > 0:
            sel = np.array(taken_indices, dtype=int)
            if gross_r_vals is not None:
                realized_r = gross_r_vals[sel] - fee_in_r_vals[sel]
            else:
                realized_r = np.where(
                    y_vals[sel] == 1,
                    b_net_vals[sel],
                    -(1.0 + fee_in_r_vals[sel])
                )
            avg_realized_r = float(np.mean(realized_r))
            cum_r = float(np.sum(realized_r))
            win_rate = float(np.mean(realized_r > 0))
            avg_ev_pred = float(np.mean(ev_vals[sel]))
        else:
            avg_realized_r, cum_r, win_rate, avg_ev_pred = 0.0, 0.0, 0.0, 0.0

        return {
            "AUC": round(float(roc_auc_score(y_test, p_win)), 4),
            "LogLoss": round(float(log_loss(y_test, p_win)), 4),
            "Brier": round(float(brier_score_loss(y_test, p_win)), 4),
            "Trades": n_trades,
            "WinRate_Taken": round(win_rate * 100, 2),
            "Avg_EV_Pred_R": round(avg_ev_pred, 3),
            "Avg_Realized_R": round(avg_realized_r, 3),
            "Total_Net_R": round(cum_r, 2)
        }
