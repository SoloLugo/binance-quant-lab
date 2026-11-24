from typing import Dict, List
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_engine import PROB_EPS, ProbabilityCalibrator
from config import settings
from splits import verify_no_temporal_overlap


class BenterQuantBrain:
    """
    Adaptador compatible de Probabilidad, Calibración (delegada a `ProbabilityCalibrator`
    sin techo duro arbitrario de 0.75), Valor Esperado (EV) y Dimensionamiento de Riesgo
    (Criterio de Kelly Fraccionado).
    """

    def __init__(
        self,
        feature_cols: List[str],
        tp_mult: float = settings.tp_multiple,
        sl_mult: float = settings.sl_multiple,
        roundtrip_fee: float = settings.roundtrip_fee,
        direction: str = "LONG",
        calibration_method: str = "isotonic",
        random_state: int = settings.random_seed,
    ):
        self.feature_cols = list(feature_cols)
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.roundtrip_fee = roundtrip_fee
        self.direction = direction.upper()
        self.calibration_method = calibration_method
        self.random_state = int(random_state)

        # Hiperparámetros regularizados y estrictamente deterministas
        self.base_clf = LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            num_leaves=12,
            learning_rate=0.03,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            n_jobs=1,
        )
        # Calibración delegada a ProbabilityCalibrator (sin techo arbitrario 0.75 ni cap [-4])
        self.calibrator = ProbabilityCalibrator(
            method=self.calibration_method,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit_and_calibrate(
        self,
        df_train: pd.DataFrame,
        y_train: pd.Series,
        df_calib: pd.DataFrame,
        y_calib: pd.Series,
        purge_bars: int = 24,
        enforce_purge: bool = False,
        bar_delta: pd.Timedelta = pd.Timedelta(minutes=15),
    ) -> None:
        """
        Paso 1: Valida separación temporal estricta entre `df_train` y `df_calib`.
                Si ambos índices son `DatetimeIndex`, exige `df_calib.index.min() > df_train.index.max()`.
                Si `enforce_purge=True`, exige además una brecha >= `purge_bars * bar_delta`.
        Paso 2: Entrena LightGBM determinista en `df_train`.
        Paso 3: Ajusta `ProbabilityCalibrator` exclusivamente sobre `(raw_calib_probs, y_calib)`.
        """
        if isinstance(df_train.index, pd.DatetimeIndex) and isinstance(
            df_calib.index, pd.DatetimeIndex
        ):
            if df_train.empty or df_calib.empty:
                raise ValueError("df_train y df_calib no pueden estar vacíos.")
            if df_calib.index.min() <= df_train.index.max():
                raise ValueError(
                    "Solapamiento temporal detectado entre df_train y df_calib: "
                    f"df_calib.index.min() ({df_calib.index.min()}) <= "
                    f"df_train.index.max() ({df_train.index.max()})."
                )
            if enforce_purge:
                verify_no_temporal_overlap(
                    df_train.index,
                    df_calib.index,
                    purge_bars=purge_bars,
                    embargo_bars=0,
                    bar_delta=bar_delta,
                )

        X_tr = df_train[self.feature_cols]
        X_ca = df_calib[self.feature_cols]

        # 1. Entrenamiento del clasificador base
        self.base_clf.fit(X_tr, y_train.astype(int))

        # 2. Probabilidades crudas en el conjunto de calibración temporalmente posterior
        raw_calib_probs = self.base_clf.predict_proba(X_ca)[:, 1]

        # 3. Ajuste del calibrador sin techo duro arbitrario 0.75 ni winsorización [-4]
        self.calibrator.fit(raw_calib_probs, y_calib.to_numpy(dtype=float))
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
        return self.calibrator.predict(raw_probs)

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

        p_clip = np.clip(p_win, PROB_EPS, 1.0 - PROB_EPS)
        auc_val = (
            float(roc_auc_score(y_test, p_win))
            if len(np.unique(y_vals)) >= 2
            else float("nan")
        )
        return {
            "AUC": round(auc_val, 4) if not np.isnan(auc_val) else float("nan"),
            "LogLoss": round(float(log_loss(y_test, p_clip, labels=[0, 1])), 4),
            "Brier": round(float(brier_score_loss(y_test, p_win)), 4),
            "Trades": n_trades,
            "WinRate_Taken": round(win_rate * 100, 2),
            "Avg_EV_Pred_R": round(avg_ev_pred, 3),
            "Avg_Realized_R": round(avg_realized_r, 3),
            "Total_Net_R": round(cum_r, 2)
        }
