from typing import Dict, List
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class BenterQuantBrain:
    """
    Motor de Probabilidad, Calibración Isotónica, Valor Esperado (EV)
    y Dimensionamiento de Riesgo (Criterio de Kelly Fraccionado).
    """
    def __init__(
        self, 
        feature_cols: List[str], 
        tp_mult: float = 2.0, 
        sl_mult: float = 1.0, 
        roundtrip_fee: float = 0.0010  # 0.05% entrada + 0.05% salida en Binance Futures
    ):
        self.feature_cols = feature_cols
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.roundtrip_fee = roundtrip_fee

        # Hiperparámetros regularizados para evitar sobreajuste
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
            n_jobs=4
        )
        self.calibrated_clf = None

    def fit_and_calibrate(
        self, 
        df_train: pd.DataFrame, 
        y_train: pd.Series, 
        df_calib: pd.DataFrame, 
        y_calib: pd.Series
    ) -> None:
        X_tr = df_train[self.feature_cols]
        X_ca = df_calib[self.feature_cols]

        self.base_clf.fit(X_tr, y_train)
        # Calibración Sigmoid (Platt) o Isotonic para que P(TP before SL) sea probabilidad real
        self.calibrated_clf = CalibratedClassifierCV(
            estimator=self.base_clf, 
            method="isotonic", 
            cv="prefit"
        )
        self.calibrated_clf.fit(X_ca, y_calib)

    def predict_calibrated_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_cols]
        return self.calibrated_clf.predict_proba(X)[:, 1]

    def compute_ev_and_kelly(
        self, 
        p_win: np.ndarray, 
        atr_pct: np.ndarray, 
        kelly_fraction: float = 0.20
    ) -> pd.DataFrame:
        """
        Aplica la fórmula de Bill Benter adaptada a futuros con comisiones:
        EV = p * b_net - (1 - p) * l_net
        """
        # Distancia porcentual de 1 unidad de riesgo (1R)
        r_dist_pct = np.maximum(atr_pct * self.sl_mult, 1e-4)
        
        # Impacto de las comisiones expresado en unidades de R
        fee_in_r = self.roundtrip_fee / r_dist_pct

        # Cuota neta de ganancia (b) y pérdida neta (l) en unidades R
        b_net = (self.tp_mult / self.sl_mult) - fee_in_r
        l_net = 1.0 + fee_in_r

        p_loss = 1.0 - p_win
        ev_r = (p_win * b_net) - (p_loss * l_net)

        # Fracción de Kelly: f* = (p*b_net - q*l_net) / (b_net * l_net)
        raw_kelly = np.where(b_net > 0, ev_r / (b_net * l_net), 0.0)
        safe_kelly = np.clip(raw_kelly * kelly_fraction, 0.0, 0.03)  # Tope máximo de 3% de riesgo por trade

        return pd.DataFrame({
            "p_tp_before_sl": p_win,
            "b_net_odds": b_net,
            "ev_in_r": ev_r,
            "risk_fraction_kelly": safe_kelly
        })

    def evaluate_model(
        self, 
        df_test: pd.DataFrame, 
        y_test: pd.Series, 
        ev_min_threshold: float = 0.15
    ) -> Dict[str, float]:
        p_win = self.predict_calibrated_proba(df_test)
        atr_pct = df_test["atr_pct"].values
        ev_df = self.compute_ev_and_kelly(p_win, atr_pct)

        # Solo ejecutamos apuestas donde EV > umbral (Filosofía Bill Benter)
        signals = ev_df["ev_in_r"].values > ev_min_threshold
        n_trades = int(np.sum(signals))

        if n_trades > 0:
            realized_r = np.where(
                y_test.values[signals] == 1,
                ev_df["b_net_odds"].values[signals],
                -(1.0 + (self.roundtrip_fee / np.maximum(atr_pct[signals] * self.sl_mult, 1e-4)))
            )
            avg_realized_r = float(np.mean(realized_r))
            cum_r = float(np.sum(realized_r))
            win_rate = float(np.mean(y_test.values[signals]))
        else:
            avg_realized_r, cum_r, win_rate = 0.0, 0.0, 0.0

        return {
            "AUC": round(float(roc_auc_score(y_test, p_win)), 4),
            "LogLoss": round(float(log_loss(y_test, p_win)), 4),
            "Brier": round(float(brier_score_loss(y_test, p_win)), 4),
            "Trades": n_trades,
            "WinRate_Taken": round(win_rate * 100, 2),
            "Avg_EV_Pred_R": round(float(ev_df["ev_in_r"].values[signals].mean()), 3) if n_trades > 0 else 0.0,
            "Avg_Realized_R": round(avg_realized_r, 3),
            "Total_Net_R": round(cum_r, 2)
        }