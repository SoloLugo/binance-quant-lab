# 🔬 Binance Quant Lab

Un laboratorio de investigación cuantitativa de grado institucional enfocado en la validación estadística, calibración de probabilidades y rechazo estricto del *curve-fitting* (sobreajuste). 

**Nota Filosófica:** Este proyecto no es un "bot de trading" diseñado para mostrar backtests perfectos. Es un *Research Gate* (Muro de Contención Científico) diseñado para destruir hipótesis débiles mediante validación en ventanas rodantes (Walk-Forward) antes de arriesgar un solo centavo.

---

## 🏗️ Arquitectura del Sistema

El laboratorio opera bajo un pipeline ETL estricto, ejecutado íntegramente en contenedores Docker (optimizado para arquitectura ARM64 / Orange Pi):

1. **Ingestion Engine (`src/data/ingestion.py`):** Descarga de datos granulares (Klines, Open Interest, Funding Rates) desde Binance Vision mediante validación criptográfica (SHA-256).
2. **Canonical Data Contract (`src/data/canonical.py`):** Fusión de series temporales asíncronas con resolución de 15 minutos, almacenadas en formato `.parquet` comprimido.
3. **Feature Engine (`src/features/builder.py`):** Generación de variables sin *look-ahead bias*, divididas por familias:
   - *Familia A:* Price & Technicals (Volatilidad ATR, Momentum RSI).
   - *Familia B:* Smart Money Concepts (Swing Highs/Lows, Liquidity Sweeps, Order Blocks paramétricos).
   - *Familia C:* Derivados (Z-Score de Tasas de Financiación, Flujo de Open Interest).
4. **Labeling Engine:** Generación de etiquetas de clasificación binaria (Target TP before SL), aislando explícitamente casos ambiguos (TP y SL en la misma vela) y bloqueos por tiempo (TIMEOUT).

---

## 🧠 Modelado y Motor Cuantitativo

Tras rigurosas pruebas de ablación y comparación de *Baselines*, el laboratorio utiliza modelos lineales fuertemente regularizados para evitar la captura de ruido estocástico inherente a los micro-regímenes de criptomonedas.

- **Baseline Oficial:** Regresión Logística Lineal (L2 Penalty).
- **Calibración:** Platt Scaling (Sigmoid) calibrado en un set de validación estrictamente aislado, evaluado mediante *Brier Score*, *Log Loss* y *Reliability Curves*.
- **Data Splitting & Purging:** Prevención de fugas de datos (Data Leakage) utilizando purga matemática (`pd.Timedelta`) basada en el horizonte máximo de retención de la operación. 
- **Walk-Forward Validation (`src/evaluation/walk_forward.py`):** Evaluación de robustez mediante 4 épocas rodantes (Ventanas expansivas) para confirmar adaptabilidad a distintos regímenes macroeconómicos.

---

## 🛡️ Trazabilidad y Experimentación (Experiment Tracker)

Garantizamos reproducibilidad científica. Cada entrenamiento genera automáticamente un `experiment_metadata.json` que registra:
- Hash del commit de Git en el momento de la ejecución.
- Periodos exactos de Train / Val / Test.
- Parámetros del modelo y semila aleatoria (*Random Seed*).
- Métricas fuera de muestra (ROC-AUC, Brier Score).

---

## 🚀 Uso Rápido (Docker)

La infraestructura está 100% aislada. Ejecuta la canalización en el siguiente orden:

1. **Ingesta de datos:**
   ```bash
   docker compose run --rm quant-env python src/data/ingestion.py