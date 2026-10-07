# Binance Quant Lab

Un laboratorio de investigacion cuantitativa de grado institucional enfocado en la validacion estadistica, calibracion de probabilidades y rechazo estricto del curve-fitting (sobreajuste). 

Nota Filosofica: Este proyecto no es un "bot de trading" diseñado para mostrar backtests perfectos. Es un Research Gate (Muro de Contencion Cientifico) diseñado para destruir hipotesis debiles mediante validacion en ventanas rodantes (Walk-Forward) antes de arriesgar capital real.

---

## Arquitectura del Sistema

El laboratorio opera bajo un pipeline ETL estricto, ejecutado integramente en contenedores Docker (optimizado para arquitectura ARM64 / Orange Pi):

1. Ingestion Engine (src/data/ingestion.py): Descarga de datos granulares historicos (Klines, Funding Rates) desde Binance Vision mediante validacion criptografica (SHA-256).
2. Canonical Data Contract (src/data/canonical.py): Fusion de series temporales asincronas con resolucion de 15 minutos, almacenadas en formato .parquet comprimido.
3. Feature Engine Purificado (src/features/builder.py): Generacion de variables sin look-ahead bias, refinadas tras rigurosos estudios de ablacion (Ablation Studies). Mantiene exclusivamente:
   - Price & Technicals: Volatilidad (ATR, Bollinger Band Width), Momentum (RSI).
   - Context Engine: Regimen Macro (Pendiente de la EMA 200).
   - Derivados: Z-Score de Tasas de Financiacion (Funding Rates).
4. Labeling Engine: Generacion de etiquetas de clasificacion binaria (Target TP before SL), aislando explicitamente casos ambiguos (TP y SL en la misma vela) y bloqueos por tiempo (TIMEOUT).

---

## Modelado y Motor Cuantitativo

Tras rigurosas pruebas de ablacion y comparacion de Baselines, el laboratorio utiliza modelos lineales fuertemente regularizados para evitar la captura de ruido estocastico inherente a los micro-regimenes de criptomonedas.

- Baseline Oficial: Regresion Logistica Lineal (L2 Penalty).
- Calibracion: Platt Scaling (Sigmoid) calibrado en un set de validacion estrictamente aislado, evaluado mediante Brier Score y Log Loss.
- Data Splitting & Purging: Prevencion de fugas de datos (Data Leakage) utilizando purga matematica (pd.Timedelta) basada en el horizonte maximo de retencion de la operacion. 
- Walk-Forward Validation (src/evaluation/walk_forward.py): Evaluacion de robustez mediante 4 epocas rodantes (Ventanas expansivas) desglosadas por contexto (Regime Analysis). El sistema actualmente mantiene un Edge estadistico (Aprobado Global) exclusivamente en BTCUSDT.

---

## Trazabilidad y Experimentacion (Experiment Tracker)

Garantizamos reproducibilidad cientifica. Cada entrenamiento genera automaticamente un experiment_metadata.json que registra:
- Hash del commit de Git en el momento de la ejecucion.
- Periodos exactos de Train / Val / Test.
- Parametros del modelo y semilla aleatoria (Random Seed).
- Metricas fuera de muestra.

---

## Uso (Docker)

La infraestructura esta 100% aislada. Ejecuta la canalizacion en el siguiente orden:

1. Ingesta de datos:
   $ docker compose run --rm quant-env python src/data/ingestion.py

2. Construccion del contrato maestro (Merge):
   $ docker compose run --rm quant-env python src/data/canonical.py

3. Generacion de Features y Labels:
   $ docker compose run --rm quant-env python src/features/builder.py
   $ docker compose run --rm quant-env python src/labels/labeler.py

4. Test de Integridad (Anti-Leakage):
   $ docker compose run --rm quant-env python src/tests/test_leakage.py

5. Entrenamiento y Validacion Rodante (Walk-Forward):
   $ docker compose run --rm quant-env python src/evaluation/walk_forward.py

6. Interpretabilidad del Modelo:
   $ docker compose run --rm quant-env python src/evaluation/feature_importance.py

---
Desarrollado bajo principios de Machine Learning Financiero (Marcos Lopez de Prado).