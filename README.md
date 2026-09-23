# Laboratorio Cuantitativo de Probabilidad para Binance Futures

## Descripcion General

Este proyecto es un laboratorio de investigación cuantitativa diseñado para ejecutarse de forma completamente reproducible en contenedores de Docker (optimizado tanto para arquitecturas x86 como ARM64, como placas Orange Pi). Su objetivo principal no es adivinar la dirección del mercado, sino calcular una probabilidad condicional estricta: predecir la probabilidad de que el precio alcance un objetivo de Take Profit antes de tocar un límite de Stop Loss ($P(TP \text{ antes de } SL)$) utilizando datos históricos de futuros de Binance en temporalidades de 15 minutos.

## Conceptos Teóricos y Financieros

El diseño de este sistema se fundamenta en principios matemáticos y de microestructura de mercado:

- **Probabilidad Condicional y Clasificación Binaria:** El modelo estima la probabilidad de éxito de una operación en función del estado actual de las variables del mercado, transformando un problema financiero en un modelo matemático supervisado.
- **Gestión de Riesgo y R-Multiples:** Las operaciones se estructuran bajo un modelo de riesgo fijo (1R de pérdida por 2R de beneficio). Esto permite estudiar el Valor Esperado (Expectancy) para determinar el umbral de probabilidad necesario que supere el punto de equilibrio (break-even).
- **Microestructura de Mercado (Futuros USD-M):** Análisis de contratos perpetuos y derivados en Binance, permitiendo capturar dinámicas tanto alcistas como bajistas.
- **Integridad Temporal (Prevención de Fuga de Datos):** Para evitar el sesgo de supervivencia y el look-ahead bias, se implementan divisiones temporales estrictas (Time-based Train/Test Split) acompañadas de una estrategia de purga (Purge) que aísla el horizonte de las etiquetas.

## Arquitectura del Pipeline y Componentes

El laboratorio está estructurado en módulos secuenciales independientes:

1. **Ingeccion (data/raw/):** El módulo `downloader.py` se conecta a la API de Binance Vision para descargar archivos mensuales de klines de 15 minutos y valida su integridad mediante criptografía y archivos `.CHECKSUM`, generando un manifiesto centralizado (`manifest.json`).
2. **Canonicalizacion (data/canonical/):** El script `canonical.py` unifica la serie temporal, limpia inconsistencias de formato, valida invariantes matemáticas del precio (como prohibir que el mínimo sea mayor al máximo) y comprime la información en formato columnar Parquet utilizando tipos de datos optimizados para memoria (`float32`).
3. **Etiquetado de Trayectoria (data/labels/):** El motor `labeler.py` simula la evolución futura del precio vela por vela dentro de un horizonte determinado para clasificar cada punto en resultados de éxito (1), fracaso (0), tiempo agotado (-1) o casos ambiguos (-2) que son descartados por rigor estadístico.
4. **Ingenieria de Características (data/features/):** El script `builder.py` calcula variables cuantitativas puramente estacionarias (retornos logarítmicos multi-escala, distancias relativas a medias móviles exponenciales, RSI, ATR relativo y microestructura de velas).
5. **Entrenamiento y Modelado (outputs/models/):** El módulo `trainer.py` entrena un modelo base de Regresión Logística con regularización L2 y balanceo de clases, serializando los resultados y el escalador mediante `joblib`.
6. **Calibrador y Análisis de Umbrales:** El script `calibrator.py` evalúa el rendimiento fuera de muestra simulando la rentabilidad teórica bajo diferentes niveles de exigencia de probabilidad (Thresholds).

## Guia de Instalacion y Ejecucion (Desde Cero)

Si clonas este repositorio en una nueva máquina (PC con Windows/Linux o placa de desarrollo), asegúrate de tener instalado Docker y Docker Compose. Luego, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU-USUARIO/binance-quant-lab.git
cd binance-quant-lab
```
### 2. Construir el entorno de contenedores
```bash
docker compose build

docker compose run --rm quant-env python src/ingestion/downloader.py

docker compose run --rm quant-env python src/processing/canonical.py

docker compose run --rm quant-env python src/labeling/labeler.py

docker compose run --rm quant-env python src/features/builder.py

docker compose run --rm quant-env python src/models/trainer.py

docker compose run --rm quant-env python src/evaluation/calibrator.py
```
