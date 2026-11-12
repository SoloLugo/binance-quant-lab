# Binance Quant Lab: Motor de Predicción de Futuros y Valor Esperado (EV)

Laboratorio de investigación cuantitativa y bot de ejecución automatizada para **Binance USD-M Futures**, diseñado para operar localmente en arquitecturas ARM64 (Orange Pi 4 Pro con NVMe bajo Armbian) o servidores x86_64 mediante contenedores Docker.

---

## 1. Fundamento Matemático: La Metodología de Bill Benter

El núcleo de toma de decisiones del bot está fundamentado en el modelo probabilístico de **Bill Benter** (*"Computer Based Horse Race Handicapping and Wagering Systems"*, 1994) y en las bases de teoría de la probabilidad aplicada del MIT (Prof. John Tsitsiklis):

1. **Separación entre Estimación de Probabilidad y Cuotas de Mercado:**
   El objetivo del modelo de Machine Learning no es "adivinar hacia dónde va el precio", sino estimar la **probabilidad empírica real ($p$)** de que el precio alcance un nivel de *Take Profit (TP)* antes de tocar un *Stop Loss (SL)* dentro de un horizonte temporal fijo (*Triple-Barrier Labeling*).
2. **Definición de Cuotas Netas con Fricción Real ($b_{\text{net}}$ y $l_{\text{net}}$):**
   En un libro de órdenes de futuros perpetuos, las cuotas las definen nuestras barreras de volatilidad ($\text{ATR}$) descontando las comisiones de ejecución (*Taker Fees* de apertura y cierre, $c \approx 0.10\%$):
   $$b_{\text{net}} = \frac{\text{TP}_{\text{mult}}}{\text{SL}_{\text{mult}}} - \frac{c}{d_{\text{SL}}}, \quad l_{\text{net}} = 1 + \frac{c}{d_{\text{SL}}}$$
   Donde $d_{\text{SL}}$ es la distancia porcentual desde el precio de entrada hasta el Stop Loss. Cuando la volatilidad se comprime ($d_{\text{SL}}$ pequeño), el peso relativo de las comisiones aumenta y reduce la cuota neta real.
3. **Regla de Decisión por Valor Esperado ($EV$):**
   El sistema evalúa cada cierre de vela de 15 minutos bajo la desigualdad fundamental de valor esperado en unidades de riesgo ($R$):
   $$EV_R = p \cdot b_{\text{net}} - (1 - p) \cdot l_{\text{net}}$$
   * Si $EV_R > \text{Umbral Mínimo}$ (ej. $+0.12R$) $\rightarrow$ **Ejecuta la operación**.
   * Si $EV_R \le \text{Umbral Mínimo}$ $\rightarrow$ **Pasa (No opera)**.
4. **Dimensionamiento de Posición (Fractional Kelly Criterion):**
   El tamaño del riesgo por operación se ajusta dinámicamente en función de la ventaja matemática estimada, aplicando una fracción conservadora ($\gamma = 0.20$) con un tope máximo estricto de riesgo de cuenta:
   $$f^* = \gamma \cdot \left( \frac{EV_R}{b_{\text{net}} \cdot l_{\text{net}}} \right)$$

---

## 2. Arquitectura del Sistema: El Context Engine

En lugar de alimentar un clasificador con decenas de indicadores técnicos aislados, la arquitectura separa la extracción de señales en cuatro capas jerárquicas que convergen en un **Context Engine** antes de pasar al motor de probabilidad:

```text
                         MARKET DATA (Binance Vision / Mainnet API)
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
   TECHNICAL ANALYSIS            SMART MONEY            CONTEXT / DERIVATIVES
   (Acción del Precio)       (Microestructura/Flujo)     (Macro & Estructura)
             │                         │                         │
   ├─ ATR (14) Volatilidad   ├─ Order Flow Imbalance   ├─ BTC HTF Regime (1D/5D)
   ├─ Retornos Normalizados  │  (Taker Buy vs Sell)    ├─ Volatility Percentile
   ├─ Distancia EMAs (20/50) ├─ Whale Trade Size Z     ├─ Flujo BTC → ETH → ALTs
   ├─ RSI (14)               └─ Absorción en Mechas    ├─ Correlación BTC-ETH
   └─ Compresión Bollinger                             ├─ Funding Rate Z-Score
             │                         │               └─ Riesgo Macro (FOMC 24h)
             └─────────────────────────┼─────────────────────────┘
                                       ▼
                             ┌───────────────────┐
                             │  CONTEXT ENGINE   │
                             │  & FEATURE MATRIX │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │ PROBABILITY ENGINE│
                             │    (LightGBM)     │
                             │  P(TP before SL)  │
                             └─────────┬─────────┘
                                       ▼
                          ISOTONIC CALIBRATION (Out-of-Sample)
                                       ▼
                          EXPECTED VALUE ENGINE (EV en R)
                                       ▼
                          RISK ENGINE (Fractional Kelly)
                                       ▼
                          EXECUTION (Testnet / Mainnet)
```

### Validación Científica Incremental (Ablation Study)
El pipeline de entrenamiento (`src/train_ablation.py`) permite demostrar matemáticamente si cada capa adicional aporta valor predictivo real fuera de muestra (*Out-of-Sample*), comparando:
* **Modelo 1 (`M1_Technical`):** Únicamente indicadores técnicos (`ta_*`).
* **Modelo 2 (`M2_TA+SmartMoney`):** Técnicos + desbalance de flujo de órdenes agresivas y actividad de ballenas (`sm_*`).
* **Modelo 3 (`M3_TA+SM+Derivatives`):** Añade el régimen y Z-Score del *Funding Rate* (`deriv_*`).
* **Modelo 4 (`M4_Full_ContextEngine`):** Modelo completo integrando el régimen macro de BTC, fuerza relativa `ETH/BTC`, percentil de volatilidad y proximidad a eventos FOMC (`ctx_*`).

Cada modelo se evalúa mediante **ROC AUC**, **Log Loss**, **Brier Score**, tasa de acierto y **Retorno Neto Acumulado en unidades $R$**.

---

## 3. Ejecución Híbrida (Mainnet Data + Testnet Execution)

El libro de órdenes de *Binance Futures Testnet* carece de liquidez orgánica y produce distorsiones artificiales en los indicadores de volumen y *funding*. Para garantizar que el modelo opere en pruebas exactamente igual que en producción:

1. **Lectura de Datos (`market_data_client`):** Se conecta en tiempo real a los endpoints públicos de **Binance Futures Mainnet** (`fapi.binance.com`) para construir las velas de 15m, el flujo *Taker* y el *Funding Rate* real del mercado.
2. **Ejecución de Órdenes (`execution_client`):** Envía las órdenes a mercado y las barreras nativas (`STOP_MARKET` y `TAKE_PROFIT_MARKET`) hacia **Binance Futures Testnet** (`testnet.binancefuture.com`) o hacia **Mainnet** según la variable `USE_TESTNET` del archivo `.env`.

---

## 4. Estructura del Repositorio

```text
binance-quant-lab/
├── data/
│   └── raw/                  # CSVs mensuales descargados de data.binance.vision (Ignorado en Git)
├── models/                   # Artefacto entrenado y calibrado quant_brain.joblib (Ignorado en Git)
├── src/
│   ├── __init__.py           # Definición del paquete
│   ├── config.py             # Configuración centralizada e inmutable
│   ├── download_data.py      # Descargador paralelo desde Binance Vision (BTC, ETH, Funding)
│   ├── data_loader.py        # Lector y consolidador de series históricas
│   ├── pipeline.py           # Context Engine, Feature Engine y Triple-Barrier Labeling
│   ├── brain.py              # Clasificador LightGBM, Calibración Isotónica, EV y Kelly
│   ├── train_ablation.py     # Estudio de ablación (M1 a M4) y exportación del modelo
│   └── live_testnet_bot.py   # Demonio de inferencia y ejecución 24/7
├── .env                      # Variables de entorno y credenciales API (Ignorado en Git)
├── .gitignore
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 5. Guía de Comandos y Uso

### Paso 1: Configurar el archivo `.env`
Crear el archivo `.env` en la raíz del proyecto con las credenciales de [Binance Futures Testnet](https://testnet.binancefuture.com/):

```env
BINANCE_TESTNET_API_KEY=tu_api_key_aqui
BINANCE_TESTNET_SECRET=tu_secret_key_aqui
USE_TESTNET=true

START_YEAR_MONTH=2024-01
END_YEAR_MONTH=2026-10
RAW_DATA_DIR=data/raw
MODEL_PATH=models/quant_brain.joblib

TP_MULTIPLE=2.0
SL_MULTIPLE=1.0
HORIZON_BARS=24
ROUNDTRIP_FEE=0.0010
EV_MIN_THRESHOLD=0.12
KELLY_FRACTION=0.20
MAX_RISK_PER_TRADE=0.03
LGBM_N_JOBS=4
```

### Paso 2: Construir la imagen de Docker
Compila el entorno ligero con soporte para múltiples hilos (`libgomp1` / OpenMP):

```bash
docker compose build
```

### Paso 3: Descargar datos históricos y entrenar los modelos (Ablation Study)
Ejecuta en un solo paso la descarga paralela desde *Binance Vision* y el entrenamiento/calibración de los 4 modelos:

```bash
docker compose --profile training run --rm trainer
```

Comandos individuales si deseas ejecutar las etapas por separado:

```bash
# Únicamente descargar o actualizar archivos CSV faltantes desde Binance Vision
docker compose --profile training run --rm trainer python src/download_data.py

# Únicamente ejecutar el estudio de ablación (M1-M4) y exportar models/quant_brain.joblib
docker compose --profile training run --rm trainer python src/train_ablation.py
```

### Paso 4: Desplegar el Bot 24/7 en Vivo
Levanta el servicio persistente en segundo plano con política de reinicio automático (`unless-stopped`):

```bash
docker compose up -d quant-bot
```

### Paso 5: Monitoreo y Operación Diaria

```bash
# Ver en tiempo real las evaluaciones de cierre de vela (Contexto, Probabilidad, EV y Órdenes)
docker logs -f bql_live_bot

# Consultar el consumo de CPU y memoria RAM del contenedor en la placa
docker stats bql_live_bot

# Reiniciar el bot tras modificar parámetros en el archivo .env o reentrenar el modelo
docker compose restart quant-bot

# Detener la ejecución del bot
docker compose stop quant-bot
```
