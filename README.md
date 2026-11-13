# Binance Quant Lab: Motor de Predicción de Futuros y Valor Esperado (EV)

Laboratorio de investigación cuantitativa, **simulador de microestructura en tiempo real (Paper Trading)** y bot de ejecución automatizada para **Binance USD-M Futures (BTCUSDT)**, diseñado para operar localmente en arquitecturas ARM64 (Orange Pi 4 Pro con NVMe bajo Armbian) o servidores x86_64 mediante contenedores Docker.

---

## 1. Fundamento Matemático: La Metodología de Bill Benter

El núcleo de toma de decisiones del bot está fundamentado en el modelo probabilístico de **Bill Benter** (*"Computer Based Horse Race Handicapping and Wagering Systems"*, 1994) y en las bases de teoría de la probabilidad aplicada del MIT (Prof. John Tsitsiklis):

1. **Separación entre Estimación de Probabilidad y Cuotas de Mercado:**
   El objetivo del modelo de Machine Learning no es "adivinar hacia dónde va el precio", sino estimar la **probabilidad empírica real ($p$)** de que el precio alcance un nivel de *Take Profit (TP)* antes de tocar un *Stop Loss (SL)* dentro de un horizonte temporal fijo (*Triple-Barrier Labeling*, tanto en dirección **LONG** como **SHORT**).
2. **Definición de Cuotas Netas con Fricción Real ($b_{\text{net}}$ y $l_{\text{net}}$):**
   En un libro de órdenes de futuros perpetuos, las cuotas las definen nuestras barreras de volatilidad ($\text{ATR}$) descontando las comisiones de ejecución (*Taker Fees* de apertura y cierre, $c \approx 0.10\%$):
   $$b_{\text{net}} = \frac{\text{TP}_{\text{mult}}}{\text{SL}_{\text{mult}}} - \frac{c}{d_{\text{SL}}}, \quad l_{\text{net}} = 1 + \frac{c}{d_{\text{SL}}}$$
   Donde $d_{\text{SL}}$ es la distancia porcentual desde el precio de entrada hasta el Stop Loss. Cuando la volatilidad se comprime ($d_{\text{SL}}$ pequeño), el peso relativo de las comisiones aumenta y reduce la cuota neta real.
3. **Regla de Decisión por Valor Esperado ($EV$):**
   El sistema evalúa cada cierre de vela de 15 minutos en ambas direcciones (`LONG` y `SHORT`) bajo la desigualdad fundamental de valor esperado en unidades de riesgo ($R$):
   $$EV_R = p \cdot b_{\text{net}} - (1 - p) \cdot l_{\text{net}}$$
   * Si $\max(EV_{\text{LONG}}, EV_{\text{SHORT}}) > \text{Umbral Mínimo}$ (ej. $+0.12R$) $\rightarrow$ **Ejecuta la operación**.
   * Si $\max(EV_{\text{LONG}}, EV_{\text{SHORT}}) \le \text{Umbral Mínimo}$ $\rightarrow$ **Pasa (No opera)**.
4. **Dimensionamiento de Posición (Fractional Kelly Criterion + Topes de Riesgo):**
   El tamaño del riesgo por operación se ajusta dinámicamente en función de la ventaja matemática estimada, aplicando una fracción conservadora ($\gamma = 0.20$) con topes estrictos de riesgo de cuenta (`MAX_RISK_PER_TRADE = 3%`) y de apalancamiento efectivo (`MAX_LEVERAGE = 5x`):
   $$f^* = \gamma \cdot \left( \frac{EV_R}{b_{\text{net}} \cdot l_{\text{net}}} \right)$$

---

## 2. Arquitectura del Sistema: El Context Engine

En lugar de alimentar un clasificador con decenas de indicadores técnicos aislados, la arquitectura separa la extracción de señales en cuatro capas jerárquicas estacionarias que convergen en un **Context Engine** antes de pasar al motor de probabilidad:

```text
                         MARKET DATA (Binance Mainnet / Vision Failover)
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
   TECHNICAL ANALYSIS            SMART MONEY            CONTEXT / DERIVATIVES
   (Acción del Precio)       (Microestructura/Flujo)     (Macro & Estructura)
             │                         │                         │
   ├─ ATR (14) Volatilidad   ├─ Order Flow Imbalance   ├─ BTC HTF Regime (1D/5D)
   ├─ Retornos Normalizados  │  (Taker Buy vs Sell)    ├─ Volatility Percentile
   ├─ Distancia EMAs (20/50) ├─ Whale Trade Size Z     ├─ Impulso Relativo ETH/BTC
   ├─ RSI (14)               └─ Absorción en Mechas    ├─ Funding Rate & Z-Score
   └─ Compresión Bollinger                             └─ Riesgo Macro (FOMC 24h)
             │                         │                         │
             └─────────────────────────┼─────────────────────────┘
                                       ▼
                             ┌───────────────────┐
                             │  CONTEXT ENGINE   │
                             │  & FEATURE MATRIX │
                             └─────────┬─────────┘
                                       ▼
                             ┌───────────────────┐
                             │ PROBABILITY ENGINE│
                             │ (LightGBM L & S)  │
                             │  P(TP before SL)  │
                             └─────────┬─────────┘
                                       ▼
                     ISOTONIC CALIBRATION (Purged Out-of-Sample)
                                       ▼
                          EXPECTED VALUE ENGINE (EV en R)
                                       ▼
                     RISK ENGINE (Fractional Kelly + Circuit Breaker)
                                       ▼
                  EXECUTION (Paper Real-Time / Testnet / Mainnet)
```

### Validación Científica Incremental (Ablation Study)
El pipeline de entrenamiento (`src/train_ablation.py`) aplica división temporal con ventana de purga (`purge_bars = 24`) sin *Look-Ahead Bias* y evalúa trades secuenciales sin solapamiento (*1 posición activa a la vez*), comparando:
* **Modelo 1 (`M1_Technical`):** Únicamente indicadores técnicos (`ta_*`).
* **Modelo 2 (`M2_TA+SmartMoney`):** Técnicos + desbalance de flujo de órdenes agresivas y actividad de ballenas (`sm_*`).
* **Modelo 3 (`M3_TA+SM+Derivatives`):** Añade el régimen y Z-Score del *Funding Rate* (`deriv_*`).
* **Modelo 4 (`M4_Full_ContextEngine`):** Modelo completo integrando el régimen macro de BTC, impulso relativo `ETH/BTC`, percentil de volatilidad y proximidad a eventos FOMC (`ctx_*`).

---

## 3. Tres Modos de Operación (`EXECUTION_MODE`)

1. **`paper` (Simulación en Tiempo Real — `src/simulator.py`):**
   * Opera en tiempo real con cotizaciones reales de Binance sin requerir llaves API.
   * Simula el cruce del spread **Bid/Ask real**, **slippage** configurable (`SLIPPAGE_BPS`), **comisiones Taker** (`0.05%` por lado) y **liquidaciones de Funding Rate cada 8 horas** (`00:00`, `08:00`, `16:00 UTC`).
   * Monitorea continuamente las 3 barreras (**Stop Loss sobre Mark Price**, **Take Profit** y **Barrera Vertical Time-Stop de 6 horas / 24 velas**).
   * Persiste el estado de la cuenta en `data/sim_state.json` y el historial auditable de operaciones en `data/trades_history.csv`.
2. **`testnet` (Ejecución Híbrida Mainnet Data + Testnet Execution):**
   * Lee velas y flujo real con failover automático (`fapi.binance.com` $\rightarrow$ `data-api.binance.vision`) y ejecuta órdenes en **Binance Futures Testnet** (`testnet.binancefuture.com`).
   * Ancla las barreras `STOP_MARKET` y `TAKE_PROFIT_MARKET` (`workingType="MARK_PRICE"`, `closePosition=True`) al **precio real de ejecución del exchange**, eliminando errores `-2021` por divergencia de precios entre Mainnet y Testnet.
3. **`mainnet` (Producción con Dinero Real):**
   * Incluye protecciones institucionales: **Rollback automático de emergencia** (si falla la colocación del `STOP_MARKET` tras abrir posición, cierra inmediatamente a mercado con `reduceOnly=True`), limpieza de órdenes condicionales huérfanas, cierre automático por **Time-Stop (24 velas)** y **Circuit Breaker de Drawdown Diario (`MAX_DAILY_DRAWDOWN_PCT = 6%`)**.

---

## 4. Estructura del Repositorio

```text
binance-quant-lab/
├── data/
│   ├── raw/                  # CSVs mensuales descargados de data.binance.vision (Ignorado en Git)
│   ├── sim_state.json        # Estado persistente del simulador en tiempo real
│   └── trades_history.csv    # Bitácora auditable de trades cerrados en simulación
├── models/                   # Artefacto entrenado y calibrado quant_brain.joblib (Ignorado en Git)
├── src/
│   ├── __init__.py           # Definición del paquete
│   ├── config.py             # Configuración centralizada e inmutable
│   ├── download_data.py      # Descargador paralelo desde Binance Vision (BTC, ETH, Funding)
│   ├── data_loader.py        # Lector y consolidador de series históricas (ms/us normalizado)
│   ├── pipeline.py           # Context Engine, Feature Engine y Triple-Barrier Outcomes
│   ├── brain.py              # Clasificador LightGBM, Calibración Isotónica, EV y Kelly
│   ├── simulator.py          # Motor de Simulación en Tiempo Real (Bid/Ask, Slippage, Funding, 3 Barreras)
│   ├── train_ablation.py     # Estudio de ablación (M1 a M4) y exportación LONG + SHORT
│   └── live_testnet_bot.py   # Demonio de inferencia, simulación y ejecución 24/7
├── tests/
│   └── test_quant_lab.py     # Suite de pruebas unitarias y de integración (pytest)
├── .env                      # Variables de entorno y credenciales API (Ignorado en Git)
├── .gitignore
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 5. Guía de Comandos y Uso

### Paso 1: Configurar el archivo `.env`
Crear el archivo `.env` en la raíz del proyecto:

```env
EXECUTION_MODE=paper
USE_TESTNET=true
BINANCE_TESTNET_API_KEY=tu_api_key_aqui
BINANCE_TESTNET_SECRET=tu_secret_key_aqui

START_YEAR_MONTH=2024-01
END_YEAR_MONTH=2026-10
RAW_DATA_DIR=data/raw
MODEL_PATH=models/quant_brain.joblib

TP_MULTIPLE=2.0
SL_MULTIPLE=1.0
HORIZON_BARS=24
ROUNDTRIP_FEE=0.0010
SLIPPAGE_BPS=1.5
EV_MIN_THRESHOLD=0.12
KELLY_FRACTION=0.20
MAX_RISK_PER_TRADE=0.03
MAX_LEVERAGE=5
MAX_DAILY_DRAWDOWN_PCT=0.06
ENABLE_SHORTS=true
INITIAL_PAPER_BALANCE=10000.0
POLL_INTERVAL_SECONDS=15
LGBM_N_JOBS=4
```

### Paso 2: Descargar datos históricos y entrenar los modelos (Ablation Study)

```bash
# Descargar archivos históricos desde Binance Vision
PYTHONPATH=src python src/download_data.py

# Ejecutar el estudio de ablación (M1-M4) y exportar models/quant_brain.joblib
PYTHONPATH=src python src/train_ablation.py
```

### Paso 3: Simulación en Tiempo Real y Pruebas

```bash
# Evaluar en vivo la última vela cerrada de 15m en modo Simulación (Paper)
PYTHONPATH=src python src/live_testnet_bot.py --mode paper --once

# Ejecutar simulación Replay acelerada sobre las últimas 500 velas reales en vivo
PYTHONPATH=src python src/live_testnet_bot.py --mode paper --replay-bars 500

# Iniciar el demonio 24/7 en tiempo real (sincronizado a cierres de 15m UTC con monitoreo cada 15s)
PYTHONPATH=src python src/live_testnet_bot.py --mode paper

# Ejecutar la suite completa de tests automatizados
PYTHONPATH=src pytest -v
```
