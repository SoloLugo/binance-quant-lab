# Binance Quant Lab — Motor de Investigación Cuantitativa para USD-M Futures

Laboratorio de investigación cuantitativa, simulación realista (Backtesting + Paper Trading) y ejecución segura en **Testnet** para **Binance USD-M Futures (`BTCUSDT` / `ETHUSDT`)**, diseñado bajo rigor estadístico estricto: cero *look-ahead bias*, separación de Test Final Sellado, calibración de probabilidades, cálculo de Valor Esperado (EV) neto después de costes y gestión de riesgo independiente del modelo.

> **Protección de Capital (Plan Maestro §1.4):** El modo **`mainnet`** y cualquier operación con capital real están **bloqueados por diseño** (`PermissionError` en configuración, CLI y motor de ejecución). El modo predeterminado es **`paper`** con **`DRY_RUN=true`**.

---

## 1. Objetivo Cuantitativo

El sistema no busca "adivinar dirección", sino responder de forma reproducible y verificable:

1. ¿Cuál es la probabilidad $P(\text{TP antes de SL dentro de } H \text{ velas})$ bajo protocolo ejecutable `next_open`?
2. ¿Qué tan calibrada y confiable es esa probabilidad (Brier, BSS, LogLoss, ECE, MCE, reliability table)?
3. ¿Existe ventaja estadística neta ($\text{EV}_{\text{net}} > 0$) después de comisiones Taker, spread, slippage, funding 8h y distribución económica real de salidas por `TIMEOUT`?
4. ¿En qué regímenes de mercado funciona mejor o peor?
5. ¿Cuánto riesgo asumir mediante un `RiskEngine` independiente capaz de rechazar cualquier señal?

---

## 2. Arquitectura Modular Implementada

```text
MARKET DATA (download_data.py + data_loader.py: SHA256, manifiestos, versionado)
       │
       ▼
DATA QUALITY GATE (data_quality.py: UTC, orden, duplicados, gaps, OHLC, alineación)
       │
       ├──► LABEL ENGINE (label_engine.py: next_open, WIN/LOSS/TIMEOUT/AMBIGUOUS)
       │
       ├──► FEATURE ENGINE (feature_engine.py: TA, proxies microestructura, Market Structure ms_*, Derivados — 0 .bfill())
       │
       ├──► CONTEXT ENGINE (context_engine.py: 6 regímenes causales + control temporal available_at)
       │
       ▼
PURGED TEMPORAL SPLITS (splits.py: Purged Walk-Forward + Test Final Sellado bloqueado)
       │
       ▼
PROBABILITY ENGINE (probability_engine.py: 5 niveles de modelos deterministas LONG/SHORT)
       │
       ▼
CALIBRATION ENGINE (calibration_engine.py: None / Platt / Isotonic sin techo arbitrario, ECE/MCE)
       │
       ▼
EV ENGINE & COST MODEL (ev_engine.py: fees, spread, slippage, funding, TIMEOUT económico real)
       │
       ├──► BACKTEST ENGINE (backtest_engine.py: estado aislado, gaps, funding 8h, escenarios adversos)
       ├──► VALIDATION / ABLATION / TRACKING (validation.py, ablation.py, experiment_tracker.py)
       │
       ▼
RISK ENGINE (risk_engine.py: independiente del modelo, Kelly incierto, límites, kill switch)
       │
       ▼
PAPER TRADING & MONITORING (simulator.py + monitoring.py: estado aislado + journal JSONL)
       │
       ▼
EXECUTION ENGINE TESTNET (execution_engine.py + live_testnet_bot.py: DRY_RUN=true, Mainnet bloqueado, clientOrderId propio, entry_time persistido)
```

---

## 3. Estructura del Repositorio

```text
binance-quant-lab/
├── AUDIT_REPORT.md              # Auditoría técnica inicial (Fase 0)
├── DATA_DICTIONARY.md           # Diccionario de datos, calidad y versionado (Fase 1)
├── src/
│   ├── config.py                # Configuración inmutable (from_env, bloqueo duro Mainnet, DRY_RUN=true)
│   ├── contracts.py             # Contratos compartidos entre capas (DataManifest, CostModel, RiskDecision, etc.)
│   ├── download_data.py         # Descarga reproducible con SHA256, .CHECKSUM y manifiestos JSON
│   ├── data_loader.py           # Carga normalizada (ms/us -> UTC) y manifiestos versionados inmutables
│   ├── data_quality.py          # Data Quality Gate bloqueante + validación multi-fuente
│   ├── label_engine.py          # Triple-Barrier ejecutable (next_open, WIN/LOSS/TIMEOUT/AMBIGUOUS + sensibilidad)
│   ├── feature_engine.py        # Familias causales activables: ta_*, sm_* (proxies microestructura), ms_* (BOS/CHoCH/FVG/OB/Sweeps), deriv_*
│   ├── context_engine.py        # 6 regímenes causales + control temporal cuádruple (event/publication/effective/available_at)
│   ├── pipeline.py              # Adaptador compatible QuantFeaturePipeline
│   ├── splits.py                # Purged Walk-Forward Split + bóveda de Test Final Sellado
│   ├── probability_engine.py    # 5 modelos deterministas (BaseRate, LogReg, RF, GB, LightGBM) LONG/SHORT
│   ├── calibration_engine.py    # Calibración (None, Platt, Isotonic), ECE/MCE, reliability tables y thresholds
│   ├── ev_engine.py             # EV Engine neto con CostModel completo y distribución real de TIMEOUT
│   ├── backtest_engine.py       # Backtest barra a barra determinista, estado aislado y escenarios adversos
│   ├── validation.py            # Walk-Forward purgado, Bootstrap CI, Monte Carlo, Deflated Sharpe y regímenes
│   ├── ablation.py              # Ablation M0–M4 sobre validación + Feature/Permutation Importance + estabilidad
│   ├── experiment_tracker.py    # Registro reproducible EXP-XXXX (JSON + JSONL + git commit hash)
│   ├── brain.py                 # Adaptador compatible BenterQuantBrain determinista
│   ├── train_ablation.py        # Pipeline de entrenamiento/ablación sobre Validation (Test Final Sellado intacto)
│   ├── risk_engine.py           # Risk Engine independiente del modelo (Kelly incierto, caps, kill switch)
│   ├── monitoring.py            # Journal JSONL de señales, comparación previsto vs realizado y errores
│   ├── simulator.py             # Simulador Paper Trading (horizonte dinámico, slippage en TP, estado aislado)
│   ├── execution_engine.py      # Motor de ejecución seguro Testnet (DRY_RUN=true, clientOrderId propio, idempotencia, persistencia entry_time)
│   └── live_testnet_bot.py      # Orquestador Paper / Testnet (Mainnet bloqueado)
├── tests/                       # Suite automatizada real (invarianza causal, reproducibilidad, seguridad, E2E)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 4. Notas de Rigor Científico

- **Cero Look-Ahead Bias:** Todas las features (`ta_*`, `sm_*`, `ms_*`, `deriv_*`, `ctx_*`) son estrictamente causales (verificado con pruebas bit-a-bit en `tests/test_leakage.py`). Los Swing Highs/Lows se confirman con retraso `swing_right`.
- **Proxies de Microestructura:** Las columnas `sm_*` (`sm_ofi_*`, `sm_avg_trade_size_zscore`, mechas) son **proxies estadísticos de microestructura** sobre velas agregadas, no detección confirmada de ballenas ni flujo institucional.
- **Interpretabilidad vs Causalidad:** La importancia de features (nativa o por permutación) mide asociación predictiva y **no implica causalidad**.
- **Test Final Sellado:** `TemporalDatasetPartitioner` bloquea el acceso por defecto (`PermissionError`) durante todo el desarrollo, ablación y selección de modelos.

---

## 5. Instalación y Comandos

```bash
# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Ejecutar la suite completa de pruebas automatizadas
PYTHONPATH=src .venv/bin/pytest -v

# Descargar datos históricos verificados desde Binance Vision
PYTHONPATH=src .venv/bin/python src/download_data.py

# Ejecutar ablación walk-forward sobre validación (sin tocar el Test Final Sellado)
PYTHONPATH=src .venv/bin/python src/train_ablation.py

# Ejecutar evaluación en modo Paper Trading (predeterminado, sin capital real)
PYTHONPATH=src .venv/bin/python src/live_testnet_bot.py --mode paper --once
```
