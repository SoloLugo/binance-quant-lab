# AUDIT_REPORT.md — Fase 0: Auditoría y Estabilización (Binance Quant Lab)

**Fecha de auditoría:** 2026-11-24  
**Rama / Commit base auditado:** `main` @ `b39ce1b`  
**Especificación de referencia:** Plan Maestro de Desarrollo, Corrección y Validación Cuantitativa — Binance Quant Lab

---

## 1. Estado Inicial Encontrado (Previo a Estabilización)

### 1.1 Inventario de Módulos y Líneas
- `src/__init__.py` (7 líneas) — versión `2.0.0`.
- `src/config.py` (73 líneas) — `QuantConfig` con lectura de `os.getenv` a tiempo de importación; permitía `EXECUTION_MODE=mainnet` y cargaba credenciales `BINANCE_MAINNET_API_KEY` / `BINANCE_MAINNET_SECRET`.
- `src/download_data.py` (103 líneas) — descarga paralela de ZIPs de Binance Vision (BTCUSDT, ETHUSDT, Funding) sin verificación SHA256 ni manifiestos.
- `src/data_loader.py` (93 líneas) — carga de CSVs con normalización ms/us; sin validación de gaps, OHLC inconsistentes ni versionado.
- `src/pipeline.py` (259 líneas) — 4 capas de features + Triple-Barrier Labeling con entrada en `close[i]` y 2 errores de look-ahead bias (`.bfill()`).
- `src/brain.py` (191 líneas) — LightGBM + `IsotonicRegression(y_min=0.01, y_max=0.75)` + EV/Kelly + `evaluate_model`. Parámetro `purge_bars` no utilizado.
- `src/train_ablation.py` (152 líneas) — split 70/15/15 y selección de `best_model_name` directamente sobre `df_test` (contaminación del conjunto de test).
- `src/simulator.py` (411 líneas) — simulador Paper Trading con persistencia JSON/CSV reutilizada entre corridas, `horizon_seconds` hardcodeado a 15m y TP sin slippage.
- `src/live_testnet_bot.py` (901 líneas) — bot Paper/Testnet/Mainnet + Replay; failover silencioso a Spot (`data-api.binance.vision`) y Testnet; `cancel_all_orders` indiscriminado; `active_exchange_entry_ts` solo en memoria.
- `tests/test_quant_lab.py` (179 líneas) — 5 pruebas unitarias.

### 1.2 Verificación de Secretos y Dependencias
- **Secretos en el repositorio:** Escaneo sobre `src/`, `tests/`, `Dockerfile`, `docker-compose.yml` y `README.md` confirmó **0 API keys o secretos hardcodeados** (solo placeholders en `README.md` y lecturas de variables de entorno).
- **Entorno e instalación:** El Python base de la VM no tenía instaladas las dependencias (`ccxt`, `pytest`, etc.). Se creó un entorno virtual reproducible `.venv` e instaló `requirements.txt` (incluyendo `ccxt 4.5.94`, `lightgbm 4.7.0`, `scikit-learn 1.9.1`, `pandas 3.1.1`, `numpy 2.5.5`, `pytest 9.2.0`).
- **Ejecución de pruebas base (antes de Sprint 0):**
  - Comando: `PYTHONPATH=src .venv/bin/pytest -v`
  - Resultado: **4 passed, 1 failed**
  - Fallo exacto: `test_safe_position_sizing_and_emergency_sl_rollback` falló con `FileNotFoundError: No se encontró el modelo entrenado en 'models/quant_brain.joblib'`.

### 1.3 Diferencias entre `README.md` y Código Real
1. `README.md` presenta `sm_whale_trade_zscore` y `sm_ofi_*` como detección de "ballenas / venta institucional agresiva", cuando son proxies de microestructura basados en volumen taker y tamaño medio de trade.
2. `README.md` indica `EV_MIN_THRESHOLD=0.12`, pero `train_ablation.py` hardcodeaba `ev_min_threshold=0.10`.
3. `README.md` describe la ablación como validación científica out-of-sample, pero `train_ablation.py` seleccionaba el mejor modelo usando el propio conjunto de test (`df_test`).
4. `README.md` documenta el modo `mainnet` como disponible para producción con dinero real, lo cual incumple la regla de protección de capital (§1.4) antes de superar validación independiente.

---

## 2. Errores y Riesgos Críticos Documentados

| ID | Severidad | Archivo y Línea | Problema | Fase de Corrección |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-01** | Crítica | `src/config.py:17,58-66`, `src/live_testnet_bot.py:44-94,866` | Mainnet y dinero real habilitados sin bloqueo estricto ni `DRY_RUN=true`. | **Fase 0 (Sprint 0)** |
| **AUD-02** | Crítica | `src/pipeline.py:110,137` | Look-ahead bias explícito mediante `.bfill()` al alinear Funding Rate y ETHUSDT. | **Fase 3 (Sprint 1)** |
| **AUD-03** | Crítica | `src/train_ablation.py:78-109` | Selección del mejor modelo (`M1`–`M4`) usando métricas del conjunto de test `df_test`. | **Fase 5 (Sprint 2) + Splits en Sprint 1** |
| **AUD-04** | Alta | `src/pipeline.py:184-239` | Entrada en `close[i]` en vez de siguiente punto ejecutable (`next Open`); `AMBIGUOUS` intravela resuelto asumiendo `SL` primero; `TIMEOUT` colapsado en etiqueta `0.0` y penalizado como `-1R` en EV. | **Fase 2 (Sprint 1) y Fase 7 (Sprint 3)** |
| **AUD-05** | Alta | `src/download_data.py`, `src/data_loader.py` | Sin Data Quality Gate, sin SHA256/checksums, sin manifiestos ni versionado de datasets. | **Fase 1 (Sprint 1)** |
| **AUD-06** | Alta | `src/brain.py:43,47-77,106` | `purge_bars` no utilizado; techo duro `y_max=0.75` sin justificación estadística; EV omite spread, slippage y funding. | **Fases 5–7 (Sprints 2–3)** |
| **AUD-07** | Alta | `src/simulator.py:38,164,257-269` | Estado persistente reutilizado entre corridas; `horizon_seconds` fijo a 15m; TP sin slippage. | **Fase 0 (parcial) y Fases 8/13 (Sprints 3 y 6)** |
| **AUD-08** | Alta | `src/live_testnet_bot.py:193-231,392,403,592,778` | Failover silencioso a Spot/Testnet; `cancel_all_orders` indiscriminado; `active_exchange_entry_ts` solo en memoria; re-entrada en la misma vela de cierre en replay; telemetría SHORT imprime `b_net_odds` de LONG. | **Fases 13–14 (Sprints 6–7)** |
| **AUD-09** | Alta | `tests/test_quant_lab.py:140` | Test falla en checkout limpio por requerir `models/quant_brain.joblib`; faltan pruebas de leakage, splits, reproducibilidad y seguridad. | **Fase 0 (Sprint 0) y Sprints 1–7** |

---

## 3. Estabilización Realizada en Sprint 0

- Entorno `.venv` creado con todas las dependencias de `requirements.txt` verificadas.
- `src/config.py`: añadido `QuantConfig.from_env()`, modo predeterminado `paper`, `dry_run=True` por defecto y bloqueo irrompible contra `mainnet` y `USE_TESTNET=false` (`PermissionError`).
- `src/contracts.py`: definidos los contratos inmutables compartidos entre todas las capas (`DataQualityReport`, `DataManifest`, `LabelSpec`, `CostModel`, `SplitSpec`, `RiskDecision`, `ExperimentRecord`).
- `src/live_testnet_bot.py` y `src/simulator.py`: validación obligatoria de modo (bloqueo de Mainnet en `__init__` y CLI), respeto a `dry_run` y soporte de `fresh_state=True` para evitar reutilización accidental de estado persistente.
- `tests/`: fixtures reales en `tests/conftest.py`, estabilización de `tests/test_quant_lab.py` para checkout limpio y nueva suite `tests/test_safety_mainnet_block.py`.

---

## 4. Lista de Tareas Priorizadas (Sprints 1–7)

1. **Sprint 1 (Fases 1, 2, 3-base, 4-base):** `data_quality.py`, SHA256/manifiestos, `DATA_DICTIONARY.md`, `label_engine.py` (`next Open`, 4 estados), `feature_engine.py` sin `.bfill()`, `context_engine.py` base, `splits.py` (Purging/Embargo + Test Final Sellado), `test_leakage.py`.
2. **Sprint 2 (Fases 5 y 6):** `probability_engine.py` (5 niveles de modelos LONG/SHORT), `calibration_engine.py` (None/Platt/Isotonic, ECE/MCE, reliability tables), selección estricta en Validation.
3. **Sprint 3 (Fases 7 y 8):** `ev_engine.py` (`CostModel` + `TIMEOUT` real) y `backtest_engine.py` (contabilidad exacta y escenarios adversos).
4. **Sprint 4 (Fases 9, 10 y 11):** Walk-forward, robustez (bootstrap/Monte Carlo/regímenes), Ablation M0–M4 + interpretabilidad, `experiment_tracker.py` (`EXP-XXXX`).
5. **Sprint 5 (Fases 3-exp y 4-exp):** Familia B algorítmica (BOS/CHoCH/FVG/OB/Sweeps), derivados y `context_engine.py` completo con control temporal cuádruple.
6. **Sprint 6 (Fases 12 y 13):** `risk_engine.py` independiente, Paper Trading aislado y `monitoring.py`.
7. **Sprint 7 (Fase 14 + Cierre):** `execution_engine.py` seguro en Testnet (`clientOrderId`, persistencia `entry_time`, `DRY_RUN=true`) + evaluación única sobre el Test Final Sellado + actualización final de `README.md`.
