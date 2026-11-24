# DATA DICTIONARY — Binance Quant Lab

Documento oficial de especificación de fuentes de datos, esquemas, reglas de normalización, controles de calidad (*Data Quality Gate*) y versionado reproducible de datasets (Fase 1 — Data Engineering).

---

## 1. Fuentes de Datos

Todas las series históricas de mercado se obtienen de los archivos mensuales públicos de **Binance Vision (USD-M Futures)**:

- **Base URL:** `https://data.binance.vision/data/futures/um/monthly`
- **Módulo de descarga:** `src/download_data.py`
- **Módulo de carga y normalización:** `src/data_loader.py`
- **Módulo de calidad de datos:** `src/data_quality.py`

| Serie | Símbolo por defecto | Frecuencia | Ruta remota (relativa a Base URL) | Uso en el sistema |
| :--- | :--- | :--- | :--- | :--- |
| Klines (Activo Principal) | `BTCUSDT` | `15m` | `/klines/BTCUSDT/15m/BTCUSDT-15m-YYYY-MM.zip` | Serie principal OHLCV, features de precio/microestructura, Contexto HTF y Triple-Barrier labeling |
| Klines (Activo Contexto) | `ETHUSDT` | `15m` | `/klines/ETHUSDT/15m/ETHUSDT-15m-YYYY-MM.zip` | Fuerza relativa (`BTC -> ETH`) y régimen cross-asset |
| Funding Rate | `BTCUSDT` | `8h` (`00:00`, `08:00`, `16:00` UTC) | `/fundingRate/BTCUSDT/BTCUSDT-fundingRate-YYYY-MM.zip` | Capa de derivados / posicionamiento y simulación de costes de funding |

### Estructura de almacenamiento local

```text
data/
├── raw/
│   ├── symbol=BTCUSDT/
│   │   ├── BTCUSDT-15m-2024-01.csv
│   │   ├── ...
│   │   └── BTCUSDT-fundingRate-2024-01.csv
│   └── symbol=ETHUSDT/
│       └── ETHUSDT-15m-2024-01.csv
└── manifests/
    ├── download_manifest_YYYYMMDDTHHMMSSZ.json
    └── BTCUSDT_15m_dataset_v1.0.json
```

---

## 2. Esquemas de Columnas y Tipos

### 2.1 Klines — CSV crudo (Binance Vision)

Cada CSV mensual contiene 12 columnas (algunos meses incluyen fila de cabecera y otros comienzan directamente con valores numéricos):

| # | Columna cruda | Tipo crudo | Descripción |
| :-: | :--- | :--- | :--- |
| 0 | `open_time` | `int64` | Epoch de apertura de la vela (en milisegundos `~1e12` o microsegundos `~1e15`) |
| 1 | `open` | `float64` | Precio de apertura (USDT) |
| 2 | `high` | `float64` | Precio máximo de la vela (USDT) |
| 3 | `low` | `float64` | Precio mínimo de la vela (USDT) |
| 4 | `close` | `float64` | Precio de cierre (USDT) |
| 5 | `volume` | `float64` | Volumen negociado en activo base (ej. BTC) |
| 6 | `close_time` | `int64` | Epoch de cierre de la vela |
| 7 | `quote_volume` | `float64` | Volumen negociado en activo cotizado (USDT) |
| 8 | `count` | `float64` | Número de trades ejecutados en la vela |
| 9 | `taker_buy_volume` | `float64` | Volumen comprado agresivamente por takers (activo base) |
| 10 | `taker_buy_quote_volume` | `float64` | Volumen comprado agresivamente por takers (USDT) |
| 11 | `ignore` | `float64` | Campo reservado de Binance (descartado) |

### 2.2 Klines — DataFrame normalizado (`HistoricalDataLoader.load_klines` / `load_and_validate_klines`)

| Campo | Rol | Tipo | Zona Horaria / Unidad | Restricciones |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | Índice (`DatetimeIndex`) | `datetime64[ns, UTC]` | **UTC estricta** | Ordenado ascendente, sin duplicados, sin `NaT` |
| `open` | Columna | `float64` | USDT | Finito, `> 0` |
| `high` | Columna | `float64` | USDT | Finito, `> 0`, `high >= max(open, close)` |
| `low` | Columna | `float64` | USDT | Finito, `> 0`, `low <= min(open, close)`, `low <= high` |
| `close` | Columna | `float64` | USDT | Finito, `> 0` |
| `volume` | Columna | `float64` | Base asset | Finito, `>= 0` |
| `quote_volume` | Columna | `float64` | USDT | Finito, `>= 0` |
| `count` | Columna | `float64` | Conteo de trades | Finito, `>= 0` |
| `taker_buy_volume` | Columna | `float64` | Base asset | Finito, `>= 0` |

### 2.3 Funding Rate — DataFrame normalizado (`HistoricalDataLoader.load_funding_rate`)

| Campo | Rol | Tipo | Zona Horaria / Unidad | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | Índice (`DatetimeIndex`) | `datetime64[ns, UTC]` | **UTC estricta** | Redondeado hacia abajo al minuto (`floor("1min")`) para alinear `00:00`, `08:00`, `16:00` UTC |
| `funding_rate` | Columna | `float64` | Tasa decimal por intervalo de 8h | Ej. `0.0001` equivale a `0.01%` |

---

## 3. Reglas de Zona Horaria (UTC) y Normalización

1. **Política UTC estricta:** Todos los índices temporales deben ser `pd.DatetimeIndex` con zona horaria `UTC`. Los timestamps *naive* (sin zona horaria) o en otra zona horaria son rechazados por el *Data Quality Gate*.
2. **Normalización milisegundos vs microsegundos (`_normalize_epoch_ms`):**
   - Binance Vision ha mezclado archivos mensuales expresados en milisegundos (`> 1e11`) y en microsegundos (`> 1e14`).
   - La conversión se realiza **fila por fila**: todo valor `> 1e14` se divide entre `1000.0` con `np.floor` antes de aplicar `pd.to_datetime(..., unit="ms", utc=True)`.
3. **Detección automática de cabecera:** Antes de leer cada CSV se inspecciona la primera celda; si no es numérica se trata como archivo con cabecera y se estandarizan los nombres de columna a `KLINE_RAW_COLS`.
4. **Ordenamiento y deduplicación:** Tras concatenar los CSV mensuales se ordena por índice UTC ascendente y se eliminan timestamps duplicados conservando el último registro (`keep="last"`).
5. **Alineación de Funding Rate:** Los timestamps de funding (que Binance a veces registra con 1–15 ms de desfase, ej. `00:00:00.002`) se truncan al minuto mediante `.dt.floor("1min")`.

---

## 4. Integridad de Descarga (`src/download_data.py`)

Cada descarga desde Binance Vision aplica controles de integridad:

- **Hash SHA256 (`compute_file_sha256`):** Se calcula el digest SHA256 de todo archivo CSV descargado o reutilizado.
- **Verificación opcional `.CHECKSUM` (`verify_remote_checksum`):** Se consulta `<url>.CHECKSUM`.
  - Si devuelve HTTP `200`, el SHA256 del `.zip` debe coincidir exactamente; en caso contrario (`mismatch`) se rechaza el archivo y no se escribe en disco.
  - Si devuelve HTTP `404`, no se aborta la descarga, pero se registra `checksum_status: "not_available_404"` en el manifiesto de descarga.
- **Detección de archivos vacíos o corruptos (`is_valid_csv_file` / `is_valid_csv_bytes`):** Se detectan ZIPs corruptos (`BadZipFile`), ZIPs sin CSV, archivos de 0 bytes, bytes NUL (`\x00`) o contenido no decodificable en UTF-8. La escritura a disco se realiza de forma atómica (`*.tmp` + `os.replace`).
- **No sobrescritura silenciosa:** Por defecto `overwrite=False`. Si un CSV válido ya existe en disco, se omite la descarga y se registra su SHA256 local.
- **Manifiesto de descarga:** Cada ejecución de `run_downloader` escribe `data/manifests/download_manifest_<YYYYMMDDTHHMMSSZ>.json`.

---

## 5. Data Quality Gate (`src/data_quality.py`)

El entrenamiento y la evaluación **se detienen inmediatamente** (`ValueError` vía `DataQualityGate.assert_valid(report)`) si los datos no superan los controles críticos.

### Timeframes soportados (`timeframe_to_timedelta`)
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`.

### Controles en `DataQualityGate.validate_klines`

| Control | Métrica en `DataQualityReport` | Severidad | Criterio de aprobación |
| :--- | :--- | :--- | :--- |
| Dataset no vacío | `row_count` | **Crítico** | `row_count > 0` |
| Índice temporal UTC | `errors` | **Crítico** | `DatetimeIndex` con `tz == UTC` |
| Orden ascendente | `out_of_order_count` | **Crítico** | `out_of_order_count == 0` |
| Timestamps duplicados | `duplicate_count` | **Crítico** | `duplicate_count == 0` |
| Velas faltantes y gaps | `missing_bars_count`, `gap_details` | **Crítico** (si excede umbral) / *Warning* | `missing_ratio <= max_missing_ratio` (por defecto `0.01` = 1%) |
| Precios inválidos (`<= 0`, `NaN`, `inf`) | `invalid_price_count` | **Crítico** | `invalid_price_count == 0` |
| Consistencia OHLC | `invalid_ohlc_count` | **Crítico** | `low <= min(open, close)`, `high >= max(open, close)`, `low <= high` (`== 0`) |
| Volúmenes negativos o no finitos | `negative_volume_count` | **Crítico** | `negative_volume_count == 0` |
| Retornos extremos sospechosos | `extreme_outlier_count` | *Warning* | `|pct_change(close)| > extreme_return_threshold` (por defecto `0.30` = 30%). Registra advertencia sin bloquear por sí solo |

### Alineación multi-fuente (`DataQualityGate.validate_alignment`)

Verifica el solapamiento temporal entre `df_primary` (BTCUSDT), `df_context` (ETHUSDT) y `df_funding` (Funding Rate 8h), reportando `overlap_bars`, `overlap_ratio`, velas desalineadas en cada dirección y cobertura temporal del funding.

---

## 6. Versionado de Datasets y Manifiestos (`DataManifest`)

Cada dataset validado se identifica mediante un nombre versionado siguiendo la convención:

```text
<SYMBOL>_<TIMEFRAME>_dataset_v<MAJOR>.<MINOR>
```

Ejemplo estándar: **`BTCUSDT_15m_dataset_v1.0`**

### Esquema del manifiesto (`src/contracts.py` -> `DataManifest`)

| Campo | Tipo | Descripción |
| :--- | :--- | :--- |
| `dataset_version` | `str` | Identificador único de versión (ej. `BTCUSDT_15m_dataset_v1.0`) |
| `symbol` | `str` | Símbolo del activo (ej. `BTCUSDT`) |
| `timeframe` | `str` | Intervalo de vela (ej. `15m`) |
| `source` | `str` | Origen de los datos (ej. `binance_vision_um_monthly`) |
| `start_timestamp` | `str` | Primer timestamp UTC en formato ISO-8601 |
| `end_timestamp` | `str` | Último timestamp UTC en formato ISO-8601 |
| `row_count` | `int` | Número total de velas normalizadas |
| `downloaded_at` | `str` | Timestamp UTC ISO-8601 de generación del manifiesto |
| `schema_version` | `str` | Versión del esquema de columnas (ej. `1.0`) |
| `dataset_hash` | `str` | SHA256 determinista calculado sobre índice UTC (ms) + columnas float64 canónicas |
| `file_hashes` | `Dict[str, str]` | Mapa `nombre_archivo.csv -> SHA256` de los CSV fuente |
| `transformations` | `List[str]` | Lista ordenada de transformaciones aplicadas durante la carga |
| `quality_passed` | `bool` | Resultado booleano del *Data Quality Gate* (`True`) |

### Regla de inmutabilidad

Al ejecutar `HistoricalDataLoader.load_and_validate_klines(..., save_manifest=True)`:

- El manifiesto se guarda en `<manifests_dir>/<dataset_version>.json`.
- Si ya existe un manifiesto con la misma `dataset_version` y el **mismo `dataset_hash`**, la operación es idempotente y se confirma reproducibilidad.
- Si ya existe un manifiesto con la misma `dataset_version` pero un **`dataset_hash` distinto**, se lanza **`FileExistsError`** para impedir la sobrescritura silenciosa de datasets históricos. Para publicar nuevos datos debe incrementarse `dataset_version` (ej. `BTCUSDT_15m_dataset_v1.1`).
