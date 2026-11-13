import argparse
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import ccxt
import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from config import settings
from simulator import RealTimeFuturesSimulator

load_dotenv()

VISION_SPOT_API = "https://data-api.binance.vision/api/v3"
TESTNET_FAPI_URL = "https://testnet.binancefuture.com/fapi/v1"


class BinanceFuturesTestnetBot:
    """
    Motor Híbrido de Inferencia en Tiempo Real, Simulación Paper Institucional y Ejecución
    Segura en Binance USD-M Futures (Testnet y Mainnet).
    """
    def __init__(
        self,
        model_path: str = settings.model_artifact_path,
        execution_mode: Optional[str] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en '{model_path}'. "
                f"Ejecuta primero: python src/train_ablation.py"
            )

        bundle = joblib.load(model_path)
        self.brain = bundle["brain"]
        self.brain_long = bundle.get("brain_long", self.brain)
        self.brain_short = bundle.get("brain_short", None)
        self.pipe = bundle["pipeline"]
        self.model_name = bundle.get("best_model_name", "M4_Full_ContextEngine")

        # Determinamos el modo de operación: 'paper', 'testnet' o 'mainnet'
        raw_mode = (execution_mode or settings.execution_mode).lower().strip()
        raw_key = settings.binance_api_key if raw_mode != "mainnet" else settings.mainnet_api_key
        raw_secret = settings.binance_secret if raw_mode != "mainnet" else settings.mainnet_secret
        api_key = (raw_key or "").strip().strip('"').strip("'")
        api_secret = (raw_secret or "").strip().strip('"').strip("'")

        # Si se pide testnet/mainnet pero no se configuraron llaves API válidas, activamos 'paper' automáticamente
        if raw_mode in ("testnet", "mainnet") and (
            not api_key or not api_secret or "tu_api_key" in api_key
        ):
            print(
                f"[AVISO] Credenciales API no configuradas para modo '{raw_mode}'. "
                f"Activando modo de Simulación en Tiempo Real ('paper')."
            )
            self.execution_mode = "paper"
        else:
            self.execution_mode = raw_mode

        # 1. Cliente Público en MAINNET con failover automático ante bloqueos regionales (HTTP 451)
        self.market_data_client = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 15000})
        self._http = requests.Session()

        # 2. Motor de Simulación en Tiempo Real (siempre disponible para telemetría o modo paper)
        self.simulator = RealTimeFuturesSimulator()

        # 3. Cliente Privado de Ejecución (Testnet o Mainnet)
        self.execution_client: Optional[ccxt.binanceusdm] = None
        self.active_exchange_entry_ts: Optional[float] = None

        if self.execution_mode in ("testnet", "mainnet"):
            self.execution_client = ccxt.binanceusdm({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "timeout": 15000,
                "options": {
                    "defaultType": "future",
                    "disableFuturesSandboxWarning": True,
                    "fetchCurrencies": False,
                    "recvWindow": 10000,
                },
            })
            if self.execution_mode == "testnet":
                use_demo = os.getenv("USE_DEMO_TRADING", "false").lower() == "true"
                if use_demo and hasattr(self.execution_client, "enable_demo_trading"):
                    self.execution_client.enable_demo_trading(True)
                else:
                    self.execution_client.set_sandbox_mode(True)
                    self.execution_client.options["disableFuturesSandboxWarning"] = True
            self._configure_exchange_account()

    def _configure_exchange_account(self) -> None:
        """
        Configura el apalancamiento máximo y el modo de margen en el exchange al iniciar.
        En modo Testnet, verifica las credenciales tanto en `testnet.binancefuture.com`
        como en `demo-fapi.binance.com`. Si ambas rechazan la API Key (-2015), conmuta
        automáticamente al Simulador en Tiempo Real ('paper') para operar sin interrupciones.
        """
        if self.execution_client is None:
            return
        try:
            if self.execution_mode == "testnet":
                try:
                    self.execution_client.fetch_balance()
                except Exception as first_exc:
                    if hasattr(self.execution_client, "enable_demo_trading"):
                        try:
                            self.execution_client.set_sandbox_mode(False)
                            self.execution_client.enable_demo_trading(True)
                            self.execution_client.fetch_balance()
                            print("[INFO] Conectado exitosamente a Binance Demo Trading (demo-fapi.binance.com).")
                        except Exception as demo_exc:
                            print(
                                f"\n[AVISO DE CREDENCIALES] La API Key en .env fue rechazada tanto en "
                                f"Futures Testnet como en Demo Trading ({demo_exc}).\n"
                                f"Causas comunes del error -2015:\n"
                                f"  1. La llave fue creada en Spot Testnet (testnet.binance.vision) o Mainnet en vez de "
                                f"Futures Testnet (testnet.binancefuture.com) o Demo Trading (demo.binance.com).\n"
                                f"  2. Falta activar el permiso 'Enable Futures' o la llave expiró.\n"
                                f"-> Conmutando automáticamente a modo SIMULACIÓN EN TIEMPO REAL ('paper') "
                                f"con $10,000 USDT virtuales y datos de mercado en vivo.\n"
                            )
                            self.execution_mode = "paper"
                            self.execution_client = None
                            return
                    else:
                        print(
                            f"[AVISO] Credenciales Testnet rechazadas ({first_exc}). "
                            f"Conmutando a modo Simulación en Tiempo Real ('paper')."
                        )
                        self.execution_mode = "paper"
                        self.execution_client = None
                        return

            self.execution_client.load_markets()
            try:
                self.execution_client.set_margin_mode(settings.margin_mode, settings.ccxt_symbol)
            except Exception:
                pass
            try:
                self.execution_client.set_leverage(settings.max_leverage, settings.ccxt_symbol)
            except Exception as exc:
                print(f"[WARN] No se pudo fijar apalancamiento {settings.max_leverage}x: {exc}")
        except Exception as exc:
            print(f"[WARN] Inicialización de mercados en {self.execution_mode} diferida: {exc}")

    def _parse_raw_klines(self, raw_klines: list) -> pd.DataFrame:
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(raw_klines, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[num_cols].dropna()

    def fetch_live_dataframe(
        self,
        symbol: str,
        limit: int = 1500,
        include_forming_bar: bool = False
    ) -> pd.DataFrame:
        """
        Descarga las últimas `limit` velas de 15m (~15.6 días) en tiempo real.
        Intenta primero Binance Futures Mainnet (`fapi.binance.com`). Si la IP del servidor
        tiene restricción geográfica (HTTP 451), hace failover transparente a `data-api.binance.vision`
        (datos orgánicos reales de Binance Mainnet) con paginación para obtener las 1500 velas completas.
        """
        clean_sym = symbol.replace("/", "").replace(":USDT", "")
        raw_klines = None

        # Intento 1: Binance Futures Mainnet directo
        try:
            raw_klines = self.market_data_client.fapiPublicGetKlines({
                "symbol": clean_sym,
                "interval": settings.timeframe,
                "limit": min(limit, 1500),
            })
        except Exception:
            raw_klines = None

        # Intento 2: Failover a Binance Vision API (Mainnet Real Data, hasta 1000 velas por página)
        if not raw_klines:
            try:
                batch_2 = self._http.get(
                    f"{VISION_SPOT_API}/klines",
                    params={"symbol": clean_sym, "interval": settings.timeframe, "limit": min(limit, 1000)},
                    timeout=15,
                )
                batch_2.raise_for_status()
                k_recent = batch_2.json()

                if limit > 1000 and len(k_recent) > 0:
                    first_open_ms = int(k_recent[0][0])
                    needed = limit - len(k_recent) + 1
                    batch_1 = self._http.get(
                        f"{VISION_SPOT_API}/klines",
                        params={
                            "symbol": clean_sym,
                            "interval": settings.timeframe,
                            "limit": min(needed, 1000),
                            "endTime": first_open_ms - 1,
                        },
                        timeout=15,
                    )
                    batch_1.raise_for_status()
                    raw_klines = batch_1.json() + k_recent
                else:
                    raw_klines = k_recent
            except Exception:
                raw_klines = None

        # Intento 3: Último recurso vía Testnet Futures endpoint
        if not raw_klines:
            resp = self._http.get(
                f"{TESTNET_FAPI_URL}/klines",
                params={"symbol": clean_sym, "interval": settings.timeframe, "limit": min(limit, 1500)},
                timeout=15,
            )
            resp.raise_for_status()
            raw_klines = resp.json()

        df = self._parse_raw_klines(raw_klines)
        if include_forming_bar:
            return df
        # Descartamos la última vela en formación para evaluar únicamente velas cerradas
        return df.iloc[:-1]

    def fetch_live_funding(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Descarga el historial reciente de Funding Rate en tiempo real con failover automático
        y normaliza los timestamps al minuto exacto (evitando desfases de milisegundos).
        """
        clean_sym = symbol.replace("/", "").replace(":USDT", "")
        raw_funding = None

        try:
            raw_funding = self.market_data_client.fapiPublicGetFundingRate({
                "symbol": clean_sym,
                "limit": 200,
            })
        except Exception:
            resp = self._http.get(
                f"{TESTNET_FAPI_URL}/fundingRate",
                params={"symbol": clean_sym, "limit": 200},
                timeout=15,
            )
            resp.raise_for_status()
            raw_funding = resp.json()

        df = pd.DataFrame(raw_funding)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True).dt.floor("1min")
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df[["funding_rate"]].dropna()

    def fetch_live_market_snapshot(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Obtiene en tiempo real el libro de órdenes (Bid / Ask), el Mark Price y el Funding Rate
        vigente para simular y ejecutar órdenes sin distorsión de precios.
        """
        clean_sym = symbol.replace("/", "").replace(":USDT", "")
        bid_price, ask_price, mark_price, funding_rate = None, None, None, 0.0001

        # 1. Bid/Ask real en Mainnet (o en el venue de ejecución si es Testnet)
        if self.execution_mode == "testnet":
            try:
                bt = self._http.get(
                    f"{TESTNET_FAPI_URL}/ticker/bookTicker", params={"symbol": clean_sym}, timeout=10
                ).json()
                bid_price = float(bt["bidPrice"])
                ask_price = float(bt["askPrice"])
            except Exception:
                pass

        if bid_price is None or ask_price is None:
            try:
                bt = self._http.get(
                    f"{VISION_SPOT_API}/ticker/bookTicker", params={"symbol": clean_sym}, timeout=10
                ).json()
                bid_price = float(bt["bidPrice"])
                ask_price = float(bt["askPrice"])
            except Exception:
                pass

        # 2. Mark Price y Last Funding Rate desde endpoint de Futuros
        try:
            pi = self._http.get(
                f"{TESTNET_FAPI_URL}/premiumIndex", params={"symbol": clean_sym}, timeout=10
            ).json()
            mark_price = float(pi.get("markPrice", 0.0))
            funding_rate = float(pi.get("lastFundingRate", 0.0001))
        except Exception:
            pass

        if bid_price is not None and ask_price is not None:
            mid = 0.5 * (bid_price + ask_price)
            # En modo paper usamos el mid orgánico como mark_price si Testnet diverge >0.5%
            if self.execution_mode == "paper" or mark_price is None or mark_price <= 0:
                mark_price = mid
            elif abs(mark_price - mid) / mid > 0.01 and self.execution_mode != "testnet":
                mark_price = mid
        elif mark_price is not None and mark_price > 0:
            bid_price = round(mark_price * 0.99995, 2)
            ask_price = round(mark_price * 1.00005, 2)
        else:
            raise RuntimeError("No fue posible obtener cotización en vivo de BTCUSDT.")

        return {
            "bid_price": float(bid_price),
            "ask_price": float(ask_price),
            "mark_price": float(mark_price),
            "funding_rate": float(funding_rate),
        }

    def compute_safe_position_size(
        self,
        available_balance_usdt: float,
        kelly_risk_fraction: float,
        entry_price: float,
        sl_distance_usdt: float,
    ) -> float:
        """
        Calcula el tamaño de posición en BTC respetando simultáneamente:
          1. Fracción de riesgo de Kelly (`kelly_risk_fraction <= max_risk_per_trade`).
          2. Tope estricto de apalancamiento (`max_leverage`).
          3. Lote mínimo (`min_qty_btc`) y nocional mínimo (`min_notional_usdt`),
             rechazando la operación (retorna 0.0) si el lote mínimo viola el riesgo máximo.
        """
        if available_balance_usdt <= 0 or kelly_risk_fraction <= 0 or sl_distance_usdt <= 0:
            return 0.0

        capped_risk_frac = min(float(kelly_risk_fraction), settings.max_risk_per_trade)
        risk_usdt = available_balance_usdt * capped_risk_frac

        # 1. Cantidad por riesgo de Stop Loss (1R = risk_usdt)
        raw_qty_by_risk = risk_usdt / sl_distance_usdt

        # 2. Límite por apalancamiento máximo permitido
        max_notional = available_balance_usdt * float(settings.max_leverage) * 0.95
        max_qty_by_leverage = max_notional / entry_price

        qty_btc = np.floor(min(raw_qty_by_risk, max_qty_by_leverage) * 1000.0) / 1000.0

        # Si el tamaño calculado es inferior al mínimo de Binance, verificamos si el mínimo es seguro
        min_qty = max(settings.min_qty_btc, np.ceil((settings.min_notional_usdt / entry_price) * 1000.0) / 1000.0)
        if qty_btc < min_qty:
            min_qty_risk_usdt = min_qty * sl_distance_usdt
            min_qty_notional = min_qty * entry_price
            # Solo permitimos redondear hacia arriba al lote mínimo si no excede el riesgo máximo ni el margen
            if (
                min_qty_risk_usdt <= available_balance_usdt * settings.max_risk_per_trade * 1.15
                and min_qty_notional <= max_notional
            ):
                qty_btc = min_qty
            else:
                return 0.0

        return round(float(qty_btc), 3)

    def reconcile_exchange_state(self) -> int:
        """
        Reconcilia el estado de órdenes y posiciones en Testnet/Mainnet:
          - Si no hay posición abierta (`open_contracts == 0`), cancela cualquier orden condicional
            huérfana (`STOP_MARKET` / `TAKE_PROFIT_MARKET`) que haya quedado activa de trades previos.
          - Si hay una posición abierta, verifica la barrera vertical de tiempo (`horizon_bars`)
            y garantiza que exista un Stop Loss activo en el libro.
        Retorna el número de contratos abiertos tras la reconciliación.
        """
        if self.execution_client is None:
            return 0

        positions = self.execution_client.fetch_positions([settings.ccxt_symbol])
        active_pos = [p for p in positions if abs(float(p.get("contracts", 0) or 0)) > 0]

        if not active_pos:
            self.active_exchange_entry_ts = None
            try:
                open_orders = self.execution_client.fetch_open_orders(settings.ccxt_symbol)
                if open_orders:
                    self.execution_client.cancel_all_orders(settings.ccxt_symbol)
                    print(f"[RECONCILIACIÓN] Canceladas {len(open_orders)} órdenes condicionales huérfanas.")
            except Exception as exc:
                print(f"[WARN] No se pudieron limpiar órdenes abiertas: {exc}")
            return 0

        pos = active_pos[0]
        contracts = abs(float(pos.get("contracts", 0)))
        side_str = str(pos.get("side", "long")).upper()
        now_ts = time.time()

        if self.active_exchange_entry_ts is None:
            pos_ts = pos.get("timestamp")
            self.active_exchange_entry_ts = (float(pos_ts) / 1000.0) if pos_ts else now_ts

        # Verificamos si expiró la barrera vertical de tiempo (horizon_bars * 15m)
        horizon_secs = settings.horizon_bars * 15 * 60
        if (now_ts - self.active_exchange_entry_ts) >= horizon_secs:
            close_side = "sell" if side_str == "LONG" else "buy"
            print(f"[TIME-STOP] Barrera vertical de {settings.horizon_bars} velas alcanzada. Cerrando a mercado...")
            try:
                self.execution_client.cancel_all_orders(settings.ccxt_symbol)
                self.execution_client.create_order(
                    settings.ccxt_symbol, "market", close_side, contracts, None, {"reduceOnly": True}
                )
                self.active_exchange_entry_ts = None
                return 0
            except Exception as exc:
                print(f"[ERROR CRÍTICO] Falló cierre por Time-Stop: {exc}")

        return 1

    def _place_exchange_bracket_orders(
        self,
        side: str,
        qty_btc: float,
        atr_pct: float,
    ) -> None:
        """
        Ejecuta la entrada a mercado en Testnet/Mainnet y coloca las órdenes nativas
        `STOP_MARKET` y `TAKE_PROFIT_MARKET` ancladas al precio real de ejecución del exchange.
        Si la colocación del Stop Loss falla, cierra inmediatamente la posición a mercado (rollback)
        para jamás dejar una posición apalancada desprotegida.
        """
        assert self.execution_client is not None
        side_upper = side.upper()
        entry_order_side = "buy" if side_upper == "LONG" else "sell"
        exit_order_side = "sell" if side_upper == "LONG" else "buy"

        # Limpiamos cualquier orden condicional residual antes de abrir
        try:
            self.execution_client.cancel_all_orders(settings.ccxt_symbol)
        except Exception:
            pass

        entry_order = self.execution_client.create_order(
            settings.ccxt_symbol, "market", entry_order_side, qty_btc
        )
        self.active_exchange_entry_ts = time.time()

        # Obtenemos el precio real de llenado en el exchange de ejecución (evita el desfase Mainnet vs Testnet)
        fill_price = float(entry_order.get("average") or entry_order.get("price") or 0.0)
        if fill_price <= 0:
            ticker = self.execution_client.fetch_ticker(settings.ccxt_symbol)
            fill_price = float(ticker.get("last") or ticker.get("close") or 0.0)

        sl_dist = fill_price * atr_pct * self.brain.sl_mult
        tp_dist = fill_price * atr_pct * self.brain.tp_mult

        if side_upper == "LONG":
            sl_price = round(fill_price - sl_dist, 1)
            tp_price = round(fill_price + tp_dist, 1)
        else:
            sl_price = round(fill_price + sl_dist, 1)
            tp_price = round(fill_price - tp_dist, 1)

        print(
            f"[{self.execution_mode.upper()}] Fill Real: {fill_price:.2f} USDT | "
            f"Colocando barreras -> TP: {tp_price:.1f} | SL: {sl_price:.1f} (MARK_PRICE)"
        )

        # Colocamos el Stop Loss con rollback de emergencia si falla
        try:
            self.execution_client.create_order(
                settings.ccxt_symbol,
                "STOP_MARKET",
                exit_order_side,
                None,
                None,
                {"stopPrice": sl_price, "closePosition": True, "workingType": "MARK_PRICE"},
            )
        except Exception as sl_exc:
            print(f"[EMERGENCIA] Falló colocación de STOP_MARKET ({sl_exc}). Ejecutando rollback inmediato...")
            self.execution_client.create_order(
                settings.ccxt_symbol, "market", exit_order_side, qty_btc, None, {"reduceOnly": True}
            )
            self.active_exchange_entry_ts = None
            raise RuntimeError(f"Posición cerrada por seguridad tras fallo en SL: {sl_exc}") from sl_exc

        # Colocamos el Take Profit
        try:
            self.execution_client.create_order(
                settings.ccxt_symbol,
                "TAKE_PROFIT_MARKET",
                exit_order_side,
                None,
                None,
                {"stopPrice": tp_price, "closePosition": True, "workingType": "MARK_PRICE"},
            )
        except Exception as tp_exc:
            print(f"[WARN] No se pudo colocar TAKE_PROFIT_MARKET ({tp_exc}), pero el SL está activo.")

    def monitor_realtime_positions(self) -> Optional[Dict[str, Any]]:
        """
        Pulso de monitoreo en tiempo real (cada `poll_interval_seconds` segundos):
          - En modo 'paper': actualiza el precio Mark/Bid/Ask, liquida funding de 8h y
            evalúa si se tocaron las barreras SL, TP o Time-Stop (6h).
          - En modo 'testnet'/'mainnet': reconcilia posiciones y limpia órdenes huérfanas.
        """
        if self.execution_mode == "paper":
            if not self.simulator.has_open_position():
                return None
            snap = self.fetch_live_market_snapshot(settings.primary_symbol)
            closed_trade = self.simulator.update_on_market_tick(
                mark_price=snap["mark_price"],
                bid_price=snap["bid_price"],
                ask_price=snap["ask_price"],
                funding_rate=snap["funding_rate"],
            )
            if closed_trade is not None:
                print(
                    f"\n[SIMULADOR - CIERRE {closed_trade['exit_reason']}] "
                    f"Trade #{closed_trade['trade_id']} ({closed_trade['side']}) | "
                    f"Entrada: {closed_trade['entry_price']:.2f} -> Salida: {closed_trade['exit_price']:.2f} | "
                    f"PnL Neto: {closed_trade['net_pnl_usdt']:+.2f} USDT ({closed_trade['realized_net_r']:+.2f}R) | "
                    f"Balance: {closed_trade['balance_after_usdt']:.2f} USDT"
                )
            return closed_trade
        else:
            self.reconcile_exchange_state()
            return None

    def evaluate_and_execute(self, ev_threshold: float = settings.ev_min_threshold) -> Dict[str, Any]:
        """
        Evalúa la última vela cerrada de 15m mediante el Context Engine y el Probability Engine
        (tanto en dirección LONG como SHORT) y ejecuta la operación si el Valor Esperado supera
        el umbral mínimo de Bill Benter.
        """
        print(f"\n--- Evaluando Cierre de Vela 15m [{self.execution_mode.upper()} | {self.model_name}] ---")

        # Primero actualizamos / reconciliamos cualquier posición abierta
        self.monitor_realtime_positions()

        df_btc = self.fetch_live_dataframe(settings.primary_symbol, limit=1500)
        df_eth = self.fetch_live_dataframe(settings.context_symbol, limit=1500)
        df_funding = self.fetch_live_funding(settings.primary_symbol)
        snap = self.fetch_live_market_snapshot(settings.primary_symbol)

        features = self.pipe.build_features(df_btc, df_eth, df_funding).dropna()
        if features.empty:
            raise RuntimeError("El DataFrame de features quedó vacío tras dropna(). Verifica las ventanas rodantes.")

        last_row = features.iloc[[-1]]
        bar_ts = last_row.index[0]
        last_close = float(df_btc["close"].loc[bar_ts])
        last_atr_pct = float(last_row["atr_pct"].iloc[0])
        bull_regime = int(last_row["ctx_btc_bull_regime"].iloc[0]) == 1

        # Inferencia LONG
        p_long = self.brain_long.predict_calibrated_proba(last_row)
        ev_long_df = self.brain_long.compute_ev_and_kelly(p_long, last_row["atr_pct"].values).iloc[0]

        # Inferencia SHORT (si está habilitada y existe brain_short)
        best_side = "LONG"
        best_decision = ev_long_df
        ev_short_val = -999.0
        p_short_val = 0.0

        if settings.enable_shorts and self.brain_short is not None:
            p_short = self.brain_short.predict_calibrated_proba(last_row)
            ev_short_df = self.brain_short.compute_ev_and_kelly(p_short, last_row["atr_pct"].values).iloc[0]
            p_short_val = float(ev_short_df["p_tp_before_sl"])
            ev_short_val = float(ev_short_df["ev_in_r"])
            if ev_short_val > float(ev_long_df["ev_in_r"]):
                best_side = "SHORT"
                best_decision = ev_short_df

        print(f"Timestamp Vela : {bar_ts}")
        print(
            f"Precio BTC     : {last_close:.2f} USDT | Mark: {snap['mark_price']:.2f} | "
            f"Spread: {snap['ask_price'] - snap['bid_price']:.2f} USDT | ATR%: {last_atr_pct*100:.3f}%"
        )
        print(f"Contexto HTF   : {'BULLISH' if bull_regime else 'BEARISH'} | Funding: {snap['funding_rate']*100:.4f}%")
        print(
            f"Señal LONG     : P(TP)={ev_long_df['p_tp_before_sl']*100:.2f}% | "
            f"b_net={ev_long_df['b_net_odds']:.2f}R | EV={ev_long_df['ev_in_r']:+.3f}R"
        )
        if settings.enable_shorts and self.brain_short is not None:
            print(
                f"Señal SHORT    : P(TP)={p_short_val*100:.2f}% | "
                f"b_net={ev_long_df['b_net_odds']:.2f}R | EV={ev_short_val:+.3f}R"
            )

        summary = {
            "bar_timestamp": str(bar_ts),
            "close_price": last_close,
            "mark_price": snap["mark_price"],
            "atr_pct": last_atr_pct,
            "selected_side": best_side,
            "p_win": float(best_decision["p_tp_before_sl"]),
            "ev_in_r": float(best_decision["ev_in_r"]),
            "action": "PASS",
        }

        if self.execution_mode == "paper":
            sim_stats = self.simulator.summary_dict()
            print(
                f"Estado Paper   : Equity={sim_stats['equity_usdt']:.2f} USDT "
                f"({sim_stats['net_return_pct']:+.2f}%) | Trades={sim_stats['total_trades']} "
                f"(WR={sim_stats['win_rate_pct']:.1f}%) | Net R={sim_stats['cumulative_net_r']:+.2f}R"
            )

        # Verificamos el Circuit Breaker de Drawdown Diario
        if self.simulator.check_daily_circuit_breaker():
            print("Decisión       : BLOQUEADO POR CIRCUIT BREAKER DIARIO (Max Daily Drawdown alcanzado).")
            summary["action"] = "HALTED_DAILY_DRAWDOWN"
            return summary

        if float(best_decision["ev_in_r"]) < ev_threshold:
            print(f"Decisión       : PASAR (Mejor EV {best_decision['ev_in_r']:+.3f}R < {ev_threshold:+.2f}R).")
            return summary

        # Verificamos si ya existe una posición activa o si ya operamos esta misma vela
        if self.execution_mode == "paper":
            if self.simulator.has_open_position():
                pos = self.simulator.state["open_position"]
                print(
                    f"Decisión       : Posición {pos['side']} ya activa en Simulador "
                    f"(PnL flotante: {pos['unrealized_pnl_usdt']:+.2f} USDT / {pos['unrealized_r']:+.2f}R)."
                )
                summary["action"] = "POSITION_ALREADY_OPEN"
                return summary
            if self.simulator.state.get("last_evaluated_bar") == str(bar_ts):
                print("Decisión       : Vela ya operada previamente en este ciclo.")
                summary["action"] = "BAR_ALREADY_TRADED"
                return summary

            usdt_free = float(self.simulator.state["wallet_balance"])
            exec_ref_price = snap["ask_price"] if best_side == "LONG" else snap["bid_price"]
            sl_distance_usdt = exec_ref_price * last_atr_pct * self.brain.sl_mult

            qty_btc = self.compute_safe_position_size(
                available_balance_usdt=usdt_free,
                kelly_risk_fraction=float(best_decision["risk_fraction_kelly"]),
                entry_price=exec_ref_price,
                sl_distance_usdt=sl_distance_usdt,
            )
            if qty_btc <= 0:
                print("Decisión       : Tamaño de posición inferior al mínimo seguro permitido.")
                summary["action"] = "SIZE_TOO_SMALL"
                return summary

            pos = self.simulator.open_simulated_position(
                side=best_side,
                qty_btc=qty_btc,
                bid_price=snap["bid_price"],
                ask_price=snap["ask_price"],
                mark_price=snap["mark_price"],
                atr_pct=last_atr_pct,
                sl_mult=self.brain.sl_mult,
                tp_mult=self.brain.tp_mult,
                horizon_bars=self.pipe.horizon_bars,
                p_win=float(best_decision["p_tp_before_sl"]),
                ev_in_r=float(best_decision["ev_in_r"]),
                b_net_odds=float(best_decision["b_net_odds"]),
                kelly_risk_fraction=float(best_decision["risk_fraction_kelly"]),
                bar_timestamp=bar_ts,
            )
            self.simulator.state["last_evaluated_bar"] = str(bar_ts)
            self.simulator._save_state()

            print(
                f"EJECUTANDO {best_side} EN SIMULADOR TIEMPO REAL: {qty_btc} BTC @ {pos['entry_price']:.2f} USDT | "
                f"TP: {pos['tp_price']:.1f} | SL: {pos['sl_price']:.1f} | Riesgo 1R: {pos['risk_1r_usdt']:.2f} USDT"
            )
            summary["action"] = f"OPENED_{best_side}_PAPER"
            summary["position"] = pos
            return summary

        # Ejecución en Exchange Real (Testnet o Mainnet)
        assert self.execution_client is not None
        open_count = self.reconcile_exchange_state()
        if open_count > 0:
            print(f"Decisión       : Posición ya activa en {self.execution_mode.upper()}. Se omite nueva entrada.")
            summary["action"] = "POSITION_ALREADY_OPEN"
            return summary

        balance = self.execution_client.fetch_balance()
        usdt_free = float(balance.get("USDT", {}).get("free", 0.0))
        exec_ref_price = snap["ask_price"] if best_side == "LONG" else snap["bid_price"]
        sl_distance_usdt = exec_ref_price * last_atr_pct * self.brain.sl_mult

        qty_btc = self.compute_safe_position_size(
            available_balance_usdt=usdt_free,
            kelly_risk_fraction=float(best_decision["risk_fraction_kelly"]),
            entry_price=exec_ref_price,
            sl_distance_usdt=sl_distance_usdt,
        )
        if qty_btc <= 0:
            print("Decisión       : Balance insuficiente o lote mínimo excede el riesgo máximo permitido.")
            summary["action"] = "SIZE_TOO_SMALL"
            return summary

        print(f"EJECUTANDO {best_side} EN {self.execution_mode.upper()}: {qty_btc} BTC...")
        self._place_exchange_bracket_orders(best_side, qty_btc, last_atr_pct)
        print(f"Órdenes enviadas y confirmadas en Binance Futures ({self.execution_mode.upper()}).")
        summary["action"] = f"OPENED_{best_side}_{self.execution_mode.upper()}"
        return summary

    def run_realtime_replay_simulation(
        self,
        replay_bars: int = 250,
        ev_threshold: float = settings.ev_min_threshold,
    ) -> Dict[str, Any]:
        """
        Ejecuta una simulación de reproducción acelerada sobre las últimas `replay_bars` velas reales
        de 15m descargadas en vivo desde Binance, verificando todo el flujo de extremo a extremo:
        Features -> Probabilidad Calibrada -> EV -> Kelly -> Spread/Slippage -> Funding 8h -> SL/TP/Time-Stop.
        """
        print(
            f"\n================ SIMULACIÓN REAL-TIME REPLAY ({replay_bars} VELAS DE 15M) ================"
        )
        total_limit = min(1500, max(1000, replay_bars + 750))
        df_btc = self.fetch_live_dataframe(settings.primary_symbol, limit=total_limit)
        df_eth = self.fetch_live_dataframe(settings.context_symbol, limit=total_limit)
        df_funding = self.fetch_live_funding(settings.primary_symbol)

        features = self.pipe.build_features(df_btc, df_eth, df_funding).dropna()
        if len(features) <= replay_bars:
            replay_bars = max(10, len(features) - 10)

        eval_features = features.iloc[-replay_bars:]
        p_long_all = self.brain_long.predict_calibrated_proba(eval_features)
        ev_long_all = self.brain_long.compute_ev_and_kelly(p_long_all, eval_features["atr_pct"].values)
        ev_long_all.index = eval_features.index

        if settings.enable_shorts and self.brain_short is not None:
            p_short_all = self.brain_short.predict_calibrated_proba(eval_features)
            ev_short_all = self.brain_short.compute_ev_and_kelly(p_short_all, eval_features["atr_pct"].values)
            ev_short_all.index = eval_features.index
        else:
            ev_short_all = None

        for bar_ts, row in eval_features.iterrows():
            bar_dt = bar_ts.to_pydatetime()
            close_p = float(df_btc.loc[bar_ts, "close"])
            high_p = float(df_btc.loc[bar_ts, "high"])
            low_p = float(df_btc.loc[bar_ts, "low"])
            atr_pct = float(row["atr_pct"])
            f_rate = float(row["deriv_funding_rate"])

            # Estimamos Bid/Ask orgánico (~1 bp spread en BTCUSDT)
            half_spread = close_p * 0.00005
            bid_p = round(close_p - half_spread, 2)
            ask_p = round(close_p + half_spread, 2)

            # 1. Si hay posición abierta, verificamos si la vela actual tocó SL, TP, Funding o Time-Stop
            if self.simulator.has_open_position():
                closed = self.simulator.update_on_market_tick(
                    mark_price=close_p,
                    bid_price=bid_p,
                    ask_price=ask_p,
                    funding_rate=f_rate,
                    candle_high=high_p,
                    candle_low=low_p,
                    now_utc=bar_dt,
                )
                if closed is not None:
                    print(
                        f"[{bar_ts}] Cierre #{closed['trade_id']} ({closed['side']}) por {closed['exit_reason']:12s} | "
                        f"Entrada: {closed['entry_price']:.1f} -> Salida: {closed['exit_price']:.1f} | "
                        f"Neto: {closed['net_pnl_usdt']:+7.2f} USDT ({closed['realized_net_r']:+5.2f}R) | "
                        f"Equity: {closed['balance_after_usdt']:.2f}"
                    )

            # 2. Si estamos planos y no bloqueados por circuit breaker, evaluamos nueva entrada al cierre de la vela
            if not self.simulator.has_open_position() and not self.simulator.check_daily_circuit_breaker(bar_dt):
                ev_l = ev_long_all.loc[bar_ts]
                best_side = "LONG"
                best_ev = ev_l

                if ev_short_all is not None:
                    ev_s = ev_short_all.loc[bar_ts]
                    if float(ev_s["ev_in_r"]) > float(ev_l["ev_in_r"]):
                        best_side = "SHORT"
                        best_ev = ev_s

                if float(best_ev["ev_in_r"]) >= ev_threshold:
                    usdt_free = float(self.simulator.state["wallet_balance"])
                    ref_p = ask_p if best_side == "LONG" else bid_p
                    sl_dist = ref_p * atr_pct * self.brain.sl_mult
                    qty = self.compute_safe_position_size(
                        available_balance_usdt=usdt_free,
                        kelly_risk_fraction=float(best_ev["risk_fraction_kelly"]),
                        entry_price=ref_p,
                        sl_distance_usdt=sl_dist,
                    )
                    if qty > 0:
                        pos = self.simulator.open_simulated_position(
                            side=best_side,
                            qty_btc=qty,
                            bid_price=bid_p,
                            ask_price=ask_p,
                            mark_price=close_p,
                            atr_pct=atr_pct,
                            sl_mult=self.brain.sl_mult,
                            tp_mult=self.brain.tp_mult,
                            horizon_bars=self.pipe.horizon_bars,
                            p_win=float(best_ev["p_tp_before_sl"]),
                            ev_in_r=float(best_ev["ev_in_r"]),
                            b_net_odds=float(best_ev["b_net_odds"]),
                            kelly_risk_fraction=float(best_ev["risk_fraction_kelly"]),
                            bar_timestamp=bar_ts,
                            now_utc=bar_dt,
                        )
                        print(
                            f"[{bar_ts}] Apertura #{pos['trade_id']} ({best_side:5s}) | "
                            f"{qty:.3f} BTC @ {pos['entry_price']:.1f} | "
                            f"TP: {pos['tp_price']:.1f} | SL: {pos['sl_price']:.1f} | "
                            f"P(TP)={pos['p_win']*100:.1f}% | EV={pos['ev_in_r']:+.3f}R"
                        )

        stats = self.simulator.summary_dict()
        print("------------------------------------------------------------------------------------------")
        print(f"Resumen Replay : {stats}")
        print("==========================================================================================\n")
        return stats

    def run_forever(self, ev_threshold: float = settings.ev_min_threshold) -> None:
        """
        Bucle principal sincronizado con el reloj UTC:
          - Monitorea posiciones abiertas y barreras cada `poll_interval_seconds` (15s).
          - Ejecuta la inferencia de cierre de vela exactamente al cruzar cada múltiplo de 15 minutos
            (`00:00`, `00:15`, `00:30`, `00:45` UTC + 2s de margen de consolidación).
        """
        print(
            f"[INICIO] Bot Cuantitativo activo en modo '{self.execution_mode.upper()}' "
            f"(Umbral EV={ev_threshold:+.2f}R, Poll={settings.poll_interval_seconds}s)"
        )
        last_evaluated_slot: Optional[int] = None

        while True:
            try:
                now_ts = time.time()
                current_15m_slot = int(now_ts // 900) * 900

                # Evaluamos al inicio o al cruzar un nuevo cierre de vela de 15 minutos (+2s buffer)
                if last_evaluated_slot is None or (
                    current_15m_slot > last_evaluated_slot and (now_ts - current_15m_slot) >= 2.0
                ):
                    self.evaluate_and_execute(ev_threshold=ev_threshold)
                    last_evaluated_slot = current_15m_slot
                else:
                    self.monitor_realtime_positions()
            except Exception as exc:
                print(f"[ERROR EN CICLO]: {exc}")

            time.sleep(settings.poll_interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Binance Quant Lab - Real-Time Simulator & Execution Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "testnet", "mainnet"],
        default=None,
        help="Modo de ejecución: paper (simulación real-time), testnet o mainnet.",
    )
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=settings.ev_min_threshold,
        help="Umbral mínimo de Valor Esperado en unidades R (por defecto: EV_MIN_THRESHOLD).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecuta una única evaluación en tiempo real sobre la última vela cerrada y termina.",
    )
    parser.add_argument(
        "--replay-bars",
        type=int,
        default=0,
        help="Reproduce N velas reales de 15m en el simulador para validar el ciclo completo de trading.",
    )
    args = parser.parse_args()

    bot = BinanceFuturesTestnetBot(execution_mode=args.mode)

    if args.replay_bars > 0:
        bot.run_realtime_replay_simulation(replay_bars=args.replay_bars, ev_threshold=args.ev_threshold)
    elif args.once:
        bot.evaluate_and_execute(ev_threshold=args.ev_threshold)
    else:
        bot.run_forever(ev_threshold=args.ev_threshold)


if __name__ == "__main__":
    main()
