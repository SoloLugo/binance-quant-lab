import pytest

from config import QuantConfig
from live_testnet_bot import BinanceFuturesTestnetBot
from simulator import RealTimeFuturesSimulator


def test_quant_config_safe_defaults():
    """Verifica que QuantConfig arranque con los valores seguros por defecto."""
    cfg = QuantConfig()
    assert cfg.execution_mode == "paper"
    assert cfg.use_testnet is True
    assert cfg.dry_run is True
    assert cfg.allow_mainnet is False


def test_quant_config_blocks_mainnet_execution_mode():
    """QuantConfig(execution_mode='mainnet') debe lanzar PermissionError."""
    with pytest.raises(PermissionError):
        QuantConfig(execution_mode="mainnet")


def test_quant_config_blocks_use_testnet_false():
    """QuantConfig(use_testnet=False) debe lanzar PermissionError."""
    with pytest.raises(PermissionError):
        QuantConfig(use_testnet=False)


def test_quant_config_blocks_allow_mainnet_true():
    """QuantConfig(allow_mainnet=True) debe lanzar PermissionError."""
    with pytest.raises(PermissionError):
        QuantConfig(allow_mainnet=True)


@pytest.mark.parametrize(
    "env_key,env_val",
    [
        ("EXECUTION_MODE", "mainnet"),
        ("USE_TESTNET", "false"),
        ("ALLOW_MAINNET", "true"),
    ],
)
def test_quant_config_from_env_blocks_mainnet(monkeypatch, env_key, env_val):
    """QuantConfig.from_env() debe lanzar PermissionError ante cualquier intento de habilitar Mainnet."""
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("USE_TESTNET", "true")
    monkeypatch.setenv("ALLOW_MAINNET", "false")
    monkeypatch.setenv(env_key, env_val)

    with pytest.raises(PermissionError):
        QuantConfig.from_env()


def test_bot_blocks_mainnet_execution_mode():
    """BinanceFuturesTestnetBot(execution_mode='mainnet') debe lanzar PermissionError."""
    with pytest.raises(PermissionError):
        BinanceFuturesTestnetBot(execution_mode="mainnet")


def test_simulator_fresh_state_ignores_previous_saved_state(tmp_path):
    """RealTimeFuturesSimulator(..., fresh_state=True) no debe reutilizar estado previo en el mismo state_path."""
    state_path = str(tmp_path / "sim_state.json")
    trade_log_path = str(tmp_path / "trades.csv")

    sim1 = RealTimeFuturesSimulator(
        state_path=state_path,
        trade_log_path=trade_log_path,
        initial_balance=10000.0,
    )
    sim1.state["equity"] = 8123.45
    sim1.state["wallet_balance"] = 8123.45
    sim1.state["total_trades"] = 7
    sim1.state["daily_halted"] = True
    sim1._save_state()

    # Confirmamos que sin fresh_state sí se recarga el estado persistido
    sim_reloaded = RealTimeFuturesSimulator(
        state_path=state_path,
        trade_log_path=trade_log_path,
        initial_balance=10000.0,
        fresh_state=False,
    )
    assert sim_reloaded.state["equity"] == pytest.approx(8123.45)
    assert sim_reloaded.state["total_trades"] == 7
    assert sim_reloaded.state["daily_halted"] is True

    # Con fresh_state=True debe reiniciar completamente a los valores iniciales
    sim_fresh = RealTimeFuturesSimulator(
        state_path=state_path,
        trade_log_path=trade_log_path,
        initial_balance=10000.0,
        fresh_state=True,
    )
    assert sim_fresh.state["equity"] == pytest.approx(10000.0)
    assert sim_fresh.state["wallet_balance"] == pytest.approx(10000.0)
    assert sim_fresh.state["total_trades"] == 0
    assert sim_fresh.state["daily_halted"] is False
    assert sim_fresh.state["open_position"] is None
