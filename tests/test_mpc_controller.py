"""Tests for the MPC controller."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.roommind.mpc_controller import MPCController
from custom_components.roommind.thermal_model import RoomModelManager, RCModel


def build_hass():
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states.get = MagicMock(return_value=None)
    return hass


def make_room(**overrides):
    room = {
        "area_id": "living_room",
        "thermostats": ["climate.living_trv"],
        "acs": [],
        "climate_mode": "auto",
        "temperature_sensor": "sensor.living_temp",
        "schedules": [],
    }
    room.update(overrides)
    return room


@pytest.mark.asyncio
async def test_mpc_evaluate_heats_when_cold():
    """Cold room triggers heating."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode == "heating"
    assert 0.0 < pf <= 1.0


@pytest.mark.asyncio
async def test_mpc_evaluate_idle_at_target():
    """At target, returns idle."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=21.0, target_temp=21.0)
    assert mode == "idle"
    assert pf == 0.0


@pytest.mark.asyncio
async def test_mpc_fallback_to_bangbang():
    """Low confidence = bang-bang: 0.1°C below target, within 0.2°C hysteresis → idle."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=20.9, target_temp=21.0)
    assert mode == "idle"
    assert pf == 0.0


@pytest.mark.asyncio
async def test_mpc_apply_heating():
    """Apply heating calls climate services."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0)
    assert hass.services.async_call.called


@pytest.mark.asyncio
async def test_mpc_managed_mode():
    """Managed mode: device self-regulates, returns heating when thermostats present."""
    hass = build_hass()
    room = make_room(temperature_sensor="")
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=False,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=None, target_temp=21.0)
    assert mode == "heating"
    assert pf == 1.0  # managed mode: device self-regulates


@pytest.mark.asyncio
async def test_mpc_outdoor_gating():
    """Cooling blocked when outdoor below threshold."""
    hass = build_hass()
    room = make_room(thermostats=[], acs=["climate.ac"])
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=10.0, settings={"outdoor_cooling_min": 16.0},
        has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=25.0, target_temp=22.0)
    assert mode == "idle"
    assert pf == 0.0


@pytest.mark.asyncio
async def test_mpc_apply_cooling():
    """Apply cooling calls climate services on ACs."""
    hass = build_hass()
    room = make_room(thermostats=[], acs=["climate.ac"])
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=30.0, settings={}, has_external_sensor=True,
    )
    await ctrl.async_apply("cooling", 23.0)
    assert hass.services.async_call.called


@pytest.mark.asyncio
async def test_mpc_apply_idle():
    """Apply idle turns off everything."""
    hass = build_hass()
    room = make_room(acs=["climate.ac"])
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    await ctrl.async_apply("idle", 21.0)
    assert hass.services.async_call.called


@pytest.mark.asyncio
async def test_mpc_path_when_confident():
    """When model confidence is high, MPC optimizer is used instead of bang-bang."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    # Pre-train to get a valid model
    model_mgr.update("living_room", 18.5, 5.0, "heating", 5.0)
    model_mgr.update("living_room", 19.0, 5.0, "heating", 5.0)
    # Mock prediction_std to be low (confident) + enough training data
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode == "heating"
    assert 0.0 < pf <= 1.0
    model_mgr.get_prediction_std.assert_called_once()


@pytest.mark.asyncio
async def test_confidence_transition_threshold():
    """pred_std >= 0.5 -> bang-bang, pred_std < 0.5 -> MPC."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    model_mgr.update("living_room", 18.5, 5.0, "heating", 5.0)
    model_mgr.update("living_room", 19.0, 5.0, "heating", 5.0)

    # Enough training data for MPC
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))

    # Just above threshold — bang-bang
    model_mgr.get_prediction_std = MagicMock(return_value=0.5)
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode == "heating"  # bang-bang also heats when cold
    assert pf == 1.0  # bang-bang: full power

    # Just below threshold — MPC path
    model_mgr.get_prediction_std = MagicMock(return_value=0.49)
    ctrl2 = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode2, pf2 = await ctrl2.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode2 == "heating"  # MPC also heats when cold
    assert 0.0 < pf2 <= 1.0


@pytest.mark.asyncio
async def test_mpc_requires_min_updates():
    """MPC falls back to bang-bang when not enough training data per mode."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()

    # Low pred_std (model thinks it's confident) but not enough samples
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)

    # Too few idle samples → bang-bang
    model_mgr.get_mode_counts = MagicMock(return_value=(30, 25, 0))
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=20.9, target_temp=21.0)
    assert mode == "idle"  # bang-bang: within hysteresis
    assert pf == 0.0

    # Enough idle but too few heating samples → bang-bang
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 10, 0))
    ctrl2 = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode2, pf2 = await ctrl2.async_evaluate(current_temp=20.9, target_temp=21.0)
    assert mode2 == "idle"  # still bang-bang
    assert pf2 == 0.0

    # Enough data → MPC (would heat at 20.9 because optimizer predicts drop)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
    ctrl3 = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode3, pf3 = await ctrl3.async_evaluate(current_temp=20.9, target_temp=21.0)
    assert mode3 == "heating"  # MPC: optimizer decides to heat proactively
    assert 0.0 < pf3 <= 1.0



# ---------------------------------------------------------------------------
# T4: _compute_horizon_blocks unit tests
# ---------------------------------------------------------------------------

class TestComputeHorizonBlocks:
    """Unit tests for MPCController._compute_horizon_blocks."""

    def _make_ctrl(self, **room_overrides):
        hass = build_hass()
        room = make_room(**room_overrides)
        model_mgr = RoomModelManager()
        return MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=5.0, settings={}, has_external_sensor=True,
        )

    def test_small_delta_returns_minimum_horizon(self):
        """Small temp delta should still produce at least MIN_HORIZON_HOURS worth of blocks."""
        from custom_components.roommind.mpc_controller import MIN_HORIZON_HOURS, PLAN_DT_MINUTES
        ctrl = self._make_ctrl()
        model = ctrl._model_manager.get_model("living_room")
        blocks = ctrl._compute_horizon_blocks(model, 20.5, 21.0)
        min_blocks = int(MIN_HORIZON_HOURS * 60 / PLAN_DT_MINUTES)
        assert blocks >= min_blocks

    def test_large_delta_increases_horizon(self):
        """Larger delta between current and target should increase horizon blocks."""
        ctrl = self._make_ctrl()
        model = ctrl._model_manager.get_model("living_room")
        blocks_small = ctrl._compute_horizon_blocks(model, 20.0, 21.0)
        blocks_large = ctrl._compute_horizon_blocks(model, 10.0, 21.0)
        assert blocks_large >= blocks_small

    def test_returns_at_least_24_blocks(self):
        """Result should always be at least 24 blocks."""
        ctrl = self._make_ctrl()
        model = ctrl._model_manager.get_model("living_room")
        blocks = ctrl._compute_horizon_blocks(model, 21.0, 21.0)
        assert blocks >= 24

    def test_zero_Q_max_returns_default(self):
        """When Q_heat and Q_cool are both 0, fall back to default horizon."""
        from custom_components.roommind.mpc_controller import MIN_HORIZON_HOURS, PLAN_DT_MINUTES
        from custom_components.roommind.thermal_model import RCModel
        ctrl = self._make_ctrl()
        model = RCModel(C=2.0, U=50.0, Q_heat=0.0, Q_cool=0.0)
        blocks = ctrl._compute_horizon_blocks(model, 15.0, 21.0)
        assert blocks == int(MIN_HORIZON_HOURS * 60 / PLAN_DT_MINUTES)

    def test_high_power_model_shorter_horizon(self):
        """High HVAC power should yield fewer blocks (faster to reach target)."""
        from custom_components.roommind.thermal_model import RCModel
        ctrl = self._make_ctrl()
        model_low = RCModel(C=2.0, U=50.0, Q_heat=400.0, Q_cool=400.0)
        model_high = RCModel(C=2.0, U=50.0, Q_heat=4000.0, Q_cool=4000.0)
        blocks_low = ctrl._compute_horizon_blocks(model_low, 15.0, 21.0)
        blocks_high = ctrl._compute_horizon_blocks(model_high, 15.0, 21.0)
        assert blocks_high <= blocks_low

    def test_high_thermal_mass_longer_horizon(self):
        """High thermal capacitance should yield more blocks (slower temperature change)."""
        from custom_components.roommind.thermal_model import RCModel
        ctrl = self._make_ctrl()
        model_small_c = RCModel(C=1.0, U=50.0, Q_heat=800.0, Q_cool=800.0)
        model_large_c = RCModel(C=10.0, U=50.0, Q_heat=800.0, Q_cool=800.0)
        blocks_small = ctrl._compute_horizon_blocks(model_small_c, 15.0, 21.0)
        blocks_large = ctrl._compute_horizon_blocks(model_large_c, 15.0, 21.0)
        assert blocks_large >= blocks_small


# ---------------------------------------------------------------------------
# T4: _build_outdoor_series unit tests
# ---------------------------------------------------------------------------

class TestBuildOutdoorSeries:
    """Unit tests for MPCController._build_outdoor_series."""

    def test_constant_outdoor_no_forecast(self):
        """Without forecast, returns constant outdoor temp repeated n_blocks times."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=8.0, settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(10)
        assert series == [8.0] * 10

    def test_fallback_when_outdoor_temp_none_no_forecast(self):
        """Without forecast and outdoor_temp=None, uses DEFAULT_OUTDOOR_TEMP_FALLBACK."""
        from custom_components.roommind.mpc_controller import DEFAULT_OUTDOOR_TEMP_FALLBACK
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=None, settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(5)
        assert series == [DEFAULT_OUTDOOR_TEMP_FALLBACK] * 5

    def test_forecast_used_when_available(self):
        """With forecast data, series uses forecast temperatures."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
            {"temperature": 7.0},
        ]
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=8.0, outdoor_forecast=forecast,
            settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        assert series == [5.0, 6.0, 7.0]

    def test_forecast_padded_when_shorter_than_n_blocks(self):
        """Forecast shorter than n_blocks should be padded with last forecast value."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
        ]
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=8.0, outdoor_forecast=forecast,
            settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(5)
        assert series == [5.0, 6.0, 6.0, 6.0, 6.0]

    def test_forecast_truncated_when_longer_than_n_blocks(self):
        """Forecast longer than n_blocks should be truncated."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
            {"temperature": 7.0},
            {"temperature": 8.0},
            {"temperature": 9.0},
        ]
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=10.0, outdoor_forecast=forecast,
            settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        assert series == [5.0, 6.0, 7.0]

    def test_forecast_missing_temperature_key_uses_outdoor_temp(self):
        """Forecast entries without 'temperature' key fall back to current outdoor_temp."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"condition": "cloudy"},  # no temperature key
            {"temperature": 7.0},
        ]
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=8.0, outdoor_forecast=forecast,
            settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        assert series == [5.0, 8.0, 7.0]

    def test_forecast_missing_temp_key_and_outdoor_none_uses_fallback(self):
        """Forecast entry without temp + outdoor_temp=None uses DEFAULT_OUTDOOR_TEMP_FALLBACK."""
        from custom_components.roommind.mpc_controller import DEFAULT_OUTDOOR_TEMP_FALLBACK
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"condition": "cloudy"},  # no temperature key
        ]
        ctrl = MPCController(
            hass, room, model_manager=model_mgr,
            outdoor_temp=None, outdoor_forecast=forecast,
            settings={}, has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        assert series[0] == DEFAULT_OUTDOOR_TEMP_FALLBACK
        # Padding should also use the fallback
        assert all(v == DEFAULT_OUTDOOR_TEMP_FALLBACK for v in series)


# ---------------------------------------------------------------------------
# Proportional control tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proportional_power_far_from_target():
    """MPC mode, large error → power_fraction near 1.0."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    model_mgr.update("living_room", 15.0, 5.0, "heating", 5.0)
    model_mgr.update("living_room", 16.0, 5.0, "heating", 5.0)
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=15.0, target_temp=21.0)
    assert mode == "heating"
    assert pf >= 0.7  # large error → high power


@pytest.mark.asyncio
async def test_proportional_power_near_target():
    """MPC mode, small error → reduced power_fraction."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    # Use a known model with high Q_heat so a small 0.3°C error yields frac < 1.
    # This tests MPC proportional behavior, not EKF learning.
    model_mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=50.0, Q_cool=75.0))
    model_mgr.get_prediction_std = MagicMock(return_value=0.1)
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 40, 0))
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=20.7, target_temp=21.0)
    if mode == "heating":
        assert pf < 1.0  # near target → less than full power


@pytest.mark.asyncio
async def test_proportional_trv_setpoint():
    """TRV setpoint is proportional between current_temp and 30°C."""
    from custom_components.roommind.mpc_controller import HEATING_BOOST_TARGET
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    # 50% power at 20°C → TRV = 20 + 0.5*(30-20) = 25°C
    await ctrl.async_apply("heating", 21.0, power_fraction=0.5, current_temp=20.0)
    calls = hass.services.async_call.call_args_list
    set_temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert set_temp_calls
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == 25.0


@pytest.mark.asyncio
async def test_bangbang_returns_full_power():
    """Bang-bang fallback → power_fraction = 1.0 for heating."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    # Large error in bang-bang mode (low confidence)
    mode, pf = await ctrl.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode == "heating"
    assert pf == 1.0  # bang-bang: always full power


@pytest.mark.asyncio
async def test_async_apply_backward_compat():
    """Calling async_apply without power_fraction uses default 1.0 → 30°C boost."""
    from custom_components.roommind.mpc_controller import HEATING_BOOST_TARGET
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0)  # no power_fraction → default 1.0
    calls = hass.services.async_call.call_args_list
    set_temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert set_temp_calls
    # Without current_temp, falls back to HEATING_BOOST_TARGET
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == HEATING_BOOST_TARGET


@pytest.mark.asyncio
async def test_mpc_apply_heating_fahrenheit():
    """set_temperature uses Fahrenheit when HA is configured for °F."""
    from homeassistant.const import UnitOfTemperature
    from custom_components.roommind.mpc_controller import HEATING_BOOST_TARGET

    hass = build_hass()
    hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT

    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    await ctrl.async_apply("heating", 21.0)

    calls = hass.services.async_call.call_args_list
    set_temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert set_temp_calls

    # HEATING_BOOST_TARGET (30°C) → 86°F
    expected_f = HEATING_BOOST_TARGET * 9 / 5 + 32
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == pytest.approx(expected_f)


@pytest.mark.asyncio
async def test_mpc_apply_cooling_fahrenheit():
    """Cooling set_temperature uses Fahrenheit when HA is configured for °F."""
    from homeassistant.const import UnitOfTemperature

    hass = build_hass()
    hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT

    room = make_room(thermostats=[], acs=["climate.ac"])
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=30.0, settings={}, has_external_sensor=True,
    )
    # Apply cooling with target 23°C
    await ctrl.async_apply("cooling", 23.0)

    calls = hass.services.async_call.call_args_list
    set_temp_calls = [c for c in calls if c[0][1] == "set_temperature"]
    assert set_temp_calls

    # 23°C → 73.4°F
    expected_f = 23.0 * 9 / 5 + 32
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == pytest.approx(expected_f)


# ---------------------------------------------------------------------------
# get_can_heat_cool unit tests
# ---------------------------------------------------------------------------

class TestGetCanHeatCool:
    """Unit tests for get_can_heat_cool."""

    def test_auto_mode_with_both_devices(self):
        """auto mode with thermostats and ACs → (True, True)."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="auto", acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is True
        assert can_cool is True

    def test_heat_only_mode(self):
        """heat_only mode → (True, False) regardless of ACs."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="heat_only", acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is True
        assert can_cool is False

    def test_cool_only_mode(self):
        """cool_only mode → (False, True) regardless of thermostats."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="cool_only", acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is False
        assert can_cool is True

    def test_no_thermostats_heat_only(self):
        """heat_only but no thermostats → (False, False)."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="heat_only", thermostats=[], acs=[])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is False
        assert can_cool is False

    def test_no_acs_cool_only(self):
        """cool_only but no ACs → (False, False)."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="cool_only", thermostats=[], acs=[])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is False
        assert can_cool is False

    def test_outdoor_temp_none_no_gating(self):
        """outdoor_temp=None → no gating applied."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(room, outdoor_temp=None)
        assert can_heat is True
        assert can_cool is True

    def test_outdoor_above_heating_max_blocks_heat(self):
        """Outdoor temp above outdoor_heating_max → can_heat=False."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(
            room, outdoor_temp=25.0, outdoor_heating_max=22.0,
        )
        assert can_heat is False
        assert can_cool is True

    def test_outdoor_below_cooling_min_blocks_cool(self):
        """Outdoor temp below outdoor_cooling_min → can_cool=False."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(acs=["climate.ac"])
        can_heat, can_cool = get_can_heat_cool(
            room, outdoor_temp=10.0, outdoor_cooling_min=16.0,
        )
        assert can_heat is True
        assert can_cool is False

    def test_outdoor_at_threshold_boundary(self):
        """Outdoor temp equal to heating_max → not blocked (>= not used, > is)."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(acs=["climate.ac"])
        # At exactly outdoor_heating_max=22.0 → 22 > 22 is False, so still allowed
        can_heat, can_cool = get_can_heat_cool(
            room, outdoor_temp=22.0, outdoor_heating_max=22.0,
        )
        assert can_heat is True
        # At exactly outdoor_cooling_min=16.0 → 16 < 16 is False, so still allowed
        can_heat2, can_cool2 = get_can_heat_cool(
            room, outdoor_temp=16.0, outdoor_cooling_min=16.0,
        )
        assert can_cool2 is True

    def test_auto_no_devices_returns_false_false(self):
        """auto mode but no devices → (False, False)."""
        from custom_components.roommind.mpc_controller import get_can_heat_cool
        room = make_room(climate_mode="auto", thermostats=[], acs=[])
        can_heat, can_cool = get_can_heat_cool(room)
        assert can_heat is False
        assert can_cool is False


# ---------------------------------------------------------------------------
# is_mpc_active unit tests
# ---------------------------------------------------------------------------

class TestIsMpcActive:
    """Unit tests for is_mpc_active."""

    def test_area_not_in_estimators(self):
        """Returns False when area_id has no estimator."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        result = is_mpc_active(model_mgr, "unknown_room", True, False, 20.0, 10.0)
        assert result is False

    def test_prediction_std_too_high(self):
        """Returns False when prediction_std >= MPC_MAX_PREDICTION_STD."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.6)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 30))
        result = is_mpc_active(model_mgr, "living_room", True, False, 20.0, 10.0)
        assert result is False

    def test_insufficient_idle_samples(self):
        """Returns False when idle samples below MIN_IDLE_UPDATES."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        model_mgr.get_mode_counts = MagicMock(return_value=(30, 30, 30))  # idle < 60
        result = is_mpc_active(model_mgr, "living_room", True, False, 20.0, 10.0)
        assert result is False

    def test_insufficient_heating_samples(self):
        """Returns False when can_heat but heating samples below MIN_ACTIVE_UPDATES."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 10, 0))  # heating < 20
        result = is_mpc_active(model_mgr, "living_room", True, False, 20.0, 10.0)
        assert result is False

    def test_insufficient_cooling_samples(self):
        """Returns False when can_cool but cooling samples below MIN_ACTIVE_UPDATES."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 0, 10))  # cooling < 20
        result = is_mpc_active(model_mgr, "living_room", False, True, 20.0, 10.0)
        assert result is False

    def test_all_conditions_met_returns_true(self):
        """Returns True when all conditions satisfied."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
        result = is_mpc_active(model_mgr, "living_room", True, False, 20.0, 10.0)
        assert result is True

    def test_heat_and_cool_both_need_samples(self):
        """When both can_heat and can_cool, both need MIN_ACTIVE_UPDATES."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        # Enough heating but not enough cooling
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 10))
        result = is_mpc_active(model_mgr, "living_room", True, True, 20.0, 10.0)
        assert result is False

    def test_no_heat_no_cool_only_idle_needed(self):
        """When neither can_heat nor can_cool, only idle check matters."""
        from custom_components.roommind.mpc_controller import is_mpc_active
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=0.1)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 0, 0))
        result = is_mpc_active(model_mgr, "living_room", False, False, 20.0, 10.0)
        assert result is True

    def test_prediction_std_at_threshold(self):
        """pred_std exactly at MPC_MAX_PREDICTION_STD (0.5) → False."""
        from custom_components.roommind.mpc_controller import is_mpc_active, MPC_MAX_PREDICTION_STD
        model_mgr = RoomModelManager()
        model_mgr.update("living_room", 20.0, 10.0, "idle", 5.0)
        model_mgr.get_prediction_std = MagicMock(return_value=MPC_MAX_PREDICTION_STD)
        model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
        result = is_mpc_active(model_mgr, "living_room", True, False, 20.0, 10.0)
        assert result is False


# ---------------------------------------------------------------------------
# Device min/max temperature clamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_clamps_to_device_max_temp():
    """Temperature is clamped to device max_temp attribute."""
    hass = build_hass()
    mock_state = MagicMock()
    mock_state.state = "off"
    mock_state.attributes = {"min_temp": 5.0, "max_temp": 25.0, "temperature": None}
    hass.states.get = MagicMock(return_value=mock_state)

    room = make_room()
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=5.0, settings={}, has_external_sensor=True,
    )
    # Heating with full power tries to set 30°C (HEATING_BOOST_TARGET)
    await ctrl.async_apply("heating", 21.0, power_fraction=1.0, current_temp=18.0)

    set_temp_calls = [
        c for c in hass.services.async_call.call_args_list
        if c[0][1] == "set_temperature"
    ]
    assert set_temp_calls
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == 25.0  # clamped to device max


@pytest.mark.asyncio
async def test_apply_clamps_to_device_min_temp():
    """Temperature is clamped to device min_temp attribute."""
    hass = build_hass()
    mock_state = MagicMock()
    mock_state.state = "off"
    mock_state.attributes = {"min_temp": 10.0, "max_temp": 30.0, "temperature": None}
    hass.states.get = MagicMock(return_value=mock_state)

    room = make_room(thermostats=[], acs=["climate.ac"])
    model_mgr = RoomModelManager()
    ctrl = MPCController(
        hass, room, model_manager=model_mgr,
        outdoor_temp=35.0, settings={}, has_external_sensor=True,
    )
    # Cooling with target below device min
    await ctrl.async_apply("cooling", 8.0)

    set_temp_calls = [
        c for c in hass.services.async_call.call_args_list
        if c[0][1] == "set_temperature"
    ]
    assert set_temp_calls
    temp_arg = set_temp_calls[0][0][2]["temperature"]
    assert temp_arg == 10.0  # clamped to device min
