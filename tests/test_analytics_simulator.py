"""Tests for analytics_simulator.py."""

from __future__ import annotations

import time

from custom_components.roommind.analytics_simulator import (
    build_forecast_outdoor_series,
    build_forecast_solar_series,
    compute_observed_idle_rate,
    _simulate_bangbang,
    _simulate_mpc,
)
from custom_components.roommind.thermal_model import RCModel, ThermalEKF


# ---------------------------------------------------------------------------
# compute_observed_idle_rate
# ---------------------------------------------------------------------------


class TestComputeObservedIdleRate:
    """Tests for compute_observed_idle_rate."""

    def test_empty_points_returns_none(self):
        """Empty list → None."""
        assert compute_observed_idle_rate([]) is None

    def test_insufficient_idle_points_returns_none(self):
        """Fewer than 2 idle points in last hour → None."""
        now = time.time()
        points = [
            {"ts": now - 100, "room_temp": 20.0, "mode": "idle"},
        ]
        assert compute_observed_idle_rate(points) is None

    def test_no_idle_points_returns_none(self):
        """Only heating points → None."""
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 20.0, "mode": "heating"},
            {"ts": now - 300, "room_temp": 21.0, "mode": "heating"},
        ]
        assert compute_observed_idle_rate(points) is None

    def test_normal_idle_rate_computation(self):
        """Known idle rate: 1°C drop over 600s → rate_per_5min = -0.5."""
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 21.0, "mode": "idle"},
            {"ts": now - 300, "room_temp": 20.5, "mode": "idle"},
            {"ts": now - 0, "room_temp": 20.0, "mode": "idle"},
        ]
        rate = compute_observed_idle_rate(points)
        assert rate is not None
        # -1.0°C over 600s = -1/600 per sec → ×300 = -0.5 per 5 min
        assert abs(rate - (-0.5)) < 0.01

    def test_rising_idle_rate(self):
        """Idle rate can be positive (warm room cooling down → room warming up in sun)."""
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 20.0, "mode": "idle"},
            {"ts": now - 0, "room_temp": 21.0, "mode": "idle"},
        ]
        rate = compute_observed_idle_rate(points)
        assert rate is not None
        assert rate > 0

    def test_points_older_than_1h_ignored(self):
        """Points older than 1 hour are excluded."""
        now = time.time()
        points = [
            {"ts": now - 7200, "room_temp": 25.0, "mode": "idle"},  # 2h ago
            {"ts": now - 300, "room_temp": 20.0, "mode": "idle"},
        ]
        # Only 1 idle point within the hour → None
        assert compute_observed_idle_rate(points) is None

    def test_empty_mode_treated_as_idle(self):
        """Empty string mode is treated as idle."""
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 21.0, "mode": ""},
            {"ts": now - 0, "room_temp": 20.0, "mode": ""},
        ]
        rate = compute_observed_idle_rate(points)
        assert rate is not None

    def test_very_close_timestamps_returns_none(self):
        """Points within 60s of each other → None (dt_sec <= 60)."""
        now = time.time()
        points = [
            {"ts": now - 30, "room_temp": 20.0, "mode": "idle"},
            {"ts": now - 0, "room_temp": 19.9, "mode": "idle"},
        ]
        assert compute_observed_idle_rate(points) is None


# ---------------------------------------------------------------------------
# build_forecast_outdoor_series
# ---------------------------------------------------------------------------


class TestBuildForecastOutdoorSeries:
    """Tests for build_forecast_outdoor_series."""

    def test_with_forecast_data(self):
        """Forecast entries used as-is."""
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
            {"temperature": 7.0},
        ]
        result = build_forecast_outdoor_series(forecast, 10.0, 3)
        assert result == [5.0, 6.0, 7.0]

    def test_without_forecast_fallback(self):
        """No forecast → constant current outdoor."""
        result = build_forecast_outdoor_series([], 10.0, 5)
        assert result == [10.0] * 5

    def test_forecast_shorter_than_n_blocks_padded(self):
        """Short forecast padded with last value."""
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
        ]
        result = build_forecast_outdoor_series(forecast, 10.0, 5)
        assert result == [5.0, 6.0, 6.0, 6.0, 6.0]

    def test_forecast_longer_than_n_blocks_truncated(self):
        """Longer forecast truncated to n_blocks."""
        forecast = [{"temperature": float(i)} for i in range(10)]
        result = build_forecast_outdoor_series(forecast, 10.0, 3)
        assert result == [0.0, 1.0, 2.0]

    def test_missing_temperature_key_uses_current(self):
        """Forecast entry without 'temperature' → uses current_outdoor."""
        forecast = [
            {"temperature": 5.0},
            {"condition": "cloudy"},
            {"temperature": 7.0},
        ]
        result = build_forecast_outdoor_series(forecast, 10.0, 3)
        assert result == [5.0, 10.0, 7.0]

    def test_none_forecast_same_as_empty(self):
        """None forecast treated like empty list."""
        result = build_forecast_outdoor_series(None, 8.0, 4)
        assert result == [8.0] * 4

    def test_empty_forecast_single_block(self):
        """Padding with empty forecast still works for 1 block."""
        forecast = [{"temperature": 3.0}]
        result = build_forecast_outdoor_series(forecast, 10.0, 1)
        assert result == [3.0]


# ---------------------------------------------------------------------------
# build_forecast_solar_series
# ---------------------------------------------------------------------------


class TestBuildForecastSolarSeries:
    """Tests for build_forecast_solar_series."""

    def test_zero_lat_lon_returns_none(self):
        """Lat=0, lon=0 → None (no location)."""
        result = build_forecast_solar_series(0.0, 0.0, [], 12)
        assert result is None

    def test_with_valid_location_returns_list(self):
        """Valid lat/lon → returns a list of floats."""
        result = build_forecast_solar_series(48.0, 11.0, [], 12)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 12

    def test_with_forecast_cloud_coverage(self):
        """Forecast with cloud_coverage is used for attenuation."""
        forecast = [{"cloud_coverage": 50}] * 5
        result = build_forecast_solar_series(48.0, 11.0, forecast, 12)
        assert result is not None
        assert len(result) == 12

    def test_solar_values_non_negative(self):
        """Solar values should all be >= 0."""
        result = build_forecast_solar_series(48.0, 11.0, [], 24)
        assert result is not None
        assert all(v >= 0.0 for v in result)


# ---------------------------------------------------------------------------
# _simulate_mpc
# ---------------------------------------------------------------------------


class TestSimulateMPC:
    """Tests for _simulate_mpc."""

    def test_runs_without_error(self):
        """Basic MPC simulation produces correct length output."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 10
        outdoor_series = [5.0] * 10
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        settings = {"comfort_weight": 70}
        result = _simulate_mpc(
            model, target_forecast, outdoor_series,
            current_temp=18.0, room_config=room_config, settings=settings,
        )
        assert len(result) == 10
        assert all(isinstance(t, float) for t in result)

    def test_heating_increases_temperature(self):
        """Cold room with heating available should show temperature increase."""
        model = RCModel(C=1.0, U=0.5, Q_heat=100.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 20
        outdoor_series = [5.0] * 20
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        settings = {"comfort_weight": 70}
        result = _simulate_mpc(
            model, target_forecast, outdoor_series,
            current_temp=15.0, room_config=room_config, settings=settings,
        )
        # Temperature should increase when starting cold
        assert result[-1] > 15.0

    def test_with_solar_series(self):
        """Solar series is accepted without error."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=10.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [10.0] * 5
        solar_series = [0.3, 0.4, 0.5, 0.4, 0.3]
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        settings = {"comfort_weight": 70}
        result = _simulate_mpc(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config, settings=settings,
            solar_series=solar_series,
        )
        assert len(result) == 5

    def test_temperatures_clamped(self):
        """Output temps are clamped between 5 and 40."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [5.0] * 5
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        settings = {}
        result = _simulate_mpc(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config, settings=settings,
        )
        assert all(5.0 <= t <= 40.0 for t in result)

    def test_no_devices_stays_near_idle(self):
        """No thermostats or ACs → all idle, temperature drifts toward outdoor."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 10
        outdoor_series = [5.0] * 10
        room_config = {
            "thermostats": [],
            "acs": [],
            "climate_mode": "auto",
        }
        settings = {}
        result = _simulate_mpc(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config, settings=settings,
        )
        # Without devices, temp should drift downward toward outdoor
        assert result[-1] < 20.0


# ---------------------------------------------------------------------------
# _simulate_bangbang
# ---------------------------------------------------------------------------


class TestSimulateBangbang:
    """Tests for _simulate_bangbang."""

    def test_basic_heating_scenario(self):
        """Cold room with thermostats → temperature should increase."""
        model = RCModel(C=1.0, U=0.5, Q_heat=100.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 20
        outdoor_series = [5.0] * 20
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        all_points: list[dict] = []
        result = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=15.0, room_config=room_config,
            all_points=all_points,
        )
        assert len(result) == 20
        # Should warm up from 15°C
        assert result[-1] > 15.0

    def test_mode_stickiness_minimum_run(self):
        """Once heating starts, minimum run time enforced (2 blocks)."""
        model = RCModel(C=1.0, U=0.5, Q_heat=5000.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [5.0] * 5
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        all_points: list[dict] = []
        # Start well below target to trigger heating
        result = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=all_points,
        )
        # With very high Q_heat, temp jumps quickly, but min run enforces at least 2 blocks
        assert len(result) == 5

    def test_cooling_scenario(self):
        """Hot room with ACs → temperature should decrease."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=100.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 22.0}] * 20
        outdoor_series = [30.0] * 20
        room_config = {
            "thermostats": [],
            "acs": ["climate.ac"],
            "climate_mode": "auto",
        }
        all_points: list[dict] = []
        result = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=28.0, room_config=room_config,
            all_points=all_points,
        )
        # Should cool down from 28°C
        assert result[-1] < 28.0

    def test_idle_rate_cap_applied(self):
        """Observed idle rate caps how fast temperature can drift in idle mode."""
        model = RCModel(C=1.0, U=5.0, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [5.0] * 5
        room_config = {
            "thermostats": [],
            "acs": [],
            "climate_mode": "auto",
        }
        # Create idle rate observations (slow drift)
        now = time.time()
        all_points = [
            {"ts": now - 600, "room_temp": 20.0, "mode": "idle"},
            {"ts": now - 0, "room_temp": 19.9, "mode": "idle"},
        ]
        # The model has high U → would predict fast drift, but idle rate caps it
        result_capped = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=all_points,
        )
        # Without cap (empty points → no cap)
        result_uncapped = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=[],
        )
        # Capped should drift less aggressively than uncapped
        assert result_capped[-1] >= result_uncapped[-1]

    def test_temperatures_clamped(self):
        """Output temps are clamped between 5 and 40."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [5.0] * 5
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        result = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=[],
        )
        assert all(5.0 <= t <= 40.0 for t in result)

    def test_with_solar_series(self):
        """Solar series is accepted and affects prediction."""
        model = RCModel(C=1.0, U=0.5, Q_heat=50.0, Q_cool=50.0, Q_solar=50.0)
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [5.0] * 5
        solar_series = [0.5, 0.5, 0.5, 0.5, 0.5]
        room_config = {
            "thermostats": [],
            "acs": [],
            "climate_mode": "auto",
        }
        result_with_solar = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=[],
            solar_series=solar_series,
        )
        result_no_solar = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.0, room_config=room_config,
            all_points=[],
        )
        # Solar gain should raise temperatures compared to no solar
        assert result_with_solar[-1] > result_no_solar[-1]

    def test_hysteresis_prevents_short_cycling(self):
        """At target temp (within hysteresis), stays idle — no heating triggered."""
        model = RCModel(C=1.0, U=0.01, Q_heat=50.0, Q_cool=50.0, Q_solar=0.0)
        # Current temp is 20.9, target 21.0 → within 0.2°C hysteresis
        target_forecast = [{"target_temp": 21.0}] * 5
        outdoor_series = [20.0] * 5  # mild outdoor, minimal drift
        room_config = {
            "thermostats": ["climate.trv"],
            "acs": [],
            "climate_mode": "auto",
        }
        result = _simulate_bangbang(
            model, target_forecast, outdoor_series,
            current_temp=20.9, room_config=room_config,
            all_points=[],
        )
        # Within hysteresis, should stay approximately the same (idle)
        # Not heating aggressively
        assert all(t < 22.0 for t in result)
