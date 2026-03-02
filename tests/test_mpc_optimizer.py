"""Tests for the MPC optimizer: MPCOptimizer, MPCPlan."""

from __future__ import annotations

import pytest

from custom_components.roommind.thermal_model import RCModel
from custom_components.roommind.mpc_optimizer import MPCOptimizer, MPCPlan


def test_optimizer_idle_at_target():
    """When at target with no heat loss, plan should be all idle."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    # Outdoor == room temp → no heat loss → idle is cheapest
    plan = opt.optimize(
        T_room=21.0,
        T_outdoor_series=[21.0] * 12,
        target_series=[21.0] * 12,
        dt_minutes=5,
    )
    assert plan.actions[0] == "idle"


def test_optimizer_heats_when_cold():
    """When below target, plan should start heating."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    plan = opt.optimize(
        T_room=17.0,
        T_outdoor_series=[5.0] * 24,
        target_series=[21.0] * 24,
        dt_minutes=5,
    )
    assert plan.actions[0] == "heating"


def test_optimizer_cools_when_hot():
    """When above target, plan should start cooling."""
    # Use moderate Q_cool so one block doesn't overshoot wildly
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=200.0)
    opt = MPCOptimizer(model, can_heat=False, can_cool=True)
    plan = opt.optimize(
        T_room=27.0,
        T_outdoor_series=[30.0] * 24,
        target_series=[23.0] * 24,
        dt_minutes=5,
    )
    assert plan.actions[0] == "cooling"


def test_optimizer_preheats():
    """Should start heating BEFORE target time to reach temp on time."""
    # Large thermal mass so pre-heating actually raises temperature that persists
    model = RCModel(C=200.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    # Target is eco (17°C) for first 6 blocks, then comfort (21°C)
    targets = [17.0] * 6 + [21.0] * 18
    plan = opt.optimize(
        T_room=17.0,
        T_outdoor_series=[17.0] * 24,
        target_series=targets,
        dt_minutes=5,
    )
    # Should start heating before block 6 (pre-heating)
    first_heat = next(i for i, a in enumerate(plan.actions) if a == "heating")
    assert first_heat < 6  # starts before target changes


def test_optimizer_stops_near_target():
    """Should not overshoot far past target; proportional power keeps temp controlled."""
    # Realistic thermal mass: time constant = C/U = 200/50 = 4 hours
    model = RCModel(C=200.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    plan = opt.optimize(
        T_room=18.0,
        T_outdoor_series=[5.0] * 72,
        target_series=[21.0] * 72,
        dt_minutes=5,
    )
    assert plan.actions[0] == "heating", "Should heat when below target"
    # With proportional control, power fraction decreases as temp approaches target
    # Final temperature should be near target, not wildly overshooting
    assert plan.temperatures[-1] < 22.0
    # Power fractions should decrease over time as room warms up
    assert plan.power_fractions[0] > plan.power_fractions[-1] or plan.actions[-1] == "idle"


def test_optimizer_min_run_time():
    """Should not create runs shorter than min_run_blocks."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model, min_run_blocks=2)
    plan = opt.optimize(
        T_room=20.5,
        T_outdoor_series=[5.0] * 12,
        target_series=[21.0] * 12,
        dt_minutes=5,
    )
    # Any heating run should be at least 2 blocks
    in_run = False
    run_length = 0
    for a in plan.actions:
        if a == "heating":
            run_length += 1
            in_run = True
        elif in_run:
            assert run_length >= 2
            run_length = 0
            in_run = False


def test_optimizer_outdoor_gating():
    """Cooling blocked when outdoor below threshold."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model, can_heat=False, can_cool=True, outdoor_cooling_min=16.0)
    plan = opt.optimize(
        T_room=25.0,
        T_outdoor_series=[10.0] * 12,  # below 16°C
        target_series=[22.0] * 12,
        dt_minutes=5,
    )
    assert all(a == "idle" for a in plan.actions)


def test_plan_get_current_action():
    """MPCPlan returns correct action for current time."""
    plan = MPCPlan(
        actions=["heating", "heating", "idle", "idle"],
        temperatures=[18.0, 19.0, 20.5, 21.0, 21.0],
        dt_minutes=5,
    )
    assert plan.get_current_action() == "heating"


def test_plan_empty():
    """Empty plan returns idle."""
    plan = MPCPlan(actions=[], temperatures=[20.0], dt_minutes=5)
    assert plan.get_current_action() == "idle"


# ---------------------------------------------------------------------------
# Proportional control tests
# ---------------------------------------------------------------------------

def test_compute_optimal_power_cold_room():
    """Cold room needs high power fraction."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    pf, mode = opt.compute_optimal_power(
        T_room=15.0, T_outdoor=5.0, target=21.0, dt_minutes=5.0,
    )
    assert mode == "heating"
    assert pf >= 0.5  # large error → high power


def test_compute_optimal_power_at_target():
    """At target with matching outdoor → near-zero or idle."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    pf, mode = opt.compute_optimal_power(
        T_room=21.0, T_outdoor=21.0, target=21.0, dt_minutes=5.0,
    )
    assert mode == "idle"
    assert pf == 0.0


def test_power_fractions_in_plan():
    """Optimize returns power_fractions list matching actions length."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    plan = opt.optimize(
        T_room=17.0,
        T_outdoor_series=[5.0] * 12,
        target_series=[21.0] * 12,
        dt_minutes=5,
    )
    assert len(plan.power_fractions) == len(plan.actions)
    # Heating blocks should have positive power fractions
    for i, a in enumerate(plan.actions):
        if a == "heating":
            assert plan.power_fractions[i] > 0.0
        elif a == "idle":
            assert plan.power_fractions[i] == 0.0


def test_get_current_power_fraction_fallback():
    """Empty power_fractions → backward-compatible defaults."""
    plan_heat = MPCPlan(actions=["heating", "idle"], temperatures=[18.0, 19.0, 19.5], dt_minutes=5)
    assert plan_heat.get_current_power_fraction() == 1.0  # active → 1.0

    plan_idle = MPCPlan(actions=["idle"], temperatures=[21.0, 21.0], dt_minutes=5)
    assert plan_idle.get_current_power_fraction() == 0.0  # idle → 0.0

    plan_empty = MPCPlan(actions=[], temperatures=[21.0], dt_minutes=5)
    assert plan_empty.get_current_power_fraction() == 0.0  # no actions → 0.0


def test_power_fraction_clamped():
    """Power fraction should always be in [MIN_POWER_FRACTION, 1.0] when active."""
    from custom_components.roommind.const import MIN_POWER_FRACTION
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    opt = MPCOptimizer(model)
    plan = opt.optimize(
        T_room=17.0,
        T_outdoor_series=[5.0] * 12,
        target_series=[21.0] * 12,
        dt_minutes=5,
    )
    for i, a in enumerate(plan.actions):
        pf = plan.power_fractions[i]
        if a != "idle":
            assert pf >= MIN_POWER_FRACTION
            assert pf <= 1.0
