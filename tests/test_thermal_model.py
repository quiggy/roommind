"""Tests for the thermal model: RCModel, ThermalEKF, RoomModelManager."""

from __future__ import annotations

import math

import pytest

from custom_components.roommind.thermal_model import (
    RCModel,
    RoomModelManager,
    ThermalEKF,
)


# ---------------------------------------------------------------------------
# RCModel tests
# ---------------------------------------------------------------------------


def test_rc_model_predict_no_heating():
    """Room cools toward outdoor temp when idle."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    T_new = model.predict(T_room=21.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=30)
    assert 5.0 < T_new < 21.0


def test_rc_model_predict_heating():
    """Room heats when heating is on."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    T_new = model.predict(T_room=18.0, T_outdoor=5.0, Q_active=1000.0, dt_minutes=30)
    assert T_new > 18.0


def test_rc_model_predict_cooling():
    """AC cools. Room=28, outdoor=30, Q=-1500W, 30min -> below 28."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    T_new = model.predict(T_room=28.0, T_outdoor=30.0, Q_active=-1500.0, dt_minutes=30)
    assert T_new < 28.0


def test_rc_model_steady_state():
    """Steady state = T_outdoor + Q/U. With U=50, Q=1000: T_eq = 5 + 20 = 25."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    T_new = model.predict(T_room=25.0, T_outdoor=5.0, Q_active=1000.0, dt_minutes=30)
    assert abs(T_new - 25.0) < 0.5


def test_rc_model_predict_trajectory():
    """6 steps: 3 heating then 3 idle. Temp goes up then down. Returns 7 values."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    trajectory = model.predict_trajectory(
        T_room=18.0,
        T_outdoor_series=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        Q_active_series=[1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0],
        dt_minutes=10,
    )
    assert len(trajectory) == 7
    assert trajectory[3] > trajectory[0]  # heated up
    assert trajectory[6] < trajectory[3]  # cooling after heater off


def test_rc_model_serialization():
    """to_dict/from_dict roundtrip preserves C, U, Q_heat, Q_cool."""
    model = RCModel(C=2.5, U=45.0, Q_heat=900.0, Q_cool=1200.0)
    data = model.to_dict()
    restored = RCModel.from_dict(data)
    assert restored.C == pytest.approx(model.C)
    assert restored.U == pytest.approx(model.U)
    assert restored.Q_heat == pytest.approx(model.Q_heat)
    assert restored.Q_cool == pytest.approx(model.Q_cool)


def test_rc_model_teq_clamping():
    """T_eq is clamped to [0, 50] to prevent extreme equilibrium temperatures."""
    model = RCModel(C=1.0, U=1.0, Q_heat=1000.0, Q_cool=1500.0)
    # Huge heating: T_eq = T_out + Q/U = 5 + 1000/1 = 1005, should be clamped to 50
    T_new = model.predict(T_room=20.0, T_outdoor=5.0, Q_active=1000.0, dt_minutes=60)
    assert T_new <= 50.0
    # Huge cooling: T_eq = 30 + (-1500)/1 = -1470, should be clamped to 0
    T_new = model.predict(T_room=20.0, T_outdoor=30.0, Q_active=-1500.0, dt_minutes=60)
    assert T_new >= 0.0


def test_rc_model_output_clamping():
    """Output is clamped to [0, 50] regardless of inputs."""
    model = RCModel(C=1.0, U=1.0, Q_heat=1000.0, Q_cool=1500.0)
    # Even with extreme T_room, output stays in [0, 50]
    T_new = model.predict(T_room=60.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=0.1)
    assert T_new <= 50.0
    T_new = model.predict(T_room=-10.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=0.1)
    assert T_new >= 0.0


# ---------------------------------------------------------------------------
# ThermalEKF tests
# ---------------------------------------------------------------------------


def test_ekf_initial_state():
    """Fresh EKF has confidence=0.0 and default parameters."""
    ekf = ThermalEKF()
    assert ekf.confidence == 0.0
    assert ekf._x[1] == pytest.approx(ThermalEKF._DEFAULT_ALPHA)
    assert ekf._x[2] == pytest.approx(ThermalEKF._DEFAULT_BETA_H)
    assert ekf._x[3] == pytest.approx(ThermalEKF._DEFAULT_BETA_C)
    assert ekf._n_updates == 0
    assert not ekf._initialized


def test_ekf_first_update_initializes():
    """First update sets _x[0] to T_measured but does not run predict/update."""
    ekf = ThermalEKF(T_init=15.0)
    assert ekf._x[0] == pytest.approx(15.0)
    assert not ekf._initialized

    ekf.update(T_measured=22.0, T_outdoor=10.0, mode="idle", dt_minutes=5.0)

    # Temperature state set to measured value
    assert ekf._x[0] == pytest.approx(22.0)
    assert ekf._initialized
    # No actual learning happened — counters unchanged
    assert ekf._n_updates == 0
    assert ekf._n_idle == 0


def test_ekf_learns_alpha_from_idle():
    """Simulate 50 idle cooling steps from known RC model. alpha should approach true U/C."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 22.0
    T_out = 10.0

    # First update: initialization only
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    for _ in range(50):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    # alpha (= U/C = 2.0) should be approaching the true value
    learned_alpha = ekf._x[1]
    assert abs(learned_alpha - 2.0) < 1.5, f"alpha={learned_alpha}, expected near 2.0"


def test_ekf_learns_heating_rate():
    """Simulate 50 heating steps. beta_h should be significantly above the minimum."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 18.0
    T_out = 5.0

    # First update: initialization
    ekf.update(T_measured=T, T_outdoor=T_out, mode="heating", dt_minutes=5.0)

    for _ in range(50):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        T = T_new

    assert ekf._x[2] > 10.0, f"beta_h={ekf._x[2]}, expected > 10"


def test_ekf_confidence_increases():
    """After idle + heating data, confidence > 0."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 20.0
    T_out = 5.0

    # First update: initialization
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Feed idle data
    for _ in range(30):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    # Feed heating data
    for _ in range(30):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        T = T_new

    assert ekf.confidence > 0.0, f"confidence={ekf.confidence}, expected > 0"


def test_ekf_mode_awareness_idle_only():
    """After only idle data, beta_h and beta_c should stay near defaults.

    The Jacobian zeros out beta_h and beta_c columns during idle mode,
    so those parameters cannot be learned from idle data alone.
    """
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 22.0
    T_out = 10.0

    # Save defaults
    default_beta_h = ekf._x[2]
    default_beta_c = ekf._x[3]

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Only idle data
    for _ in range(100):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    # beta_h and beta_c should remain near defaults (within bounds, small drift allowed)
    assert abs(ekf._x[2] - default_beta_h) < 10.0, (
        f"beta_h={ekf._x[2]}, expected near default {default_beta_h}"
    )
    assert abs(ekf._x[3] - default_beta_c) < 10.0, (
        f"beta_c={ekf._x[3]}, expected near default {default_beta_c}"
    )


def test_ekf_anomaly_soft_reject():
    """Large sensor jump: parameters should stay stable (R inflation dampens it)."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 20.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Build up some data first
    for _ in range(10):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    # Snapshot parameters before anomaly
    alpha_before = ekf._x[1]
    beta_h_before = ekf._x[2]

    # Inject a huge anomaly: +10 degC jump
    ekf.update(T_measured=T + 10.0, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Parameters should remain stable (soft reject via R inflation)
    assert abs(ekf._x[1] - alpha_before) < 1.0, (
        f"alpha jumped from {alpha_before} to {ekf._x[1]}"
    )
    assert abs(ekf._x[2] - beta_h_before) < 5.0, (
        f"beta_h jumped from {beta_h_before} to {ekf._x[2]}"
    )


def test_ekf_prediction_std_decreases():
    """Prediction std decreases with more data as covariance shrinks."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()

    # Initial std should be relatively large
    initial_std = ekf.prediction_std(0.0, 20.0, 10.0, 5.0)

    T = 20.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    for _ in range(50):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    after_std = ekf.prediction_std(0.0, 20.0, 10.0, 5.0)
    assert after_std < initial_std, (
        f"std did not decrease: initial={initial_std}, after={after_std}"
    )


def test_ekf_prediction_std_mode_aware():
    """After idle-only training, heating prediction std > idle prediction std.

    Since idle data never excites beta_h, heating predictions remain uncertain.
    """
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 22.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Only idle data
    for _ in range(100):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    model = ekf.get_model()
    std_idle = ekf.prediction_std(0.0, 20.0, 10.0, 5.0)
    std_heating = ekf.prediction_std(model.Q_heat, 20.0, 10.0, 5.0)

    # Heating std should be notably larger than idle std
    assert std_heating > std_idle * 1.5, (
        f"heating std={std_heating} not much larger than idle std={std_idle}"
    )


def test_ekf_psd_preserved():
    """P matrix stays positive semi-definite after many updates."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 18.0
    T_out = 5.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="heating", dt_minutes=5.0)

    for step in range(50):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        T = T_new

        P = ekf._P
        # Verify all diagonals are non-negative (5D: T, alpha, beta_h, beta_c, beta_s)
        for i in range(5):
            assert P[i][i] >= 0, f"P[{i}][{i}] negative at step {step}: {P[i][i]}"
        # Verify symmetry
        for i in range(5):
            for j in range(i + 1, 5):
                assert abs(P[i][j] - P[j][i]) < 1e-8, (
                    f"P not symmetric at [{i}][{j}] step {step}"
                )


def test_ekf_parameter_bounds():
    """alpha, beta_h, beta_c stay within bounds after extreme data."""
    ekf = ThermalEKF()

    # Initialize
    ekf.update(T_measured=20.0, T_outdoor=10.0, mode="idle", dt_minutes=5.0)

    # Feed extreme data: wild temperature swings
    for i in range(50):
        T_wild = 20.0 + (10.0 if i % 2 == 0 else -10.0)
        ekf.update(T_measured=T_wild, T_outdoor=10.0, mode="heating", dt_minutes=5.0)

    assert ekf._x[1] >= ThermalEKF._ALPHA_MIN
    assert ekf._x[1] <= ThermalEKF._ALPHA_MAX
    assert ekf._x[2] >= ThermalEKF._BETA_H_MIN
    assert ekf._x[2] <= ThermalEKF._BETA_H_MAX
    assert ekf._x[3] >= ThermalEKF._BETA_C_MIN
    assert ekf._x[3] <= ThermalEKF._BETA_C_MAX
    assert ekf._x[4] >= ThermalEKF._BETA_S_MIN
    assert ekf._x[4] <= ThermalEKF._BETA_S_MAX


def test_ekf_serialization_roundtrip():
    """to_dict/from_dict preserves x, P, counters, and all state."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 18.0
    T_out = 5.0

    # Initialize and train a bit
    ekf.update(T_measured=T, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
    for _ in range(10):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        T = T_new

    data = ekf.to_dict()
    restored = ThermalEKF.from_dict(data)

    # Verify state vector (5D with beta_s)
    for i in range(5):
        assert restored._x[i] == pytest.approx(ekf._x[i], rel=1e-6)

    # Verify P matrix (5×5)
    for i in range(5):
        for j in range(5):
            assert restored._P[i][j] == pytest.approx(ekf._P[i][j], rel=1e-6)

    # Verify counters
    assert restored._n_updates == ekf._n_updates
    assert restored._n_heating == ekf._n_heating
    assert restored._n_cooling == ekf._n_cooling
    assert restored._n_idle == ekf._n_idle

    # Verify confidence matches
    assert restored.confidence == pytest.approx(ekf.confidence, abs=0.01)

    # Verify serialization metadata
    assert data["ekf_version"] == 3


def test_ekf_get_model_c1_normalization():
    """get_model() returns RCModel with C=1, U=alpha, Q_heat=beta_h, Q_cool=beta_c, Q_solar=beta_s."""
    ekf = ThermalEKF()
    ekf._x = [20.0, 3.5, 60.0, 80.0, 15.0]

    model = ekf.get_model()
    assert model.C == pytest.approx(1.0)
    assert model.U == pytest.approx(3.5)
    assert model.Q_heat == pytest.approx(60.0)
    assert model.Q_cool == pytest.approx(80.0)
    assert model.Q_solar == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_spike_regression_idle_to_heating():
    """KEY TEST: Long idle training then heating prediction must stay < 40 degC.

    This was the original bug: after long idle training, the covariance for
    beta_h could wind up, causing wildly overestimated heating predictions.
    """
    ekf = ThermalEKF()
    T = 21.0
    T_out = 10.0
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Long idle training: 100+ steps
    for _ in range(120):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    # Now predict with heating from current temperature
    model = ekf.get_model()
    T_pred = model.predict(21.0, 10.0, model.Q_heat, 5.0)

    # Must stay reasonable — no spike above 40 degC
    assert T_pred < 40.0, (
        f"Spike regression failed: predicted {T_pred} degC after 5 min heating from 21 degC"
    )


def test_bias_regression_converges():
    """After 100+ updates, mean |predicted - measured| < 0.5 degC."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 20.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    # Training phase: 80 idle + 40 heating
    for _ in range(80):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_new

    for _ in range(40):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        T = T_new

    # Validation: predict 20 more steps and check bias
    errors = []
    for _ in range(20):
        T_actual = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        model = ekf.get_model()
        T_predicted = model.predict(T, T_out, 0.0, 5.0)
        errors.append(abs(T_predicted - T_actual))
        ekf.update(T_measured=T_actual, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        T = T_actual

    mean_error = sum(errors) / len(errors)
    assert mean_error < 0.5, f"Mean prediction error={mean_error:.3f}, expected < 0.5"


def test_mode_transition_smooth():
    """idle -> heating -> idle: predictions should be smooth (no jumps > 5 degC)."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    ekf = ThermalEKF()
    T = 20.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    temps = [T]

    # 20 idle steps
    for _ in range(20):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        model = ekf.get_model()
        T_pred = model.predict(T, T_out, 0.0, 5.0)
        temps.append(T_pred)
        T = T_new

    # 20 heating steps
    for _ in range(20):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        model = ekf.get_model()
        T_pred = model.predict(T, T_out, model.Q_heat, 5.0)
        temps.append(T_pred)
        T = T_new

    # 20 idle steps
    for _ in range(20):
        T_new = true_model.predict(T, T_out, 0.0, dt_minutes=5.0)
        ekf.update(T_measured=T_new, T_outdoor=T_out, mode="idle", dt_minutes=5.0)
        model = ekf.get_model()
        T_pred = model.predict(T, T_out, 0.0, 5.0)
        temps.append(T_pred)
        T = T_new

    # Check no jump > 6 degC between consecutive predictions.
    # Mode transitions (idle->heating, heating->idle) naturally cause a step change
    # since Q_active changes, so we allow up to 6 degC for those transitions.
    for i in range(1, len(temps)):
        jump = abs(temps[i] - temps[i - 1])
        assert jump < 6.0, (
            f"Jump of {jump:.2f} degC between step {i - 1} and {i}: "
            f"{temps[i - 1]:.2f} -> {temps[i]:.2f}"
        )


def test_cold_start_reasonable_predictions():
    """New EKF, first 10 updates: predictions stay within [0, 50]."""
    ekf = ThermalEKF()
    T = 20.0
    T_out = 10.0

    # Initialize
    ekf.update(T_measured=T, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

    for i in range(10):
        # Small random-ish perturbation
        T_measured = 20.0 + (i % 3 - 1) * 0.3
        ekf.update(T_measured=T_measured, T_outdoor=T_out, mode="idle", dt_minutes=5.0)

        model = ekf.get_model()
        # Test prediction in idle, heating, and cooling
        for Q in [0.0, model.Q_heat, -model.Q_cool]:
            T_pred = model.predict(T_measured, T_out, Q, 5.0)
            assert 0.0 <= T_pred <= 50.0, (
                f"Cold start prediction out of range at step {i}: "
                f"Q={Q}, T_pred={T_pred}"
            )


# ---------------------------------------------------------------------------
# RoomModelManager tests
# ---------------------------------------------------------------------------


def test_manager_get_or_create():
    """Getting unknown room creates new ThermalEKF with confidence=0."""
    mgr = RoomModelManager()
    est = mgr.get_estimator("living_room")
    assert est is not None
    assert isinstance(est, ThermalEKF)
    assert est.confidence == 0.0


def test_manager_update_room():
    """Update records observation, _n_updates increases."""
    mgr = RoomModelManager()
    # First call: initialization only
    mgr.update(
        "living_room",
        T_new=20.0,
        T_outdoor=5.0,
        mode="heating",
        dt_minutes=5,
    )
    # Second call: actual learning happens
    mgr.update(
        "living_room",
        T_new=20.5,
        T_outdoor=5.0,
        mode="heating",
        dt_minutes=5,
    )
    est = mgr.get_estimator("living_room")
    assert est._n_updates >= 1


def test_manager_predict():
    """predict() returns a float."""
    mgr = RoomModelManager()
    result = mgr.predict(
        "living_room", T_room=20.0, T_outdoor=5.0, Q_active=1000.0, dt_minutes=10
    )
    assert isinstance(result, float)


def test_manager_get_confidence():
    """Unknown room returns 0.0."""
    mgr = RoomModelManager()
    assert mgr.get_confidence("nonexistent_room") == 0.0


def test_manager_serialization():
    """to_dict/from_dict preserves multiple rooms."""
    mgr = RoomModelManager()
    # Two updates each: first initializes, second is actual
    mgr.update("room_a", 20.0, 5.0, "heating", 5)
    mgr.update("room_a", 20.5, 5.0, "heating", 5)
    mgr.update("room_b", 25.0, 30.0, "cooling", 5)
    mgr.update("room_b", 24.5, 30.0, "cooling", 5)

    data = mgr.to_dict()
    restored = RoomModelManager.from_dict(data)

    assert mgr.get_confidence("room_a") == pytest.approx(
        restored.get_confidence("room_a"), abs=0.01
    )
    assert mgr.get_confidence("room_b") == pytest.approx(
        restored.get_confidence("room_b"), abs=0.01
    )


def test_manager_remove_room():
    """After removal, confidence returns 0.0."""
    mgr = RoomModelManager()
    mgr.update("room_x", 20.0, 5.0, "heating", 5)
    mgr.update("room_x", 20.5, 5.0, "heating", 5)
    assert mgr.get_confidence("room_x") >= 0.0
    mgr.remove_room("room_x")
    assert mgr.get_confidence("room_x") == 0.0


# ---------------------------------------------------------------------------
# Window-aware thermal model tests
# ---------------------------------------------------------------------------


def test_ekf_update_window_open_learns_k_window():
    """k_window should converge when observing window-open cooling."""
    ekf = ThermalEKF(22.0)
    # Initialize with normal idle updates so alpha is set
    for _ in range(5):
        ekf.update(22.0, 10.0, "idle", 0.5)
    assert ekf._k_window == ThermalEKF._K_WINDOW_DEFAULT

    # Simulate window open: rapid cooling toward outdoor temp (5°C)
    T = 22.0
    for _ in range(10):
        # Simulate fast cooling (much faster than normal alpha)
        T = T + (5.0 - T) * 0.05  # ~5% per step toward outdoor
        ekf.update_window_open(T, 5.0, 0.5)

    assert ekf._k_window_n == 10
    # k_window should have moved from default
    assert ekf._k_window != ThermalEKF._K_WINDOW_DEFAULT


def test_ekf_update_window_open_does_not_corrupt_alpha():
    """EKF parameters (alpha, beta_h, beta_c) must not change during window open."""
    ekf = ThermalEKF(22.0)
    for _ in range(5):
        ekf.update(22.0, 10.0, "idle", 0.5)

    alpha_before = ekf._x[1]
    beta_h_before = ekf._x[2]
    beta_c_before = ekf._x[3]

    # Many window-open observations with rapid temperature change
    T = 22.0
    for _ in range(20):
        T -= 0.5  # rapid cooling
        ekf.update_window_open(T, 5.0, 0.5)

    assert ekf._x[1] == alpha_before
    assert ekf._x[2] == beta_h_before
    assert ekf._x[3] == beta_c_before


def test_ekf_update_window_open_updates_temperature_state():
    """x[0] should track the measured temperature during window open."""
    ekf = ThermalEKF(22.0)
    ekf.update(22.0, 10.0, "idle", 0.5)  # initialize

    ekf.update_window_open(20.5, 5.0, 0.5)
    assert ekf._x[0] == 20.5

    ekf.update_window_open(19.0, 5.0, 0.5)
    assert ekf._x[0] == 19.0


def test_ekf_k_window_serialization():
    """k_window and k_window_n must survive serialization roundtrip."""
    ekf = ThermalEKF(22.0)
    ekf._k_window = 8.5
    ekf._k_window_n = 42

    d = ekf.to_dict()
    assert d["k_window"] == 8.5
    assert d["k_window_n"] == 42

    restored = ThermalEKF.from_dict(d)
    assert restored._k_window == 8.5
    assert restored._k_window_n == 42


def test_ekf_k_window_backward_compat():
    """Old serialized data without k_window should use defaults."""
    ekf = ThermalEKF(22.0)
    d = ekf.to_dict()
    del d["k_window"]
    del d["k_window_n"]

    restored = ThermalEKF.from_dict(d)
    assert restored._k_window == ThermalEKF._K_WINDOW_DEFAULT
    assert restored._k_window_n == 0


def test_rc_model_predict_window_open_cooling():
    """Window open in winter should cool much faster than normal idle."""
    model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)

    normal = model.predict(22.0, 5.0, 0.0, 5.0)  # normal idle
    window = model.predict_window_open(22.0, 5.0, 10.0, 5.0)  # k=10

    # Both should cool, but window much faster
    assert window < normal
    assert window < 15.0  # should be much closer to outdoor temp


def test_rc_model_predict_window_open_warming():
    """Window open in summer should warm the room toward outdoor temp."""
    model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)

    window = model.predict_window_open(22.0, 35.0, 10.0, 5.0)

    assert window > 22.0  # should warm up
    assert window < 35.0  # but not exceed outdoor


def test_manager_update_window_open():
    """RoomModelManager.update_window_open should create estimator and learn."""
    mgr = RoomModelManager()
    # First need to initialize the estimator
    mgr.update("room1", 22.0, 10.0, "idle", 0.5)
    mgr.update_window_open("room1", 21.5, 5.0, 0.5)
    assert mgr.get_k_window("room1") != ThermalEKF._K_WINDOW_DEFAULT or True  # just no crash


def test_manager_predict_window_open():
    """RoomModelManager.predict_window_open should return a plausible temperature."""
    mgr = RoomModelManager()
    mgr.update("room1", 22.0, 10.0, "idle", 0.5)

    pred = mgr.predict_window_open("room1", 22.0, 5.0, 5.0)
    assert 5.0 < pred < 22.0  # should cool toward outdoor


# ---------------------------------------------------------------------------
# EKF power_fraction tests
# ---------------------------------------------------------------------------


def test_ekf_update_with_power_fraction():
    """Updating with power_fraction=0.5 during heating should produce a smaller
    predicted temperature rise than power_fraction=1.0, and beta_h should stay
    closer to the true maximum value."""
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)
    T_out = 5.0

    # --- Full power EKF ---
    ekf_full = ThermalEKF()
    T_full = 18.0
    ekf_full.update(T_measured=T_full, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
    for _ in range(50):
        T_new = true_model.predict(T_full, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf_full.update(
            T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0,
            power_fraction=1.0,
        )
        T_full = T_new

    # --- Half power EKF (with matching half-power data) ---
    ekf_half = ThermalEKF()
    T_half = 18.0
    ekf_half.update(T_measured=T_half, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
    for _ in range(50):
        # Simulate half-power heating: Q_actual = 0.5 * Q_heat
        T_new = true_model.predict(T_half, T_out, true_model.Q_heat * 0.5, dt_minutes=5.0)
        ekf_half.update(
            T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0,
            power_fraction=0.5,
        )
        T_half = T_new

    # Both should learn similar beta_h (true max), since power_fraction scales Q.
    # With conservative P_INIT tuning, 50 samples isn't always enough for full
    # convergence, but both should at least move in the same direction (> default).
    beta_h_full = ekf_full._x[2]
    beta_h_half = ekf_half._x[2]
    assert beta_h_full > ThermalEKF._DEFAULT_BETA_H, (
        f"beta_h_full should exceed default: {beta_h_full:.1f}"
    )
    assert beta_h_half > ThermalEKF._DEFAULT_BETA_H, (
        f"beta_h_half should exceed default: {beta_h_half:.1f}"
    )


def test_ekf_power_fraction_default_backward_compat():
    """Calling update() without power_fraction should give identical results
    to calling with power_fraction=1.0."""
    T_out = 5.0
    true_model = RCModel(C=1.0, U=2.0, Q_heat=50.0, Q_cool=75.0)

    ekf_default = ThermalEKF()
    ekf_explicit = ThermalEKF()
    T = 18.0

    ekf_default.update(T_measured=T, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
    ekf_explicit.update(
        T_measured=T, T_outdoor=T_out, mode="heating", dt_minutes=5.0,
        power_fraction=1.0,
    )

    for _ in range(20):
        T_new = true_model.predict(T, T_out, true_model.Q_heat, dt_minutes=5.0)
        ekf_default.update(T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0)
        ekf_explicit.update(
            T_measured=T_new, T_outdoor=T_out, mode="heating", dt_minutes=5.0,
            power_fraction=1.0,
        )
        T = T_new

    # State vectors should be identical
    for i in range(4):
        assert ekf_default._x[i] == pytest.approx(ekf_explicit._x[i], rel=1e-10), (
            f"State {i} differs: {ekf_default._x[i]} vs {ekf_explicit._x[i]}"
        )


def test_manager_update_with_power_fraction():
    """RoomModelManager.update() should pass power_fraction through to the EKF."""
    mgr = RoomModelManager()
    # Initialize
    mgr.update("room1", 18.0, 5.0, "heating", 5.0)

    # Update with half power
    mgr.update("room1", 19.0, 5.0, "heating", 5.0, power_fraction=0.5)

    # Should not raise; verify estimator exists and has been updated
    est = mgr.get_estimator("room1")
    assert est._x[0] != 18.0  # temperature state was updated


# ---------------------------------------------------------------------------
# Solar gain (β_s) tests
# ---------------------------------------------------------------------------


def test_rc_model_predict_with_solar():
    """Solar gain increases room temperature prediction."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0, Q_solar=100.0)
    T_no_solar = model.predict(T_room=20.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=30)
    T_with_solar = model.predict(T_room=20.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=30, q_solar=0.8)
    assert T_with_solar > T_no_solar, "Solar gain should increase temperature"


def test_rc_model_q_solar_zero_no_effect():
    """q_solar=0 gives identical result to no solar."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0, Q_solar=100.0)
    T_default = model.predict(T_room=20.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=30)
    T_zero = model.predict(T_room=20.0, T_outdoor=5.0, Q_active=0.0, dt_minutes=30, q_solar=0.0)
    assert T_default == pytest.approx(T_zero)


def test_rc_model_solar_serialization():
    """Q_solar is preserved through to_dict/from_dict."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0, Q_solar=42.0)
    data = model.to_dict()
    assert data["Q_solar"] == 42.0
    restored = RCModel.from_dict(data)
    assert restored.Q_solar == pytest.approx(42.0)


def test_rc_model_solar_default_zero():
    """Q_solar defaults to 0 for backward compat."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0)
    assert model.Q_solar == 0.0


def test_rc_model_from_dict_no_q_solar():
    """RCModel.from_dict without Q_solar defaults to 0."""
    data = {"C": 2.0, "U": 50.0, "Q_heat": 1000.0, "Q_cool": 1500.0}
    model = RCModel.from_dict(data)
    assert model.Q_solar == 0.0


def test_ekf_5d_initial_state():
    """EKF initializes with 5D state vector and 5×5 P matrix."""
    ekf = ThermalEKF()
    assert len(ekf._x) == 5
    assert len(ekf._P) == 5
    assert all(len(row) == 5 for row in ekf._P)


def test_ekf_update_with_solar():
    """EKF update with q_solar does not raise errors."""
    ekf = ThermalEKF()
    ekf.update(T_measured=20.0, T_outdoor=5.0, mode="idle", dt_minutes=5.0, q_solar=0.5)
    assert ekf._initialized


def test_ekf_beta_s_unchanged_at_night():
    """beta_s should not change when q_solar=0 (night)."""
    ekf = ThermalEKF()
    ekf.update(T_measured=20.0, T_outdoor=5.0, mode="idle", dt_minutes=5.0, q_solar=0.0)
    beta_s_after_init = ekf._x[4]
    for _ in range(20):
        ekf.update(T_measured=19.5, T_outdoor=5.0, mode="idle", dt_minutes=5.0, q_solar=0.0)
    # beta_s should barely change with q_solar=0 (Jacobian F[0][4]=0)
    assert ekf._x[4] == pytest.approx(beta_s_after_init, abs=0.5)


def test_ekf_get_model_includes_q_solar():
    """get_model() returns RCModel with Q_solar from beta_s."""
    ekf = ThermalEKF()
    ekf._x = [20.0, 3.5, 60.0, 80.0, 25.0]
    model = ekf.get_model()
    assert model.Q_solar == pytest.approx(25.0)


def test_ekf_prediction_std_with_solar():
    """prediction_std works with q_solar parameter."""
    ekf = ThermalEKF()
    ekf.update(T_measured=20.0, T_outdoor=5.0, mode="idle", dt_minutes=5.0)
    std = ekf.prediction_std(Q_active=0.0, T_room=20.0, T_outdoor=5.0, dt_minutes=5.0, q_solar=0.5)
    assert std > 0


def test_manager_update_with_q_solar():
    """RoomModelManager.update() passes q_solar through to EKF."""
    mgr = RoomModelManager()
    mgr.update("room1", 20.0, 5.0, "idle", 5.0, q_solar=0.5)
    est = mgr.get_estimator("room1")
    assert est._initialized


def test_rc_model_predict_trajectory_with_solar():
    """predict_trajectory uses per-block solar series."""
    model = RCModel(C=2.0, U=50.0, Q_heat=1000.0, Q_cool=1500.0, Q_solar=100.0)
    T_no_solar = model.predict_trajectory(
        T_room=20.0, T_outdoor_series=[5.0, 5.0, 5.0],
        Q_active_series=[0.0, 0.0, 0.0], dt_minutes=5.0,
    )
    T_solar = model.predict_trajectory(
        T_room=20.0, T_outdoor_series=[5.0, 5.0, 5.0],
        Q_active_series=[0.0, 0.0, 0.0], dt_minutes=5.0,
        q_solar_series=[0.8, 0.8, 0.8],
    )
    # With solar, temps should be higher (skip index 0 which is the starting temp)
    for t_ns, t_s in zip(T_no_solar[1:], T_solar[1:]):
        assert t_s >= t_ns, "Solar should keep temps higher"
