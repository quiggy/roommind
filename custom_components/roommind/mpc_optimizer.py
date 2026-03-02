"""MPC Optimizer using Dynamic Programming for RoomMind."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .const import MIN_POWER_FRACTION, MODE_COOLING, MODE_HEATING, MODE_IDLE
from .thermal_model import RCModel


@dataclass
class MPCPlan:
    """Result of MPC optimization — a planned sequence of actions."""

    actions: list[str]
    temperatures: list[float]  # len = len(actions) + 1 (includes initial)
    dt_minutes: float = 5.0
    power_fractions: list[float] = field(default_factory=list)

    def get_current_action(self) -> str:
        """Return the action for the current (first) time block."""
        if not self.actions:
            return MODE_IDLE
        return self.actions[0]

    def get_current_power_fraction(self) -> float:
        """Power fraction for the current block. Backward-compatible."""
        if not self.power_fractions:
            return 1.0 if self.actions and self.actions[0] != MODE_IDLE else 0.0
        return self.power_fractions[0]


@dataclass
class MPCOptimizer:
    """Dynamic Programming optimizer for heating/cooling control.

    Plans the optimal on/off schedule over a prediction horizon
    to minimize a weighted sum of temperature deviation and energy use.
    """

    model: RCModel
    can_heat: bool = True
    can_cool: bool = True
    w_comfort: float = 10.0
    w_energy: float = 1.0
    min_run_blocks: int = 2  # minimum 2 blocks (10 min) per run
    outdoor_cooling_min: float = 16.0
    outdoor_heating_max: float = 22.0
    temp_min: float = 5.0  # frost protection
    temp_max: float = 30.0  # overheat protection

    def optimize(
        self,
        T_room: float,
        T_outdoor_series: list[float],
        target_series: list[float],
        dt_minutes: float = 5.0,
        *,
        solar_series: list[float] | None = None,
    ) -> MPCPlan:
        """Find optimal action sequence over the planning horizon.

        Uses forward simulation with greedy optimization per block,
        considering minimum run time constraints.
        """
        n_blocks = min(len(T_outdoor_series), len(target_series))
        if n_blocks == 0 or not math.isfinite(T_room):
            return MPCPlan(actions=[], temperatures=[T_room], dt_minutes=dt_minutes)

        q_solar = solar_series or [0.0] * n_blocks

        actions: list[str] = []
        temperatures: list[float] = [T_room]
        power_fractions: list[float] = []
        current_temp = T_room
        current_mode = MODE_IDLE
        blocks_in_mode = 0

        for i in range(n_blocks):
            T_out = T_outdoor_series[i]
            target = target_series[i]
            qs = q_solar[i] if i < len(q_solar) else 0.0

            # Determine available actions this block
            available = [MODE_IDLE]
            if self.can_heat and not self._is_outdoor_gated(MODE_HEATING, T_out):
                available.append(MODE_HEATING)
            if self.can_cool and not self._is_outdoor_gated(MODE_COOLING, T_out):
                available.append(MODE_COOLING)

            # If in a run and below min_run_blocks, must continue
            if current_mode != MODE_IDLE and blocks_in_mode < self.min_run_blocks:
                if current_mode in available:
                    best_action = current_mode
                else:
                    best_action = MODE_IDLE  # forced off by constraint
            else:
                # Evaluate each action: look ahead to find best
                best_action = MODE_IDLE
                best_cost = float("inf")
                future_solar = q_solar[i:] if q_solar else None
                for action in available:
                    cost = self._evaluate_action(
                        action, current_temp, T_out, target,
                        T_outdoor_series[i:], target_series[i:], dt_minutes,
                        future_solar=future_solar,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_action = action

            # Compute proportional power fraction for this block
            pf, _ = self.compute_optimal_power(current_temp, T_out, target, dt_minutes, q_solar=qs)
            if best_action == MODE_IDLE:
                pf = 0.0
            elif best_action != MODE_IDLE and pf == 0.0:
                pf = 1.0  # min_run_blocks enforcement: keep full power

            # Apply action with proportional Q for accurate forward prediction
            if best_action == MODE_HEATING:
                Q = pf * self.model.Q_heat
            elif best_action == MODE_COOLING:
                Q = -(pf * self.model.Q_cool)
            else:
                Q = 0.0
            next_temp = self.model.predict(current_temp, T_out, Q, dt_minutes, q_solar=qs)
            next_temp = max(self.temp_min, min(next_temp, self.temp_max))

            actions.append(best_action)
            temperatures.append(round(next_temp, 2))
            power_fractions.append(round(pf, 3))

            # Track run length
            if best_action == current_mode:
                blocks_in_mode += 1
            else:
                current_mode = best_action
                blocks_in_mode = 1

            current_temp = next_temp

        return MPCPlan(
            actions=actions,
            temperatures=temperatures,
            dt_minutes=dt_minutes,
            power_fractions=power_fractions,
        )

    def _evaluate_action(
        self,
        action: str,
        T_room: float,
        T_outdoor: float,
        target: float,
        future_T_outdoor: list[float],
        future_targets: list[float],
        dt_minutes: float,
        *,
        future_solar: list[float] | None = None,
    ) -> float:
        """Evaluate the cost of taking an action, looking a few steps ahead."""
        lookahead = min(6, len(future_T_outdoor))  # 30 min lookahead for local decision
        Q = self._action_to_Q(action)
        total_cost = 0.0
        T = T_room
        solar = future_solar or []

        for j in range(lookahead):
            qs = solar[j] if j < len(solar) else 0.0
            # Simulate HVAC for min_run_blocks (not just 1 block) to correctly
            # value sustained heating/cooling over the lookahead horizon.
            T = self.model.predict(
                T, future_T_outdoor[j],
                Q if j < self.min_run_blocks else 0.0,
                dt_minutes,
                q_solar=qs,
            )
            # Clamp temperature in lookahead to prevent cost explosion
            # from implausible model predictions
            T = max(self.temp_min, min(self.temp_max, T))
            tgt = future_targets[j]
            # Comfort cost: squared deviation from target
            total_cost += self.w_comfort * (T - tgt) ** 2
            # Energy cost: proportional to HVAC power for min_run blocks
            if j < self.min_run_blocks and action != MODE_IDLE:
                total_cost += self.w_energy * abs(Q) / 1000.0

        return total_cost

    def _action_to_Q(self, action: str) -> float:
        if action == MODE_HEATING:
            return self.model.Q_heat
        if action == MODE_COOLING:
            return -self.model.Q_cool
        return 0.0

    def _is_outdoor_gated(self, mode: str, T_outdoor: float) -> bool:
        if mode == MODE_COOLING and T_outdoor < self.outdoor_cooling_min:
            return True
        if mode == MODE_HEATING and T_outdoor > self.outdoor_heating_max:
            return True
        return False

    def compute_optimal_power(
        self,
        T_room: float,
        T_outdoor: float,
        target: float,
        dt_minutes: float,
        *,
        q_solar: float = 0.0,
    ) -> tuple[float, str]:
        """Analytical closed-form optimal heating/cooling power.

        Returns (power_fraction in [0,1], mode).
        """
        if not math.isfinite(T_room) or not math.isfinite(target):
            return 0.0, MODE_IDLE

        dt_h = dt_minutes / 60.0
        alpha = self.model.U
        if alpha < 0.01:
            beta = alpha * dt_h  # Euler approx for tiny alpha
        else:
            beta = 1.0 - math.exp(-alpha * dt_h)

        if beta < 1e-9:
            return 0.0, MODE_IDLE

        # Drift temperature: where the room would go with no HVAC
        T_drift = T_room + beta * (T_outdoor - T_room)
        # Add predicted solar gain to drift
        if alpha > 0.01:
            T_drift += beta * self.model.Q_solar * q_solar / alpha
        else:
            T_drift += self.model.Q_solar * q_solar * dt_h

        Q_required = (target - T_drift) * alpha / beta

        # Energy penalty: bias toward less power based on comfort/energy weights
        energy_bias = (self.w_energy / max(self.w_comfort, 0.01)) * alpha / beta * 0.1
        if Q_required > 0:
            Q_required = max(0.0, Q_required - energy_bias)
        elif Q_required < 0:
            Q_required = min(0.0, Q_required + energy_bias)

        if Q_required > 0 and self.can_heat and not self._is_outdoor_gated(MODE_HEATING, T_outdoor):
            frac = min(Q_required / max(self.model.Q_heat, 0.01), 1.0)
            return max(frac, MIN_POWER_FRACTION), MODE_HEATING
        elif Q_required < 0 and self.can_cool and not self._is_outdoor_gated(MODE_COOLING, T_outdoor):
            frac = min(abs(Q_required) / max(self.model.Q_cool, 0.01), 1.0)
            return max(frac, MIN_POWER_FRACTION), MODE_COOLING
        else:
            return 0.0, MODE_IDLE
