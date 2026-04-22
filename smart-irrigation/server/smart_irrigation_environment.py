# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Irrigation environment implementation.

The environment simulates a small farm. On each step, the agent chooses
an irrigation level and the farm state updates based on water, rain,
evaporation, and crop usage.
"""

from __future__ import annotations

import random
from typing import Any, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        SmartIrrigationAction,
        SmartIrrigationObservation,
        SmartIrrigationState,
    )
except ImportError:
    from models import (
        SmartIrrigationAction,
        SmartIrrigationObservation,
        SmartIrrigationState,
    )


DifficultyName = Literal["easy", "medium", "difficult"]


class SmartIrrigationEnvironment(
    Environment[
        SmartIrrigationAction,
        SmartIrrigationObservation,
        SmartIrrigationState,
    ]
):
    """Environment for three smart irrigation tasks with rising difficulty."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS = 5
    DEFAULT_WATER_BUDGET = 100.0
    IRRIGATION_EFFECTS = [0.0, 5.0, 10.0, 18.0]
    HUMIDITY_EFFECTS = [0.0, 1.0, 2.0, 3.0]
    RAIN_AMOUNT = 8.0
    WATER_COSTS = [0.0, 1.0, 2.0, 3.0]
    RAW_REWARD_MIN = -46.0
    RAW_REWARD_MAX = 14.0
    REWARD_EPSILON = 0.02
    STRESS_PENALTY_WEIGHT = 0.45

    def __init__(self) -> None:
        """Initialize the environment and its random generator."""
        self._rng = random.Random()
        self._state: SmartIrrigationState = SmartIrrigationState(
            episode_id=str(uuid4()),
            step_count=0,
        )
        self._difficulty: DifficultyName = "easy"
        self._soil_moisture = 50.0
        self._temperature = 25.0
        self._humidity = 55.0
        self._rainfall_forecast: int | None = 0
        self._rain_probability = 0.0
        self._crop_stage = 0.0
        self._crop_stress_accumulation = 0.0
        self._water_remaining: float | None = None
        self._total_water_used = 0.0
        self._last_reward = self.REWARD_EPSILON
        self._last_irrigation_level = 0
        self._update_state()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "easy",
        scenario: Optional[str] = None,
        water_budget: Optional[float] = None,
        **kwargs,
    ) -> SmartIrrigationObservation:
        """
        Reset the farm to a fresh starting condition.

        A seed can be passed to make the start state reproducible.
        """
        del kwargs

        if seed is not None:
            self._rng.seed(seed)

        if water_budget is not None and water_budget < 0:
            raise ValueError("water_budget must be non-negative.")

        self._difficulty = self._resolve_difficulty(scenario or difficulty)
        self._state = SmartIrrigationState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._soil_moisture = round(self._rng.uniform(42.0, 58.0), 2)
        self._temperature = round(self._rng.uniform(22.0, 32.0), 2)
        self._humidity = round(self._rng.uniform(45.0, 70.0), 2)
        self._crop_stage = 0.0
        self._crop_stress_accumulation = 0.0
        self._water_remaining = (
            round(
                self.DEFAULT_WATER_BUDGET if water_budget is None else water_budget,
                2,
            )
            if self._uses_water_budget()
            else None
        )
        self._total_water_used = 0.0
        self._last_reward = self.REWARD_EPSILON
        self._last_irrigation_level = 0
        self._set_rain_signal()
        self._update_state()

        return self._build_observation(
            reward=self._last_reward,
            done=False,
            metadata={
                "message": "Smart irrigation environment ready.",
                "day": self._state.step_count,
                "total_water_used": self._total_water_used,
                "water_remaining": self._water_remaining,
                "rain_probability": self._rain_probability,
                "crop_stress_accumulation": self._crop_stress_accumulation,
            },
        )

    def step(
        self,
        action: SmartIrrigationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SmartIrrigationObservation:
        """Apply one irrigation decision and update the farm state."""
        del timeout_s, kwargs

        self._state.step_count += 1

        requested_irrigation_level = action.irrigation_level
        irrigation_level = self._constrain_irrigation_level(requested_irrigation_level)
        irrigation_added = self.IRRIGATION_EFFECTS[irrigation_level]
        water_cost = self.WATER_COSTS[irrigation_level]
        actual_rainfall = self.RAIN_AMOUNT if self._should_rain() else 0.0
        evaporation = max(0.0, 0.1 * self._temperature - 0.05 * self._humidity)
        crop_usage = 2.0 + 3.0 * self._crop_stage

        new_moisture = (
            self._soil_moisture
            + irrigation_added
            + actual_rainfall
            - evaporation
            - crop_usage
        )
        self._soil_moisture = round(self._clamp(new_moisture, 0.0, 100.0), 2)

        new_humidity = (
            self._humidity
            + self.HUMIDITY_EFFECTS[irrigation_level]
            - (evaporation * 0.5)
        )
        self._humidity = round(self._clamp(new_humidity, 0.0, 100.0), 2)

        if self._water_remaining is not None:
            self._water_remaining = round(
                max(0.0, self._water_remaining - water_cost),
                2,
            )

        self._total_water_used = round(self._total_water_used + irrigation_added, 2)
        self._last_irrigation_level = irrigation_level
        stress_delta = self._compute_crop_stress_delta(action=irrigation_level)
        self._crop_stress_accumulation = round(
            self._clamp(
                self._crop_stress_accumulation + stress_delta,
                0.0,
                100.0,
            ),
            2,
        )

        raw_reward = self._compute_reward(
            moisture=self._soil_moisture,
            action=irrigation_level,
            stress_delta=stress_delta,
        )
        self._last_reward = round(self._normalize_reward(raw_reward), 4)

        done = self._state.step_count >= self.MAX_STEPS
        self._crop_stage = round(
            min(1.0, self._state.step_count / self.MAX_STEPS),
            2,
        )

        self._update_weather()
        self._update_state()

        metadata = {
            "day": self._state.step_count,
            "requested_irrigation_level": requested_irrigation_level,
            "applied_irrigation_level": irrigation_level,
            "action_adjusted": irrigation_level != requested_irrigation_level,
            "irrigation_added": irrigation_added,
            "water_cost": water_cost,
            "actual_rainfall": actual_rainfall,
            "evaporation": round(evaporation, 2),
            "crop_usage": round(crop_usage, 2),
            "crop_stress_delta": round(stress_delta, 2),
            "crop_stress_accumulation": self._crop_stress_accumulation,
            "total_water_used": self._total_water_used,
            "water_remaining": self._water_remaining,
            "raw_reward": round(raw_reward, 2),
            "normalized_reward": self._last_reward,
        }
        if done:
            metadata["final_crop_health"] = round(
                100.0 - self._crop_stress_accumulation,
                2,
            )

        return self._build_observation(
            reward=self._last_reward,
            done=done,
            metadata=metadata,
        )

    @property
    def state(self) -> SmartIrrigationState:
        """Return the hidden environment state."""
        return self._state

    def _compute_reward(self, moisture: float, action: int, stress_delta: float) -> float:
        """Reward that balances crop health, water savings, and scenario goals."""
        reward = 0.0

        if 40.0 <= moisture <= 70.0:
            reward += 10.0
        elif moisture < 40.0:
            reward -= (40.0 - moisture) * 0.5
        else:
            reward -= (moisture - 70.0) * 0.7

        if moisture < 30.0:
            reward -= 10.0

        if moisture > 80.0:
            reward -= 8.0

        reward -= self.WATER_COSTS[action]

        if self._difficulty == "difficult":
            if self._rain_probability > 0.7 and action > 0:
                reward -= 4.0
            elif self._rain_probability > 0.7 and action == 0:
                reward += 4.0
        elif self._rainfall_forecast == 1 and action > 0:
            reward -= 3.0
        elif self._rainfall_forecast == 1 and action == 0:
            reward += 3.0

        if (
            self._uses_water_budget()
            and self._water_remaining is not None
            and self._water_remaining < 10.0
            and moisture > 60.0
        ):
            reward -= 5.0

        reward -= stress_delta * self.STRESS_PENALTY_WEIGHT

        return reward

    def _compute_crop_stress_delta(self, action: int) -> float:
        """Compute the irreversible crop stress added by the current step."""
        stress_delta = 0.0

        moisture = self._soil_moisture
        if 40.0 <= moisture <= 70.0:
            stress_delta += 0.0
        elif 35.0 <= moisture < 40.0 or 70.0 < moisture <= 75.0:
            stress_delta += 0.4
        elif 30.0 <= moisture < 35.0 or 75.0 < moisture <= 80.0:
            stress_delta += 0.9
        elif 20.0 <= moisture < 30.0 or 80.0 < moisture <= 90.0:
            stress_delta += 1.6
        else:
            stress_delta += 2.4

        temperature = self._temperature
        if 22.0 <= temperature <= 32.0:
            stress_delta += 0.0
        elif 20.0 <= temperature < 22.0 or 32.0 < temperature <= 35.0:
            stress_delta += 0.3
        elif 18.0 <= temperature < 20.0 or 35.0 < temperature <= 38.0:
            stress_delta += 0.7
        else:
            stress_delta += 1.2

        humidity = self._humidity
        if 45.0 <= humidity <= 75.0:
            stress_delta += 0.0
        elif 35.0 <= humidity < 45.0 or 75.0 < humidity <= 85.0:
            stress_delta += 0.2
        elif 25.0 <= humidity < 35.0 or 85.0 < humidity <= 95.0:
            stress_delta += 0.5
        else:
            stress_delta += 0.8

        if (self._rain_probability >= 0.7 or self._rainfall_forecast == 1) and action >= 2:
            stress_delta += 0.15
        elif moisture < 35.0 and temperature > 30.0 and action == 0:
            stress_delta += 0.15

        return round(stress_delta, 2)

    def _normalize_reward(self, raw_reward: float) -> float:
        """Map the raw reward into a strict interior range for graders and logs."""
        return self._clamp(
            (raw_reward - self.RAW_REWARD_MIN)
            / (self.RAW_REWARD_MAX - self.RAW_REWARD_MIN),
            self.REWARD_EPSILON,
            1.0 - self.REWARD_EPSILON,
        )

    def _update_weather(self) -> None:
        """Generate the next step's weather values."""
        temperature_shift = self._rng.uniform(-2.0, 2.0)
        humidity_shift = self._rng.uniform(-6.0, 6.0)

        self._temperature = round(
            self._clamp(self._temperature + temperature_shift, 18.0, 40.0),
            2,
        )
        self._humidity = round(
            self._clamp(self._humidity + humidity_shift, 20.0, 100.0),
            2,
        )
        self._set_rain_signal()

    def _build_observation(
        self,
        *,
        reward: float,
        done: bool,
        metadata: dict[str, Any],
    ) -> SmartIrrigationObservation:
        """Create the outward-facing observation payload."""
        return SmartIrrigationObservation(
            soil_moisture=self._soil_moisture,
            temperature=self._temperature,
            humidity=self._humidity,
            rainfall_forecast=self._rainfall_forecast,
            rain_probability=self._rain_probability,
            crop_stage=self._crop_stage,
            crop_stress_accumulation=self._crop_stress_accumulation,
            water_remaining=self._water_remaining,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _resolve_difficulty(self, difficulty: str) -> DifficultyName:
        """Validate the requested scenario difficulty."""
        normalized = difficulty.strip().lower()
        if normalized == "hard":
            normalized = "difficult"
        if normalized not in {"easy", "medium", "difficult"}:
            raise ValueError(
                "difficulty must be one of: easy, medium, hard, difficult."
            )
        return normalized  # type: ignore[return-value]

    def _uses_water_budget(self) -> bool:
        """Return whether the active mode tracks a water budget."""
        return self._difficulty in {"medium", "difficult"}

    def _constrain_irrigation_level(self, irrigation_level: int) -> int:
        """Apply water-budget limits before executing the action."""
        if self._water_remaining is None:
            return irrigation_level

        affordable_levels = [
            level
            for level, water_cost in enumerate(self.WATER_COSTS)
            if water_cost <= self._water_remaining
        ]
        max_affordable_level = affordable_levels[-1] if affordable_levels else 0
        return min(irrigation_level, max_affordable_level)

    def _should_rain(self) -> bool:
        """Sample actual rain from the current rain signal."""
        if self._rain_probability <= 0.0:
            return False
        if self._rain_probability >= 1.0:
            return True
        return self._rng.random() < self._rain_probability

    def _set_rain_signal(self) -> None:
        """Update the forecast representation for the active difficulty."""
        estimated_probability = round(self._estimate_rain_probability(), 2)

        if self._difficulty == "difficult":
            self._rainfall_forecast = None
            self._rain_probability = estimated_probability
            return

        binary_forecast = 1 if self._rng.random() < estimated_probability else 0
        self._rainfall_forecast = binary_forecast
        self._rain_probability = float(binary_forecast)

    def _estimate_rain_probability(self) -> float:
        """Estimate rain likelihood from the current weather conditions."""
        humidity_component = 0.15 + (self._humidity / 100.0) * 0.65
        temperature_penalty = max(0.0, (self._temperature - 30.0) * 0.015)
        weather_noise = self._rng.uniform(-0.15, 0.15)
        return self._clamp(
            humidity_component - temperature_penalty + weather_noise,
            0.05,
            0.95,
        )

    def _update_state(self) -> None:
        """Store the latest internal values inside the OpenEnv state object."""
        self._state.difficulty = self._difficulty
        self._state.soil_moisture = self._soil_moisture
        self._state.temperature = self._temperature
        self._state.humidity = self._humidity
        self._state.rainfall_forecast = self._rainfall_forecast
        self._state.rain_probability = self._rain_probability
        self._state.crop_stage = self._crop_stage
        self._state.crop_stress_accumulation = self._crop_stress_accumulation
        self._state.water_remaining = self._water_remaining
        self._state.total_water_used = self._total_water_used
        self._state.last_reward = self._last_reward
        self._state.last_irrigation_level = self._last_irrigation_level
        self._state.max_steps = self.MAX_STEPS

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        """Clamp a value to a fixed range."""
        return max(lower, min(value, upper))
