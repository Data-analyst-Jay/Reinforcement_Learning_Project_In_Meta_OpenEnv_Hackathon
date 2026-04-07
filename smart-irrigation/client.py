# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Irrigation environment client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    SmartIrrigationAction,
    SmartIrrigationObservation,
    SmartIrrigationState,
)


class SmartIrrigationEnv(
    EnvClient[
        SmartIrrigationAction,
        SmartIrrigationObservation,
        SmartIrrigationState,
    ]
):
    """
    Client for the Smart Irrigation environment.

    This client talks to the OpenEnv server over WebSocket and converts
    raw JSON messages into typed irrigation action, observation, and state objects.
    """

    async def reset(
        self,
        difficulty: str = "easy",
        scenario: Optional[str] = None,
        **kwargs: Any,
    ) -> StepResult[SmartIrrigationObservation]:
        """Reset the environment and optionally choose a difficulty mode."""
        reset_kwargs: Dict[str, Any] = dict(kwargs)
        reset_kwargs["difficulty"] = difficulty
        if scenario is not None:
            reset_kwargs["scenario"] = scenario
        return await super().reset(**reset_kwargs)

    def _step_payload(self, action: SmartIrrigationAction) -> Dict[str, Any]:
        """Convert a smart irrigation action into the server payload."""
        return {
            "irrigation_level": action.irrigation_level,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SmartIrrigationObservation]:
        """Convert a server response into a typed irrigation observation."""
        obs_data = payload.get("observation", {})
        observation = SmartIrrigationObservation(
            difficulty=obs_data.get("difficulty", "easy"),
            soil_moisture=obs_data.get("soil_moisture", 50.0),
            temperature=obs_data.get("temperature", 25.0),
            humidity=obs_data.get("humidity", 50.0),
            rainfall_forecast=obs_data.get("rainfall_forecast"),
            rain_probability=obs_data.get(
                "rain_probability",
                float(obs_data.get("rainfall_forecast", 0) or 0.0),
            ),
            crop_stage=obs_data.get("crop_stage", 0.0),
            water_remaining=obs_data.get("water_remaining"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SmartIrrigationState:
        """Convert the raw state response into a typed irrigation state."""
        return SmartIrrigationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            soil_moisture=payload.get("soil_moisture", 50.0),
            temperature=payload.get("temperature", 25.0),
            humidity=payload.get("humidity", 50.0),
            rainfall_forecast=payload.get("rainfall_forecast"),
            rain_probability=payload.get(
                "rain_probability",
                float(payload.get("rainfall_forecast", 0) or 0.0),
            ),
            crop_stage=payload.get("crop_stage", 0.0),
            water_remaining=payload.get("water_remaining"),
            total_water_used=payload.get("total_water_used", 0.0),
            last_reward=payload.get("last_reward", 0.0),
            last_irrigation_level=payload.get("last_irrigation_level", 0),
            max_steps=payload.get("max_steps", 20),
        )
