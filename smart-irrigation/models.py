# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Irrigation environment.

This file defines what the agent can do and what it can observe.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SmartIrrigationAction(Action):
    """One irrigation decision made by the controller."""

    irrigation_level: int = Field(
        ...,
        ge=0,
        le=3,
        description="Irrigation decision: 0=no irrigation, 1=low, 2=medium, 3=high.",
    )


class SmartIrrigationObservation(Observation):
    """Current farm condition visible to the agent after each step."""

    soil_moisture: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Current soil moisture on a 0 to 100 scale.",
    )
    temperature: float = Field(
        default=25.0,
        description="Current temperature in degrees Celsius.",
    )
    humidity: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Current air humidity in percent.",
    )
    rainfall_forecast: int | None = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary rain forecast used in easy and medium modes.",
    )
    rain_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of rain on the next step.",
    )
    crop_stage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Crop growth stage from 0.0 (early) to 1.0 (late).",
    )
    crop_stress_accumulation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Irreversible accumulated crop stress on a 0 to 100 scale.",
    )
    water_remaining: float | None = Field(
        default=None,
        ge=0.0,
        description="Remaining irrigation budget for medium and difficult modes.",
    )
    crop_component: float = Field(
        default=0.0,
        description="Signed crop-safety reward component from the most recent step.",
    )
    decision_component: float = Field(
        default=0.0,
        description="Signed water-decision reward component from the most recent step.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Signed total reward from the most recent step.",
    )


class SmartIrrigationState(State):
    """Full internal state for the smart irrigation environment."""

    difficulty: Literal["easy", "medium", "difficult"] = Field(
        default="easy",
        description="Active irrigation scenario difficulty.",
    )
    soil_moisture: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Current soil moisture on a 0 to 100 scale.",
    )
    temperature: float = Field(
        default=25.0,
        description="Current temperature in degrees Celsius.",
    )
    humidity: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Current air humidity in percent.",
    )
    rainfall_forecast: int | None = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary rain forecast used in easy and medium modes.",
    )
    rain_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of rain on the next step.",
    )
    crop_stage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Crop growth stage from 0.0 (early) to 1.0 (late).",
    )
    crop_stress_accumulation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Irreversible accumulated crop stress on a 0 to 100 scale.",
    )
    water_remaining: float | None = Field(
        default=None,
        ge=0.0,
        description="Remaining irrigation budget for medium and difficult modes.",
    )
    total_water_used: float = Field(
        default=0.0,
        ge=0.0,
        description="Total irrigation water used in the current episode.",
    )
    last_reward: float = Field(
        default=0.0,
        description="Compatibility alias for the most recent step's total_reward.",
    )
    last_crop_component: float = Field(
        default=0.0,
        description="Most recent signed crop-safety reward component.",
    )
    last_decision_component: float = Field(
        default=0.0,
        description="Most recent signed water-decision reward component.",
    )
    last_total_reward: float = Field(
        default=0.0,
        description="Most recent signed total reward.",
    )
    last_irrigation_level: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Most recent irrigation action.",
    )
    max_steps: int = Field(
        default=5,
        ge=1,
        description="Maximum number of steps in one episode.",
    )
