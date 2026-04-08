"""
Inference script for the Smart Irrigation environment.

Required environment variables:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional environment variables:
- IMAGE_NAME or LOCAL_IMAGE_NAME for Docker-based startup
- ENV_BASE_URL to connect to an already-running environment server
- DIFFICULTY to choose easy, medium, hard, or difficult
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from .client import SmartIrrigationEnv
    from .models import SmartIrrigationAction, SmartIrrigationObservation
except ImportError:
    from client import SmartIrrigationEnv
    from models import SmartIrrigationAction, SmartIrrigationObservation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
TASK_NAME = os.getenv("TASK_NAME", "smart-irrigation")
BENCHMARK = os.getenv("BENCHMARK", "smart-irrigation")
REQUESTED_DIFFICULTY = os.getenv("DIFFICULTY", os.getenv("SCENARIO", "easy"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.0"))
ACTION_WATER_COSTS = [0, 1, 2, 3]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control irrigation for a smart farm.
    Choose exactly one irrigation level:
    0 = no irrigation
    1 = low water
    2 = medium water
    3 = high water

    Goal:
    - keep soil moisture roughly in the 40 to 70 range
    - avoid wasting water
    - preserve water budget when it exists
    - avoid irrigating when rain is likely

    Reply with only one digit: 0, 1, 2, or 3.
    """
).strip()


def normalize_difficulty(difficulty: str) -> str:
    """Accept common aliases and return the canonical server difficulty."""
    difficulty_aliases = {
        "easy": "easy",
        "medium": "medium",
        "hard": "difficult",
        "difficult": "difficult",
    }
    normalized = difficulty.strip().lower()
    if normalized not in difficulty_aliases:
        allowed = ", ".join(sorted(difficulty_aliases))
        raise ValueError(f"DIFFICULTY must be one of: {allowed}.")
    return difficulty_aliases[normalized]


DIFFICULTY = normalize_difficulty(REQUESTED_DIFFICULTY)


def log_start(task: str, env: str, model: str, difficulty: str) -> None:
    print(
        f"[START] task={task} env={env} model={model} difficulty={difficulty}",
        flush=True,
    )


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_value}",
        flush=True,
    )


def max_available_action(observation: SmartIrrigationObservation) -> int:
    """Return the highest irrigation level allowed by the current water budget."""
    if observation.water_remaining is None:
        return 3

    for irrigation_level in range(3, -1, -1):
        if ACTION_WATER_COSTS[irrigation_level] <= observation.water_remaining:
            return irrigation_level
    return 0


def safe_action(
    irrigation_level: int, observation: SmartIrrigationObservation
) -> int:
    """Clamp model output so it respects the active water budget."""
    return min(irrigation_level, max_available_action(observation))


def heuristic_action(observation: SmartIrrigationObservation) -> int:
    """Fallback policy when an LLM call is unavailable or invalid."""
    available_action = max_available_action(observation)
    rain_probability = observation.rain_probability

    if available_action == 0:
        return 0

    if rain_probability > 0.7:
        if observation.soil_moisture < 30:
            return min(1, available_action)
        return 0

    if observation.soil_moisture < 25:
        return min(3, available_action)
    if observation.soil_moisture < 40:
        return min(2, available_action)
    if observation.soil_moisture < 50 and observation.crop_stage > 0.6:
        return min(1, available_action)
    return 0


def build_user_prompt(step: int, observation: SmartIrrigationObservation) -> str:
    water_remaining = (
        f"{observation.water_remaining:.2f}"
        if observation.water_remaining is not None
        else "not used in this mode"
    )
    rainfall_forecast = (
        str(observation.rainfall_forecast)
        if observation.rainfall_forecast is not None
        else "probabilistic mode"
    )
    return textwrap.dedent(
        f"""
        Difficulty: {observation.difficulty}
        Step: {step}
        Soil moisture: {observation.soil_moisture:.2f}
        Temperature: {observation.temperature:.2f}
        Humidity: {observation.humidity:.2f}
        Rain probability: {observation.rain_probability:.2f}
        Rain forecast: {rainfall_forecast}
        Crop stage: {observation.crop_stage:.2f}
        Water remaining: {water_remaining}
        Choose the next irrigation level.
        """
    ).strip()


def choose_action(
    client: OpenAI, step: int, observation: SmartIrrigationObservation
) -> int:
    """Ask the model for an irrigation decision, then fall back safely if needed."""
    if not HF_TOKEN:
        return safe_action(heuristic_action(observation), observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(step, observation)},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        content = (completion.choices[0].message.content or "").strip()
        for char in content:
            if char in {"0", "1", "2", "3"}:
                return safe_action(int(char), observation)
    except Exception:
        pass

    return safe_action(heuristic_action(observation), observation)


async def create_environment() -> SmartIrrigationEnv:
    """Connect to a running environment or start one from a Docker image."""
    if ENV_BASE_URL:
        env = SmartIrrigationEnv(base_url=ENV_BASE_URL)
        await env.connect()
        return env

    image_name = IMAGE_NAME or "smart-irrigation:latest"
    return await SmartIrrigationEnv.from_docker_image(image_name)


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-token")
    env = await create_environment()

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME, difficulty=DIFFICULTY)

    try:
        result = await env.reset(difficulty=DIFFICULTY)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            irrigation_level = choose_action(llm_client, step, result.observation)
            action = SmartIrrigationAction(irrigation_level=irrigation_level)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"irrigation_level={irrigation_level}",
                reward=reward,
                done=result.done,
                error=None,
            )

            if result.done:
                break

        success = sum(rewards) >= SUCCESS_THRESHOLD
    finally:
        try:
            await env.close()
        finally:
            log_end(success=success, steps=steps_taken, rewards=rewards)


def run() -> None:
    """Synchronous entry point for direct script and console execution."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
