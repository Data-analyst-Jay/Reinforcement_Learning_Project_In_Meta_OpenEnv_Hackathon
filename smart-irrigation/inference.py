"""
Inference script for the Smart Irrigation environment.

Required environment variables:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional environment variables:
- IMAGE_NAME or LOCAL_IMAGE_NAME for Docker-based startup
- ENV_BASE_URL to connect to an already-running environment server

This script always evaluates the agent across all three task difficulties
in order: easy, medium, and hard (canonical server name: difficult).
"""

from __future__ import annotations

import asyncio
import math
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
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.80"))
SCORE_SIGMOID_SCALE = float(os.getenv("SCORE_SIGMOID_SCALE", "3.0"))
SCORE_EPSILON = 1e-6
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


EPISODE_DIFFICULTIES = [
    normalize_difficulty(difficulty) for difficulty in ("easy", "medium", "hard")
]


def log_start(task: str, env: str, model: str, difficulty: str) -> None:
    print(
        f"[START] task={task} env={env} model={model} difficulty={difficulty}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    crop_component: float,
    decision_component: float,
    total_reward: float,
    done: bool,
) -> None:
    print(
        f"[STEP] step={step} action={action} crop_component={crop_component:.2f} decision_component={decision_component:.2f} total_reward={total_reward:.2f} done={str(done).lower()}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    total_rewards: List[float],
    crop_health: Optional[float] = None,
) -> None:
    total_rewards_value = ",".join(
        f"{total_reward:.2f}" for total_reward in total_rewards
    )
    crop_health_value = (
        f"{crop_health:.2f}" if crop_health is not None else "null"
    )
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} total_rewards={total_rewards_value} crop_health={crop_health_value}",
        flush=True,
    )


def episode_score(total_rewards: List[float]) -> float:
    """Map mean total reward into a human-friendly score in the (0, 1) range."""
    if not total_rewards:
        return 0.5

    raw_score = sum(total_rewards) / len(total_rewards)
    scaled_score = raw_score / SCORE_SIGMOID_SCALE
    if scaled_score >= 0.0:
        exp_term = math.exp(-scaled_score)
        score = 1.0 / (1.0 + exp_term)
    else:
        exp_term = math.exp(scaled_score)
        score = exp_term / (1.0 + exp_term)
    return min(max(score, SCORE_EPSILON), 1.0 - SCORE_EPSILON)


def episode_succeeds(score: float) -> bool:
    """Treat success as a threshold over the final task score."""
    return score >= SUCCESS_THRESHOLD


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


async def run_episode(
    env: SmartIrrigationEnv,
    llm_client: OpenAI,
    difficulty: str,
) -> None:
    """Run one full task episode and emit a complete START/STEP/END log block."""
    total_rewards: List[float] = []
    steps_taken = 0
    score = episode_score(total_rewards)
    success = False
    crop_health: Optional[float] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME, difficulty=difficulty)

    try:
        result = await env.reset(difficulty=difficulty)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            irrigation_level = choose_action(llm_client, step, result.observation)
            action = SmartIrrigationAction(irrigation_level=irrigation_level)
            result = await env.step(action)

            total_reward = float(result.observation.total_reward)
            total_rewards.append(total_reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"irrigation_level={irrigation_level}",
                crop_component=float(result.observation.crop_component),
                decision_component=float(result.observation.decision_component),
                total_reward=total_reward,
                done=result.done,
            )

            if result.done:
                break

        score = episode_score(total_rewards)
        success = episode_succeeds(score)
        final_observation = result.observation
        crop_health = final_observation.metadata.get("final_crop_health")
        if crop_health is None:
            crop_health = round(
                100.0 - final_observation.crop_stress_accumulation,
                2,
            )
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            total_rewards=total_rewards,
            crop_health=crop_health,
        )


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing-token")
    env = await create_environment()

    try:
        for difficulty in EPISODE_DIFFICULTIES:
            await run_episode(env=env, llm_client=llm_client, difficulty=difficulty)
    finally:
        await env.close()


def run() -> None:
    """Synchronous entry point for direct script and console execution."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
