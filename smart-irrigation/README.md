---
title: Smart Irrigation Environment Server
  - openenv
  - reinforcement-learning
  - agriculture
---

# Smart Irrigation Environment

This project is an OpenEnv environment for a smart irrigation task.

The agent acts like an irrigation controller. On each step it chooses one
irrigation level. The environment then updates soil moisture, humidity,
temperature, rain conditions, crop stage, and an optional water budget. The
environment supports three toggleable scenarios named `easy`, `medium`, and
`difficult`. The inference client also accepts `hard` and normalizes it to the
canonical `difficult` mode expected by the server.

## Main Components

- `SmartIrrigationAction`: the agent chooses `irrigation_level` from `0` to `3`
- `SmartIrrigationObservation`: the visible farm condition after each step
- `SmartIrrigationState`: the full hidden environment state
- `SmartIrrigationEnvironment`: the core farm simulation logic
- `SmartIrrigationEnv`: the client used to interact with the server

## State Features

- `difficulty`
- `soil_moisture`
- `temperature`
- `humidity`
- `rainfall_forecast`
- `rain_probability`
- `crop_stage`
- `water_remaining`

## Action Space

- `0`: no irrigation
- `1`: low irrigation
- `2`: medium irrigation
- `3`: high irrigation

## Difficulty Modes

- `easy`: base crop-health control with binary rain forecast
- `medium`: adds a fixed episode water budget and water-saving pressure
- `difficult`: keeps the water budget and replaces fixed rain with probabilistic rain

## Reward Logic

The reward:
- gives a positive score when soil moisture stays in the healthy range
- penalizes under-watering and over-watering
- penalizes unnecessary water usage
- penalizes irrigating when rain is forecast or highly probable
- adds a water-budget penalty in the `medium` and `difficult` scenarios

All rewards are linearly normalized into the `0.0` to `1.0` range before
being returned to the agent.

## Run Locally

Start the API server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Run the inference script in the same package-oriented style used by
submission tooling:

```bash
uv run --project . python -m smart_irrigation.inference
```

You can also run the file directly for local debugging:

```bash
python inference.py
```

Choose a scenario:

```powershell
$env:DIFFICULTY = "medium"
uv run --project . python -m smart_irrigation.inference
```

Accepted difficulty values: `easy`, `medium`, `hard or difficult`.

Optional script entry points after `uv sync`:

```bash
uv run --project . inference
uv run --project . server
```

## Build Docker Image

```bash
docker build -t smart-irrigation:latest .
```

## Submission Checklist Notes

Verified in this repository:

- `inference.py` exists at the project root.
- The inference client uses the OpenAI client.
- Structured `[START]`, `[STEP]`, and `[END]` logs are emitted.
- Rewards are normalized to the `0.0` to `1.0` range.
- `openenv.yaml` and `Dockerfile` are present.
- HF Space deployment returning HTTP 200 and responding to `reset()`.
- Docker build in the submission environment.
- Pre-submission validator run.
- Hosted grader runs across all tasks.

## Project Structure

```text
smart-irrigation/
|-- __init__.py
|-- client.py
|-- Dockerfile
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
|-- uv.lock
`-- server/
    |-- __init__.py
    |-- app.py
    |-- requirements.txt
    `-- smart_irrigation_environment.py
```
## Baseline Scores
- Steps run: 20
- Model Used for testing: Qwen/Qwen2.5-72B-Instruct
- For easy task: [0.91,0.98,0.91,0.75,0.64,0.50,0.44,0.42,0.42,0.36,0.35,0.35,0.29,0.27,0.27,0.32,0.33,0.34,0.40,0.46]
- For medium task: [0.91,0.91,0.91,0.91,0.98,0.98,0.91,0.91,0.91,0.71,0.69,0.71,0.63,0.62,0.60,0.49,0.45,0.46,0.67,0.70]
- For difficult task: [0.91,0.91,0.71,0.43,0.38,0.41,0.45,0.64,0.37,0.34,0.40,0.47,0.68,0.42,0.65,0.39,0.62,0.45,0.67,0.93]