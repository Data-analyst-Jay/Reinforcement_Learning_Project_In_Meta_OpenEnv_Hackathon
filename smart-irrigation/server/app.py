# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Smart Irrigation environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SmartIrrigationAction, SmartIrrigationObservation
    from .smart_irrigation_environment import SmartIrrigationEnvironment
except ImportError:
    from models import SmartIrrigationAction, SmartIrrigationObservation
    from server.smart_irrigation_environment import SmartIrrigationEnvironment


app = create_app(
    SmartIrrigationEnvironment,
    SmartIrrigationAction,
    SmartIrrigationObservation,
    env_name="smart-irrigation",
    max_concurrent_envs=1,
)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server with the provided network settings."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn server.app:app --workers 4
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
