# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hospital Scheduler Environment Client."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
try:
    from .models import HospitalAction, HospitalObservation, HospitalState
except ImportError:  # When running from source (e.g. `python inference.py`)
    from models import HospitalAction, HospitalObservation, HospitalState

class HospitalSchedulerEnv(EnvClient[HospitalAction, HospitalObservation, HospitalState]):
    """
    Client for the Hospital Scheduler Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example (from source checkout):
        >>> import asyncio
        >>> from client import HospitalSchedulerEnv
        >>>
        >>> async def demo():
        ...     env = HospitalSchedulerEnv(base_url="http://localhost:8000")
        ...     await env.connect()
        ...     try:
        ...         result = await env.reset()
        ...         print(result.observation.message)
        ...         result = await env.help()
        ...         print(result.observation.message)
        ...     finally:
        ...         await env.close()
        >>> asyncio.run(demo())

    Example with Docker:
        >>> # Automatically start container and connect
        >>> # (requires Docker)
        >>> # client = await HospitalSchedulerEnv.from_docker_image(
        >>> #     "hospital_scheduler-env:latest",
        >>> #     env_vars={"HOSPITAL_TASK": "easy"},
        >>> # )
    """

    def _step_payload(self, action: HospitalAction) -> Dict:
        """
        Convert HospitalAction to JSON payload for step message.

        Args:
            action: HospitalAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {"command": action.command, "parameters": action.parameters}

    async def do(self, command: str, **parameters: Any) -> StepResult[HospitalObservation]:
        """Convenience helper: execute a command without manually building HospitalAction."""

        return await self.step(HospitalAction(command=command, parameters=parameters))

    # ----
    # Convenience command helpers
    # ----

    async def help(self) -> StepResult[HospitalObservation]:
        return await self.do("help")

    async def finalize(self) -> StepResult[HospitalObservation]:
        return await self.do("finalize")

    async def assign_bed(self, patient_id: str, bed_id: str) -> StepResult[HospitalObservation]:
        return await self.do("assign_bed", patient_id=patient_id, bed_id=bed_id)

    async def unassign_bed(self, patient_id: str) -> StepResult[HospitalObservation]:
        return await self.do("unassign_bed", patient_id=patient_id)

    async def assign_shift(
        self, nurse_id: str, ward_id: str, day: int
    ) -> StepResult[HospitalObservation]:
        return await self.do("assign_shift", nurse_id=nurse_id, ward_id=ward_id, day=day)

    async def remove_shift(
        self, nurse_id: str, ward_id: str, day: int
    ) -> StepResult[HospitalObservation]:
        return await self.do("remove_shift", nurse_id=nurse_id, ward_id=ward_id, day=day)

    async def triage(self, patient_id: str, label: str) -> StepResult[HospitalObservation]:
        return await self.do("triage", patient_id=patient_id, label=label)

    async def discharge(self, patient_id: str) -> StepResult[HospitalObservation]:
        return await self.do("discharge", patient_id=patient_id)

    async def call_in(self, nurse_id: str) -> StepResult[HospitalObservation]:
        return await self.do("call_in", nurse_id=nurse_id)

    async def assign_nurse(self, nurse_id: str, ward_id: str) -> StepResult[HospitalObservation]:
        return await self.do("assign_nurse", nurse_id=nurse_id, ward_id=ward_id)

    def _parse_result(self, payload: Dict) -> StepResult[HospitalObservation]:
        """
        Parse server response into StepResult[HospitalObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HospitalObservation
        """
        obs_data = payload.get("observation", {})
        observation = HospitalObservation(
            message=obs_data.get("message", ""),
            task=obs_data.get("task", "easy"),
            progress=obs_data.get("progress", 0.0),
            violations=obs_data.get("violations", []),
            snapshot=obs_data.get("snapshot", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HospitalState:
        """
        Parse server response into HospitalState object.

        Args:
            payload: JSON response from state request

        Returns:
            HospitalState object
        """
        return HospitalState(**payload)


class HospitalSchedulerSync:
    """Synchronous wrapper around HospitalSchedulerEnv for simple scripts.

    This is optional: the underlying OpenEnv client is async. Use this wrapper
    when you're new to async and just want a blocking API.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url
        self._env: Optional[HospitalSchedulerEnv] = None

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "HospitalSchedulerSync cannot run inside an existing event loop. "
            "Use HospitalSchedulerEnv (async) instead."
        )

    def connect(self) -> None:
        self._env = HospitalSchedulerEnv(base_url=self._base_url)
        self._run(self._env.connect())

    def reset(self):
        assert self._env is not None
        return self._run(self._env.reset())

    def step(self, action: HospitalAction):
        assert self._env is not None
        return self._run(self._env.step(action))

    def state(self) -> HospitalState:
        assert self._env is not None
        return self._run(self._env.state())

    def close(self) -> None:
        if self._env is None:
            return
        self._run(self._env.close())
        self._env = None

    def __enter__(self) -> "HospitalSchedulerSync":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
