# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Hospital Bed & Staff Scheduler environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class HospitalAction(Action):
    """Generic command-style action.

    Using a command + parameters pattern keeps the API stable while the
    environment supports multiple tasks (easy/medium/hard).
    """

    command: str = Field(
        ...,
        description=(
            "Command to execute (e.g. 'assign_bed', 'assign_shift', 'triage')."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command parameters as a JSON object.",
    )


class HospitalObservation(Observation):
    """Observation containing a structured snapshot and progress signals."""

    message: str = Field(default="", description="Human-readable result message")
    task: str = Field(default="easy", description="Active task name")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current task score/progress in [0, 1]",
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of rule violations or action errors",
    )
    snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured state snapshot useful for agents",
    )


class HospitalState(State):
    """Serializable environment state (queried via the /state endpoint)."""

    task: str = Field(default="easy", description="Active task name")
    seed: Optional[int] = Field(default=None, description="Random seed used for this episode (None = random)")
    max_steps: int = Field(default=60, description="Episode step limit")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Current score")
    last_action_error: Optional[str] = Field(
        default=None,
        description="Last action error message (or null)",
    )

    # Domain entities (kept JSON-serializable for web UI + graders)
    wards: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    beds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    patients: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    nurses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Assignments
    bed_assignments: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping patient_id -> bed_id",
    )
    shift_assignments: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="Mapping day -> ward_id -> list[nurse_id]",
    )
    nurse_ward_assignments: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping nurse_id -> ward_id (hard task)",
    )

    # Hard-task fields
    triage: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping patient_id -> triage label",
    )
    on_call_nurses: List[str] = Field(default_factory=list)
    available_nurses: List[str] = Field(default_factory=list)
