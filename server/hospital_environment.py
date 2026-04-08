"""Hospital Bed & Staff Scheduler environment implementation.

Implements three tasks with increasing difficulty:
- easy: assign patients to beds (no conflicts)
- medium: schedule nurse coverage across a week (labor rules)
- hard: mass casualty event (triage + reallocation + call-ins)

The environment is deterministic: the same actions lead to the same score.
"""

from __future__ import annotations

import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..graders import grade
    from ..models import HospitalAction, HospitalObservation, HospitalState
except ImportError:  # When running as a top-level module (uvicorn server.app:app)
    from graders import grade
    from models import HospitalAction, HospitalObservation, HospitalState


class HospitalSchedulerEnvironment(Environment):
    """Hospital Scheduler environment with 3 tasks and deterministic graders."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = self._build_state(task=os.getenv("HOSPITAL_TASK", "easy"))

    def reset(self) -> HospitalObservation:
        task = os.getenv("HOSPITAL_TASK", self._state.task)
        self._state = self._build_state(task=task)
        self._state.score = grade(self._state)

        return self._observation(
            message=f"Hospital Scheduler ready (task={self._state.task}).",
            violations=[],
            done=False,
            reward=0.0,
        )

    def step(self, action: HospitalAction) -> HospitalObservation:  # type: ignore[override]
        prev_score = float(self._state.score)
        self._state.last_action_error = None

        # If episode already reached max steps, freeze and return.
        if self._state.step_count >= self._state.max_steps:
            return self._observation(
                message="Episode step limit reached.",
                violations=[],
                done=True,
                reward=0.0,
            )

        cmd = (action.command or "").strip()
        params = action.parameters or {}

        violations: list[str] = []
        message = ""

        # Count step for all commands except 'help' and 'status'
        if cmd not in {"help", "status"}:
            self._state.step_count += 1

        if cmd == "help":
            message = self._help_text()
        elif cmd == "status":
            message = "OK"
        elif cmd == "finalize":
            message = "Finalizing episode."
        else:
            # Task-specific commands
            if self._state.task == "easy":
                message, violations = self._step_easy(cmd, params)
            elif self._state.task == "medium":
                message, violations = self._step_medium(cmd, params)
            elif self._state.task == "hard":
                message, violations = self._step_hard(cmd, params)
            else:
                violations = [f"Unknown task '{self._state.task}'."]
                message = "Invalid task configuration."

        if violations:
            self._state.last_action_error = "; ".join(violations)

        # Recompute score deterministically.
        self._state.score = grade(self._state)

        # Dense reward = delta score + small time penalty.
        reward = (self._state.score - prev_score) - 0.01

        done = False
        if cmd == "finalize":
            done = True
        elif self._state.score >= 0.999:
            done = True
        elif self._state.step_count >= self._state.max_steps:
            done = True

        return self._observation(
            message=message,
            violations=violations,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> HospitalState:
        return self._state

    # -----------------
    # Task initialization
    # -----------------

    def _build_state(self, task: str) -> HospitalState:
        task = (task or "easy").strip().lower()
        if task not in {"easy", "medium", "hard"}:
            task = "easy"

        base = HospitalState(
            episode_id=str(uuid4()),
            step_count=0,
            task=task,
            max_steps={"easy": 40, "medium": 80, "hard": 120}[task],
            score=0.0,
            last_action_error=None,
        )

        if task == "easy":
            return self._init_easy(base)
        if task == "medium":
            return self._init_medium(base)
        return self._init_hard(base)

    def _init_easy(self, state: HospitalState) -> HospitalState:
        state.wards = {
            "A": {"name": "Ward A"},
            "B": {"name": "Ward B"},
            "C": {"name": "Ward C"},
        }
        state.beds = {
            **{f"A{i}": {"ward": "A"} for i in range(1, 5)},
            **{f"B{i}": {"ward": "B"} for i in range(1, 5)},
            **{f"C{i}": {"ward": "C"} for i in range(1, 5)},
        }

        # 10 patients, each with a required ward. Deterministic.
        state.patients = {
            "P1": {"required_ward": "A", "acuity": 2},
            "P2": {"required_ward": "A", "acuity": 1},
            "P3": {"required_ward": "A", "acuity": 3},
            "P4": {"required_ward": "B", "acuity": 2},
            "P5": {"required_ward": "B", "acuity": 1},
            "P6": {"required_ward": "B", "acuity": 2},
            "P7": {"required_ward": "C", "acuity": 1},
            "P8": {"required_ward": "C", "acuity": 2},
            "P9": {"required_ward": "C", "acuity": 3},
            "P10": {"required_ward": "A", "acuity": 1},
        }

        state.bed_assignments = {}
        state.shift_assignments = {}
        state.nurses = {}
        state.triage = {}
        state.on_call_nurses = []
        state.available_nurses = []
        state.nurse_ward_assignments = {}
        return state

    def _init_medium(self, state: HospitalState) -> HospitalState:
        # 3 wards with daily demand across 7 days. Demand is stable and deterministic.
        state.wards = {
            "A": {"name": "Ward A", "demand": [1, 2, 2, 1, 2, 1, 1]},
            "B": {"name": "Ward B", "demand": [2, 2, 1, 2, 2, 1, 2]},
            "C": {"name": "Ward C", "demand": [1, 1, 2, 2, 1, 2, 1]},
        }

        state.nurses = {
            **{f"N{i}": {"name": f"Nurse {i}"} for i in range(1, 9)}
        }

        # Start with empty assignments.
        state.shift_assignments = {str(day): {"A": [], "B": [], "C": []} for day in range(7)}

        # Not used in this task but kept for consistent state schema.
        state.patients = {}
        state.beds = {}
        state.bed_assignments = {}
        state.triage = {}
        state.on_call_nurses = []
        state.available_nurses = []
        state.nurse_ward_assignments = {}
        return state

    def _init_hard(self, state: HospitalState) -> HospitalState:
        state.wards = {
            "ED": {"name": "Emergency"},
            "ICU": {"name": "ICU"},
            "GEN": {"name": "General"},
        }

        state.beds = {
            **{f"ED{i}": {"ward": "ED"} for i in range(1, 5)},
            **{f"ICU{i}": {"ward": "ICU"} for i in range(1, 5)},
            **{f"G{i}": {"ward": "GEN"} for i in range(1, 7)},
        }

        # Existing patients (already occupying some beds)
        existing = {
            "E1": {"is_casualty": False, "acuity": 1, "required_ward": "GEN"},
            "E2": {"is_casualty": False, "acuity": 2, "required_ward": "GEN"},
            "E3": {"is_casualty": False, "acuity": 2, "required_ward": "ICU"},
            "E4": {"is_casualty": False, "acuity": 3, "required_ward": "ICU"},
        }

        # Incoming casualties: injury_score drives ground-truth triage.
        casualties = {
            "C1": {"is_casualty": True, "injury_score": 9, "symptoms": "unconscious, low BP"},
            "C2": {"is_casualty": True, "injury_score": 7, "symptoms": "fracture, bleeding controlled"},
            "C3": {"is_casualty": True, "injury_score": 5, "symptoms": "chest pain, stable vitals"},
            "C4": {"is_casualty": True, "injury_score": 4, "symptoms": "minor lacerations"},
            "C5": {"is_casualty": True, "injury_score": 8, "symptoms": "respiratory distress"},
            "C6": {"is_casualty": True, "injury_score": 6, "symptoms": "burns, moderate pain"},
        }

        state.patients = {**existing, **casualties}

        # Start occupancy
        state.bed_assignments = {
            "E1": "G1",
            "E2": "G2",
            "E3": "ICU1",
            "E4": "ICU2",
        }

        # Nurses: 1 on-duty per ward + 3 on-call.
        state.nurses = {
            "D1": {"name": "Day Nurse 1"},
            "D2": {"name": "Day Nurse 2"},
            "D3": {"name": "Day Nurse 3"},
            "OC1": {"name": "On-call 1"},
            "OC2": {"name": "On-call 2"},
            "OC3": {"name": "On-call 3"},
        }
        state.available_nurses = ["D1", "D2", "D3"]
        state.on_call_nurses = ["OC1", "OC2", "OC3"]

        state.nurse_ward_assignments = {
            "D1": "ED",
            "D2": "ICU",
            "D3": "GEN",
        }

        state.shift_assignments = {}
        state.triage = {}
        return state

    # -----------------
    # Task step handlers
    # -----------------

    def _step_easy(self, cmd: str, params: dict) -> tuple[str, list[str]]:
        violations: list[str] = []
        if cmd == "assign_bed":
            patient_id = str(params.get("patient_id", ""))
            bed_id = str(params.get("bed_id", ""))
            if patient_id not in self._state.patients:
                violations.append("Unknown patient_id")
            if bed_id not in self._state.beds:
                violations.append("Unknown bed_id")
            if not violations:
                self._state.bed_assignments[patient_id] = bed_id
                return f"Assigned {patient_id} -> {bed_id}", []
            return "Invalid assign_bed", violations

        if cmd == "unassign_bed":
            patient_id = str(params.get("patient_id", ""))
            if patient_id in self._state.bed_assignments:
                self._state.bed_assignments.pop(patient_id, None)
                return f"Unassigned bed for {patient_id}", []
            return "Nothing to unassign", []

        return "Unknown command for easy task", [f"Unknown command '{cmd}'"]

    def _step_medium(self, cmd: str, params: dict) -> tuple[str, list[str]]:
        violations: list[str] = []

        def _day_key() -> str:
            day = params.get("day")
            try:
                d = int(day)
            except Exception:
                return ""
            return str(d) if 0 <= d <= 6 else ""

        if cmd == "assign_shift":
            nurse_id = str(params.get("nurse_id", ""))
            ward_id = str(params.get("ward_id", ""))
            day_key = _day_key()

            if nurse_id not in self._state.nurses:
                violations.append("Unknown nurse_id")
            if ward_id not in self._state.wards:
                violations.append("Unknown ward_id")
            if day_key == "":
                violations.append("Invalid day (expected 0-6)")

            if violations:
                return "Invalid assign_shift", violations

            day_map = self._state.shift_assignments.setdefault(day_key, {})
            day_map.setdefault(ward_id, [])

            # Enforce: a nurse can only be assigned to one ward per day.
            for w, ns in day_map.items():
                if nurse_id in ns and w != ward_id:
                    violations.append("Nurse already assigned on this day")
                    return "Rule violation", violations

            if nurse_id not in day_map[ward_id]:
                day_map[ward_id].append(nurse_id)

            return f"Assigned {nurse_id} to {ward_id} on day {day_key}", []

        if cmd == "remove_shift":
            nurse_id = str(params.get("nurse_id", ""))
            ward_id = str(params.get("ward_id", ""))
            day_key = _day_key()
            day_map = self._state.shift_assignments.get(day_key, {})
            ward_list = day_map.get(ward_id, [])
            if nurse_id in ward_list:
                ward_list.remove(nurse_id)
                return f"Removed {nurse_id} from {ward_id} on day {day_key}", []
            return "Nothing to remove", []

        return "Unknown command for medium task", [f"Unknown command '{cmd}'"]

    def _step_hard(self, cmd: str, params: dict) -> tuple[str, list[str]]:
        violations: list[str] = []

        if cmd == "triage":
            patient_id = str(params.get("patient_id", ""))
            label = str(params.get("label", "")).lower()
            if patient_id not in self._state.patients:
                violations.append("Unknown patient_id")
            if not self._state.patients.get(patient_id, {}).get("is_casualty", False):
                violations.append("Patient is not a casualty")
            if label not in {"red", "yellow", "green"}:
                violations.append("Invalid label (red/yellow/green)")
            if not violations:
                self._state.triage[patient_id] = label
                return f"Triage set for {patient_id}: {label}", []
            return "Invalid triage", violations

        if cmd == "assign_bed":
            patient_id = str(params.get("patient_id", ""))
            bed_id = str(params.get("bed_id", ""))
            if patient_id not in self._state.patients:
                violations.append("Unknown patient_id")
            if bed_id not in self._state.beds:
                violations.append("Unknown bed_id")
            if not violations:
                self._state.bed_assignments[patient_id] = bed_id
                return f"Assigned {patient_id} -> {bed_id}", []
            return "Invalid assign_bed", violations

        if cmd == "discharge":
            patient_id = str(params.get("patient_id", ""))
            if patient_id not in self._state.patients:
                return "Invalid discharge", ["Unknown patient_id"]
            if self._state.patients[patient_id].get("is_casualty", False):
                return "Invalid discharge", ["Cannot discharge casualty"]
            # Allow discharge only for low acuity (<=2)
            if float(self._state.patients[patient_id].get("acuity", 99)) > 2:
                return "Invalid discharge", ["Patient too high acuity to discharge"]

            self._state.bed_assignments.pop(patient_id, None)
            self._state.patients[patient_id]["discharged"] = True
            return f"Discharged {patient_id}", []

        if cmd == "call_in":
            nurse_id = str(params.get("nurse_id", ""))
            if nurse_id not in self._state.on_call_nurses:
                return "Invalid call_in", ["Nurse not on-call"]
            self._state.on_call_nurses.remove(nurse_id)
            if nurse_id not in self._state.available_nurses:
                self._state.available_nurses.append(nurse_id)
            return f"Called in {nurse_id}", []

        if cmd == "assign_nurse":
            nurse_id = str(params.get("nurse_id", ""))
            ward_id = str(params.get("ward_id", ""))
            if nurse_id not in self._state.nurses:
                violations.append("Unknown nurse_id")
            if ward_id not in self._state.wards:
                violations.append("Unknown ward_id")
            if nurse_id not in self._state.available_nurses:
                violations.append("Nurse not available (call in first)")
            if not violations:
                self._state.nurse_ward_assignments[nurse_id] = ward_id
                return f"Assigned {nurse_id} to {ward_id}", []
            return "Invalid assign_nurse", violations

        return "Unknown command for hard task", [f"Unknown command '{cmd}'"]

    # -----------------
    # Observation helpers
    # -----------------

    def _observation(
        self, *, message: str, violations: list[str], done: bool, reward: float
    ) -> HospitalObservation:
        snapshot = self._snapshot()
        return HospitalObservation(
            message=message,
            task=self._state.task,
            progress=float(self._state.score),
            violations=violations,
            snapshot=snapshot,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "score": float(self._state.score),
                "last_action_error": self._state.last_action_error,
            },
        )

    def _snapshot(self) -> dict:
        # Keep snapshots concise and task-focused.
        if self._state.task == "easy":
            return {
                "patients": self._state.patients,
                "beds": self._state.beds,
                "bed_assignments": self._state.bed_assignments,
            }

        if self._state.task == "medium":
            return {
                "wards": self._state.wards,
                "nurses": list(self._state.nurses.keys()),
                "shift_assignments": self._state.shift_assignments,
                "constraints": {
                    "max_shifts_per_week": 5,
                    "max_consecutive_days": 3,
                    "one_ward_per_day": True,
                },
            }

        # hard
        casualty = {
            pid: p
            for pid, p in self._state.patients.items()
            if bool(p.get("is_casualty", False))
        }
        return {
            "wards": self._state.wards,
            "beds": self._state.beds,
            "patients_casualty": casualty,
            "triage": self._state.triage,
            "bed_assignments": self._state.bed_assignments,
            "available_nurses": self._state.available_nurses,
            "on_call_nurses": self._state.on_call_nurses,
            "nurse_ward_assignments": self._state.nurse_ward_assignments,
        }

    def _help_text(self) -> str:
        if self._state.task == "easy":
            return (
                "Commands (easy): "
                "assign_bed{patient_id,bed_id}, unassign_bed{patient_id}, finalize"
            )
        if self._state.task == "medium":
            return (
                "Commands (medium): "
                "assign_shift{nurse_id,ward_id,day}, remove_shift{nurse_id,ward_id,day}, finalize"
            )
        return (
            "Commands (hard): "
            "triage{patient_id,label}, assign_bed{patient_id,bed_id}, discharge{patient_id}, "
            "call_in{nurse_id}, assign_nurse{nurse_id,ward_id}, finalize"
        )
