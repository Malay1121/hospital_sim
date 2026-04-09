"""Baseline inference script for the Hospital Scheduler OpenEnv environment.

MANDATORY (hackathon): emits exactly three stdout line types:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Env vars:
- HF_TOKEN          (required for LLM calls; no default)
- API_BASE_URL      (default: https://router.huggingface.co/v1)
- MODEL_NAME        (default: Qwen/Qwen2.5-72B-Instruct)
- LOCAL_IMAGE_NAME  (optional; default: hospital-scheduler-env:latest)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from client import HospitalSchedulerEnv
    from models import HospitalAction
except ImportError:
    from hospital_scheduler import HospitalSchedulerEnv, HospitalAction  # type: ignore


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = os.getenv("HOSPITAL_BENCHMARK", "hospital_scheduler")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "hospital-scheduler-env:latest"

# When set, skip Docker and connect to this already-running server.
# Useful for local testing without Docker:
#   Start server: uvicorn server.app:app --port 8000
#   Run inference: LOCAL_SERVER_URL=http://localhost:8000 python inference.py
LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL")

TASKS = ["easy", "medium", "hard"]


def _fmt_bool(x: bool) -> str:
    return "true" if x else "false"


def _fmt_reward(r: Optional[float]) -> str:
    if r is None:
        r = 0.0
    return f"{float(r):.2f}"


def _action_str(command: str, params: Dict[str, Any]) -> str:
    # Compact, stable string for log parsing.
    if command in {"assign_bed", "triage", "discharge", "call_in", "assign_nurse", "assign_shift", "remove_shift"}:
        keys = [
            "patient_id",
            "bed_id",
            "label",
            "nurse_id",
            "ward_id",
            "day",
        ]
        args = [str(params[k]) for k in keys if k in params]
        return f"{command}({','.join(args)})"
    if command == "finalize":
        return "finalize()"
    return f"{command}({json.dumps(params, separators=(',',':'))})"


def _heuristic_plan(task: str, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic fallback plan (used if model output can't be parsed)."""

    actions: List[Dict[str, Any]] = []

    if task == "easy":
        patients: Dict[str, Dict[str, Any]] = snapshot.get("patients", {})
        beds: Dict[str, Dict[str, Any]] = snapshot.get("beds", {})

        # Greedy: assign each patient to first free bed in required ward.
        used = set()
        for pid, p in patients.items():
            req = p.get("required_ward")
            for bid, b in beds.items():
                if bid in used:
                    continue
                if b.get("ward") == req:
                    used.add(bid)
                    actions.append({"command": "assign_bed", "parameters": {"patient_id": pid, "bed_id": bid}})
                    break
        actions.append({"command": "finalize", "parameters": {}})
        return actions

    if task == "medium":
        wards: Dict[str, Dict[str, Any]] = snapshot.get("wards", {})
        nurse_ids: List[str] = list(snapshot.get("nurses", []))

        # Greedy coverage with simple labor constraints.
        max_shifts_per_week = 5
        max_consecutive_days = 3

        shifts_per_nurse: Dict[str, int] = {n: 0 for n in nurse_ids}
        days_worked: Dict[str, List[int]] = {n: [] for n in nurse_ids}

        def can_work(nurse_id: str, day: int) -> bool:
            if shifts_per_nurse.get(nurse_id, 0) >= max_shifts_per_week:
                return False
            worked = sorted(days_worked.get(nurse_id, []))
            if day in worked:
                return False
            # check resulting max consecutive streak
            worked2 = sorted(worked + [day])
            streak = 1
            for i in range(1, len(worked2)):
                if worked2[i] == worked2[i - 1] + 1:
                    streak += 1
                    if streak > max_consecutive_days:
                        return False
                else:
                    streak = 1
            return True

        for day in range(7):
            used_today = set()
            # Deterministic order: ward_id sort
            for ward_id in sorted(wards.keys()):
                demand = int(wards[ward_id].get("demand", [0] * 7)[day])
                for _ in range(demand):
                    # Pick lowest-load nurse not used today.
                    candidates = [
                        n
                        for n in nurse_ids
                        if n not in used_today and can_work(n, day)
                    ]
                    if not candidates:
                        break
                    candidates.sort(key=lambda n: (shifts_per_nurse.get(n, 0), n))
                    nurse_id = candidates[0]
                    used_today.add(nurse_id)
                    shifts_per_nurse[nurse_id] = shifts_per_nurse.get(nurse_id, 0) + 1
                    days_worked.setdefault(nurse_id, []).append(day)
                    actions.append(
                        {
                            "command": "assign_shift",
                            "parameters": {"nurse_id": nurse_id, "ward_id": ward_id, "day": day},
                        }
                    )
        actions.append({"command": "finalize", "parameters": {}})
        return actions

    # hard
    casualties: Dict[str, Dict[str, Any]] = snapshot.get("patients_casualty", {})
    beds: Dict[str, Dict[str, Any]] = snapshot.get("beds", {})
    on_call: List[str] = list(snapshot.get("on_call_nurses", []))

    # Call in everyone (safe baseline).
    for nid in on_call:
        actions.append({"command": "call_in", "parameters": {"nurse_id": nid}})

    # Triage from injury_score thresholds.
    def label_for(injury: float) -> str:
        if injury >= 8:
            return "red"
        if injury >= 5:
            return "yellow"
        return "green"

    # Assign triage first.
    for pid, p in casualties.items():
        injury = float(p.get("injury_score", 0))
        actions.append(
            {"command": "triage", "parameters": {"patient_id": pid, "label": label_for(injury)}}
        )

    # Assign beds: pick ward based on label.
    def allowed_wards(lbl: str) -> List[str]:
        if lbl == "red":
            return ["ICU", "ED"]
        if lbl == "yellow":
            return ["ED", "GEN"]
        return ["GEN"]

    used_beds = set()
    # Mark already occupied beds if present in snapshot
    for pid, bid in snapshot.get("bed_assignments", {}).items():
        used_beds.add(bid)

    for pid, p in casualties.items():
        injury = float(p.get("injury_score", 0))
        lbl = label_for(injury)
        placed = False
        for ward in allowed_wards(lbl):
            if placed:
                break
            for bid, b in beds.items():
                if bid in used_beds:
                    continue
                if b.get("ward") == ward:
                    used_beds.add(bid)
                    actions.append({"command": "assign_bed", "parameters": {"patient_id": pid, "bed_id": bid}})
                    placed = True
                    break

    # Assign called-in nurses to wards to improve staffing score.
    # After call_in, they appear in available_nurses but snapshot is from reset,
    # so we assign OC1/OC2 -> ED and OC3 -> ICU (highest acuity wards).
    for nid in on_call:
        target = "ED" if nid in {"OC1", "OC2"} else "ICU"
        actions.append({"command": "assign_nurse", "parameters": {"nurse_id": nid, "ward_id": target}})

    actions.append({"command": "finalize", "parameters": {}})
    return actions


def _model_plan(
    llm: OpenAI,
    task: str,
    observation: Dict[str, Any],
    max_actions: int,
) -> List[Dict[str, Any]]:
    """Ask the model for a full action plan for the episode."""

    system = (
        "You are an agent controlling a hospital scheduling environment. "
        "Return ONLY valid JSON with a top-level key 'actions'. "
        "Each action is an object with keys: command (string) and parameters (object). "
        "Do not include explanations."
    )

    user = {
        "task": task,
        "max_actions": max_actions,
        "allowed_commands": {
            "easy": ["assign_bed", "unassign_bed", "finalize"],
            "medium": ["assign_shift", "remove_shift", "finalize"],
            "hard": ["triage", "assign_bed", "discharge", "call_in", "assign_nurse", "finalize"],
        }[task],
        "observation": observation,
        "output_schema": {"actions": [{"command": "...", "parameters": {}}]},
    }

    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for a in actions[:max_actions]:
        if not isinstance(a, dict):
            continue
        cmd = a.get("command")
        params = a.get("parameters", {})
        if not isinstance(cmd, str) or not isinstance(params, dict):
            continue
        parsed.append({"command": cmd, "parameters": params})

    return parsed


async def _run_task(task: str, server_url: Optional[str] = None) -> Tuple[bool, int, float, List[float]]:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "NO_KEY")

    local_url = server_url or LOCAL_SERVER_URL
    if local_url:
        # Connect to an already-running server (no Docker needed).
        env = HospitalSchedulerEnv(base_url=local_url)
        await env.connect()
    else:
        # Start a fresh container per task (standard evaluation path).
        env = await HospitalSchedulerEnv.from_docker_image(
            IMAGE_NAME,
            env_vars={"HOSPITAL_TASK": task},
        )

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.02

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        reset_result = await env.reset()
        obs = reset_result.observation

        # Plan once with model; fall back to deterministic heuristic.
        max_actions = int(os.getenv("MAX_STEPS", "60"))
        plan: List[Dict[str, Any]] = []
        if API_KEY:
            try:
                plan = _model_plan(llm, task, obs.model_dump(), max_actions=max_actions)
            except Exception:
                plan = []

        if not plan:
            plan = _heuristic_plan(task, obs.snapshot)

        for a in plan:
            steps += 1
            action = HospitalAction(command=a["command"], parameters=a.get("parameters", {}))
            result = await env.step(action)

            r = float(result.reward or 0.0)
            rewards.append(r)

            err = None
            if isinstance(result.observation.metadata, dict):
                err = result.observation.metadata.get("last_action_error")
            err_str = "null" if not err else str(err)

            done = bool(result.done)
            print(
                f"[STEP]  step={steps} action={_action_str(action.command, action.parameters)} "
                f"reward={_fmt_reward(r)} done={_fmt_bool(done)} error={err_str}",
                flush=True,
            )

            if done:
                break

        st = await env.state()
        score = float(getattr(st, "score", 0.02) or 0.02)
        success = score >= 0.98

    except Exception as exc:
        # If something goes wrong, we still must emit [END].
        print(f"[DEBUG] Task {task} failed: {exc}", flush=True)
        score = 0.02
        success = False
    finally:
        try:
            await env.close()
        except Exception:
            pass

        rewards_str = ",".join(_fmt_reward(r) for r in rewards)
        print(
            f"[END]   success={_fmt_bool(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )

    return success, steps, score, rewards


async def main() -> None:
    # Run tasks sequentially to keep runtime predictable.
    for task in TASKS:
        await _run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
