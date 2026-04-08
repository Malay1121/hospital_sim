"""Local smoke tests (no server, no Docker).

Runs the environment logic directly and checks that:
1. Each task reaches a high score with the heuristic agent.
2. Different seeds produce different scenarios (randomization is working).
3. The same seed always reproduces the same scenario.

Usage (from hospital_sim/):
    ./.venv/Scripts/python.exe smoke_test.py
"""

from __future__ import annotations

import os

from models import HospitalAction
from server.hospital_environment import HospitalSchedulerEnvironment


# ---------------------------------------------------------------------------
# Heuristic solvers
# ---------------------------------------------------------------------------

def _run_easy(seed=None) -> float:
    os.environ["HOSPITAL_TASK"] = "easy"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)

    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    patients = snap["patients"]
    beds = snap["beds"]

    used_beds: set[str] = set()
    for patient_id, patient in patients.items():
        required = patient["required_ward"]
        for bed_id, bed in beds.items():
            if bed_id in used_beds:
                continue
            if bed["ward"] == required:
                used_beds.add(bed_id)
                env.step(HospitalAction(
                    command="assign_bed",
                    parameters={"patient_id": patient_id, "bed_id": bed_id},
                ))
                break

    return float(env.state.score)


def _run_medium(seed=None) -> float:
    os.environ["HOSPITAL_TASK"] = "medium"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)

    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    wards = snap["wards"]
    nurses = list(snap["nurses"])

    max_shifts_per_week = 5
    max_consecutive_days = 3
    shifts_per_nurse = {n: 0 for n in nurses}
    days_worked = {n: [] for n in nurses}

    def can_work(nurse_id: str, day: int) -> bool:
        if shifts_per_nurse[nurse_id] >= max_shifts_per_week:
            return False
        worked = sorted(days_worked[nurse_id])
        if day in worked:
            return False
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
        used_today: set[str] = set()
        for ward_id in sorted(wards.keys()):
            demand = int(wards[ward_id]["demand"][day])
            for _ in range(demand):
                candidates = [
                    n for n in nurses
                    if n not in used_today and can_work(n, day)
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda n: (shifts_per_nurse[n], n))
                nurse_id = candidates[0]
                used_today.add(nurse_id)
                shifts_per_nurse[nurse_id] += 1
                days_worked[nurse_id].append(day)
                env.step(HospitalAction(
                    command="assign_shift",
                    parameters={"nurse_id": nurse_id, "ward_id": ward_id, "day": day},
                ))

    return float(env.state.score)


def _run_hard(seed=None) -> float:
    os.environ["HOSPITAL_TASK"] = "hard"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)

    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    casualties = snap["patients_casualty"]
    beds = snap["beds"]
    occupied = set(snap.get("bed_assignments", {}).values())

    for nurse_id in list(snap.get("on_call_nurses", [])):
        env.step(HospitalAction(command="call_in", parameters={"nurse_id": nurse_id}))

    def label_for(injury: float) -> str:
        if injury >= 8:
            return "red"
        if injury >= 5:
            return "yellow"
        return "green"

    def allowed_wards(lbl: str) -> list[str]:
        if lbl == "red":
            return ["ICU", "ED"]
        if lbl == "yellow":
            return ["ED", "GEN"]
        return ["GEN"]

    for patient_id, patient in casualties.items():
        injury = float(patient.get("injury_score", 0))
        env.step(HospitalAction(
            command="triage",
            parameters={"patient_id": patient_id, "label": label_for(injury)},
        ))

    used = set(occupied)
    for patient_id, patient in casualties.items():
        injury = float(patient.get("injury_score", 0))
        lbl = label_for(injury)
        placed = False
        for ward in allowed_wards(lbl):
            if placed:
                break
            for bed_id, bed in beds.items():
                if bed_id in used:
                    continue
                if bed["ward"] == ward:
                    used.add(bed_id)
                    env.step(HospitalAction(
                        command="assign_bed",
                        parameters={"patient_id": patient_id, "bed_id": bed_id},
                    ))
                    placed = True
                    break

    for nurse_id in list(env.state.available_nurses):
        if nurse_id.startswith("OC"):
            target = "ED" if nurse_id in {"OC1", "OC2"} else "ICU"
            env.step(HospitalAction(
                command="assign_nurse",
                parameters={"nurse_id": nurse_id, "ward_id": target},
            ))

    return float(env.state.score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _easy_snapshot(seed) -> dict:
    """Return the patient→required_ward mapping for a given seed."""
    os.environ["HOSPITAL_TASK"] = "easy"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)
    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    return {pid: p["required_ward"] for pid, p in snap["patients"].items()}


def _medium_demand(seed) -> dict:
    """Return ward demand schedules for a given seed."""
    os.environ["HOSPITAL_TASK"] = "medium"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)
    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    return {wid: w["demand"] for wid, w in snap["wards"].items()}


def _hard_injuries(seed) -> dict:
    """Return casualty injury scores for a given seed."""
    os.environ["HOSPITAL_TASK"] = "hard"
    env = HospitalSchedulerEnvironment()
    env.reset(seed=seed)
    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    return {pid: p["injury_score"] for pid, p in snap["patients_casualty"].items()}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Correctness: heuristic scores ===")
    easy = _run_easy(seed=42)
    medium = _run_medium(seed=42)
    hard = _run_hard(seed=42)

    print(f"easy_score={easy:.2f}   (seed=42)")
    print(f"medium_score={medium:.2f} (seed=42)")
    print(f"hard_score={hard:.2f}   (seed=42)")

    assert 0.99 <= easy <= 1.01,   f"Easy score too low: {easy}"
    assert 0.50 <= medium <= 1.01, f"Medium score too low: {medium}"
    assert 0.40 <= hard <= 1.01,   f"Hard score too low: {hard}"

    print("\n=== Randomization: different seeds produce different scenarios ===")
    snap_42  = _easy_snapshot(42)
    snap_99  = _easy_snapshot(99)
    snap_123 = _easy_snapshot(123)
    print(f"seed=42  patients: {snap_42}")
    print(f"seed=99  patients: {snap_99}")
    print(f"seed=123 patients: {snap_123}")
    assert snap_42 != snap_99 or snap_42 != snap_123, \
        "Different seeds produced identical easy scenarios — randomization may be broken"

    demand_42  = _medium_demand(42)
    demand_99  = _medium_demand(99)
    print(f"seed=42  demand: {demand_42}")
    print(f"seed=99  demand: {demand_99}")
    assert demand_42 != demand_99, \
        "Different seeds produced identical medium demand — randomization may be broken"

    inj_42 = _hard_injuries(42)
    inj_99 = _hard_injuries(99)
    print(f"seed=42  injuries: {inj_42}")
    print(f"seed=99  injuries: {inj_99}")
    assert inj_42 != inj_99, \
        "Different seeds produced identical hard injuries — randomization may be broken"

    print("\n=== Reproducibility: same seed produces same scenario ===")
    snap_42_again = _easy_snapshot(42)
    assert snap_42 == snap_42_again, "Same seed produced different easy scenarios!"
    demand_42_again = _medium_demand(42)
    assert demand_42 == demand_42_again, "Same seed produced different medium demand!"
    inj_42_again = _hard_injuries(42)
    assert inj_42 == inj_42_again, "Same seed produced different hard injuries!"
    print("seed=42 reproduced identically across all tasks")

    print("\nSMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
