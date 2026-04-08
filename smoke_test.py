"""Local smoke tests (no server, no Docker).

Runs the environment logic directly and checks that each task can reach a
reasonable score without errors.

Usage (PowerShell):
    ./.venv/Scripts/python.exe smoke_test.py
"""

from __future__ import annotations

import os

from models import HospitalAction
from server.hospital_environment import HospitalSchedulerEnvironment


def _run_easy() -> float:
    os.environ["HOSPITAL_TASK"] = "easy"
    env = HospitalSchedulerEnvironment()
    env.reset()

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
                env.step(
                    HospitalAction(
                        command="assign_bed",
                        parameters={"patient_id": patient_id, "bed_id": bed_id},
                    )
                )
                break

    return float(env.state.score)


def _run_medium() -> float:
    os.environ["HOSPITAL_TASK"] = "medium"
    env = HospitalSchedulerEnvironment()
    env.reset()

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
                    n
                    for n in nurses
                    if n not in used_today and can_work(n, day)
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda n: (shifts_per_nurse[n], n))
                nurse_id = candidates[0]

                used_today.add(nurse_id)
                shifts_per_nurse[nurse_id] += 1
                days_worked[nurse_id].append(day)

                env.step(
                    HospitalAction(
                        command="assign_shift",
                        parameters={
                            "nurse_id": nurse_id,
                            "ward_id": ward_id,
                            "day": day,
                        },
                    )
                )

    return float(env.state.score)


def _run_hard() -> float:
    os.environ["HOSPITAL_TASK"] = "hard"
    env = HospitalSchedulerEnvironment()
    env.reset()

    snap = env.step(HospitalAction(command="status", parameters={})).snapshot
    casualties = snap["patients_casualty"]
    beds = snap["beds"]
    occupied = set(snap.get("bed_assignments", {}).values())

    # Call in all on-call nurses.
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

    # Triage everyone.
    for patient_id, patient in casualties.items():
        injury = float(patient.get("injury_score", 0))
        env.step(
            HospitalAction(
                command="triage",
                parameters={"patient_id": patient_id, "label": label_for(injury)},
            )
        )

    # Place beds without conflicts.
    used = set(occupied)
    for patient_id, patient in casualties.items():
        injury = float(patient.get("injury_score", 0))
        lbl = label_for(injury)
        placed = False
        for ward in allowed_wards(lbl):
            for bed_id, bed in beds.items():
                if bed_id in used:
                    continue
                if bed["ward"] == ward:
                    used.add(bed_id)
                    env.step(
                        HospitalAction(
                            command="assign_bed",
                            parameters={"patient_id": patient_id, "bed_id": bed_id},
                        )
                    )
                    placed = True
                    break
            if placed:
                break

    # Assign called-in nurses to wards to improve staffing.
    # (Any assignment is allowed once they are available.)
    for nurse_id in list(env.state.available_nurses):
        if nurse_id.startswith("OC"):
            target = "ED" if nurse_id in {"OC1", "OC2"} else "ICU"
            env.step(
                HospitalAction(
                    command="assign_nurse",
                    parameters={"nurse_id": nurse_id, "ward_id": target},
                )
            )

    return float(env.state.score)


def main() -> None:
    easy = _run_easy()
    medium = _run_medium()
    hard = _run_hard()

    print(f"easy_score={easy:.2f}")
    print(f"medium_score={medium:.2f}")
    print(f"hard_score={hard:.2f}")

    assert 0.99 <= easy <= 1.01
    assert 0.50 <= medium <= 1.01
    assert 0.40 <= hard <= 1.01

    print("SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
