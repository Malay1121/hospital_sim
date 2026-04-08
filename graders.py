"""Deterministic task graders for the Hospital Scheduler environment.

These graders are used for:
- Reward shaping (dense progress signals)
- Final episode scoring (0.0–1.0)

They are intentionally deterministic and rely only on the environment state.
"""

from __future__ import annotations

from math import ceil
from typing import Dict, Iterable, List, Set

try:
    from .models import HospitalState
except ImportError:  # When running from source
    from models import HospitalState


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def grade(state: HospitalState) -> float:
    """Grade the current episode state, returning a score in [0, 1]."""

    if state.task == "easy":
        return grade_easy_beds(state)
    if state.task == "medium":
        return grade_medium_staffing(state)
    if state.task == "hard":
        return grade_hard_mass_casualty(state)
    return 0.0


def grade_easy_beds(state: HospitalState) -> float:
    patients = state.patients
    beds = state.beds
    assignments = state.bed_assignments

    if not patients:
        return 0.0

    # Bed usage conflicts
    bed_to_patients: Dict[str, List[str]] = {}
    for patient_id, bed_id in assignments.items():
        bed_to_patients.setdefault(bed_id, []).append(patient_id)

    conflicted_patients: Set[str] = set()
    for bed_id, patient_ids in bed_to_patients.items():
        if len(patient_ids) > 1:
            conflicted_patients.update(patient_ids)

    correct = 0
    total = len(patients)

    for patient_id, patient in patients.items():
        bed_id = assignments.get(patient_id)
        if not bed_id:
            continue
        if bed_id not in beds:
            continue
        if patient_id in conflicted_patients:
            continue

        required_ward = patient.get("required_ward")
        bed_ward = beds[bed_id].get("ward")
        if required_ward and bed_ward == required_ward:
            correct += 1

    # Primary progress: fraction correctly placed with no conflicts.
    score = correct / total

    # Small penalty for conflicts and wrong-ward placements.
    wrong_or_conflict = 0
    for patient_id, bed_id in assignments.items():
        if patient_id not in patients or bed_id not in beds:
            wrong_or_conflict += 1
            continue
        if patient_id in conflicted_patients:
            wrong_or_conflict += 1
            continue
        required_ward = patients[patient_id].get("required_ward")
        if beds[bed_id].get("ward") != required_ward:
            wrong_or_conflict += 1

    penalty = 0.02 * wrong_or_conflict
    return _clamp01(score - penalty)


def grade_medium_staffing(state: HospitalState) -> float:
    # Demand is encoded per ward as a list[int] length 7.
    wards = state.wards
    nurses = state.nurses
    assignments = state.shift_assignments

    # Constraints (kept simple + deterministic)
    max_shifts_per_week = 5
    max_consecutive_days = 3

    ward_ids = list(wards.keys())
    if not ward_ids:
        return 0.0

    total_demand = 0
    filled = 0

    # Track per-nurse work days
    nurse_days: Dict[str, Set[int]] = {n: set() for n in nurses.keys()}

    # Coverage calculation + rule bookkeeping
    multi_ward_same_day = 0

    for day in range(7):
        day_key = str(day)
        day_data = assignments.get(day_key, {})

        nurse_to_wards: Dict[str, Set[str]] = {}

        for ward_id in ward_ids:
            demand_list = wards[ward_id].get("demand", [])
            demand = int(demand_list[day]) if day < len(demand_list) else 0
            total_demand += demand

            assigned_nurses = list(day_data.get(ward_id, []))
            unique_assigned = []
            seen = set()
            for nurse_id in assigned_nurses:
                if nurse_id in seen:
                    continue
                seen.add(nurse_id)
                unique_assigned.append(nurse_id)

            # Count filled up to demand
            filled += min(len(unique_assigned), demand)

            for nurse_id in unique_assigned:
                nurse_to_wards.setdefault(nurse_id, set()).add(ward_id)
                if nurse_id in nurse_days:
                    nurse_days[nurse_id].add(day)

        for nurse_id, ward_set in nurse_to_wards.items():
            if len(ward_set) > 1:
                multi_ward_same_day += 1

    if total_demand == 0:
        return 0.0

    coverage = filled / total_demand

    # Labor-rule penalties
    penalty = 0.0

    penalty += 0.05 * multi_ward_same_day

    for nurse_id, days in nurse_days.items():
        shifts = len(days)
        if shifts > max_shifts_per_week:
            penalty += 0.02 * (shifts - max_shifts_per_week)

        if days:
            sorted_days = sorted(days)
            streak = 1
            for i in range(1, len(sorted_days)):
                if sorted_days[i] == sorted_days[i - 1] + 1:
                    streak += 1
                    if streak > max_consecutive_days:
                        penalty += 0.02
                else:
                    streak = 1

    return _clamp01(coverage - penalty)


def grade_hard_mass_casualty(state: HospitalState) -> float:
    patients = state.patients
    beds = state.beds
    triage = state.triage
    bed_assignments = state.bed_assignments

    # Identify casualty patients
    casualty_ids = [
        pid for pid, p in patients.items() if bool(p.get("is_casualty", False))
    ]
    if not casualty_ids:
        return 0.0

    # Ground truth triage from injury_score
    def truth(pid: str) -> str:
        injury = float(patients[pid].get("injury_score", 0))
        if injury >= 8:
            return "red"
        if injury >= 5:
            return "yellow"
        return "green"

    # Score 1: triage correctness
    triage_correct = 0
    for pid in casualty_ids:
        if triage.get(pid) == truth(pid):
            triage_correct += 1
    triage_score = triage_correct / len(casualty_ids)

    # Bed conflicts
    bed_to_patients: Dict[str, List[str]] = {}
    for patient_id, bed_id in bed_assignments.items():
        bed_to_patients.setdefault(bed_id, []).append(patient_id)
    conflicted_beds = {b for b, ps in bed_to_patients.items() if len(ps) > 1}

    # Allowed wards by triage
    allowed = {
        "red": {"ICU", "ED"},
        "yellow": {"ED", "GEN"},
        "green": {"GEN"},
    }

    placed_ok = 0
    for pid in casualty_ids:
        bed_id = bed_assignments.get(pid)
        if not bed_id or bed_id not in beds:
            continue
        if bed_id in conflicted_beds:
            continue
        label = triage.get(pid)
        if label not in allowed:
            continue
        ward = beds[bed_id].get("ward")
        if ward in allowed[label]:
            placed_ok += 1

    bed_score = placed_ok / len(casualty_ids)

    # Staffing adequacy (simple workload model)
    ward_workload: Dict[str, float] = {}
    ward_ids = list(state.wards.keys())

    for pid, bed_id in bed_assignments.items():
        if bed_id not in beds:
            continue
        ward = str(beds[bed_id].get("ward"))
        p = patients.get(pid, {})

        base = float(p.get("acuity", 1.0))
        if p.get("is_casualty"):
            label = triage.get(pid)
            if label == "red":
                base = 2.0
            elif label == "yellow":
                base = 1.5
            elif label == "green":
                base = 1.0
        ward_workload[ward] = ward_workload.get(ward, 0.0) + base

    # Required nurses per ward: ceil(workload / 4)
    required: Dict[str, int] = {}
    for ward in ward_ids:
        wl = ward_workload.get(ward, 0.0)
        required[ward] = max(1, ceil(wl / 4.0)) if wl > 0 else 1

    provided: Dict[str, int] = {ward: 0 for ward in ward_ids}
    for nurse_id, ward in state.nurse_ward_assignments.items():
        if ward in provided:
            provided[ward] += 1

    adequacies: List[float] = []
    for ward in ward_ids:
        req = required.get(ward, 1)
        prov = provided.get(ward, 0)
        adequacies.append(min(1.0, prov / req if req > 0 else 1.0))

    staffing_score = sum(adequacies) / len(adequacies) if adequacies else 0.0

    return _clamp01(0.4 * triage_score + 0.4 * bed_score + 0.2 * staffing_score)
