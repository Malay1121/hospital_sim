"""Small demo for the Hospital Scheduler client.

Run the server first:
  $env:HOSPITAL_TASK="easy"
  ./.venv/Scripts/python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000

Then run:
  ./.venv/Scripts/python.exe client_demo.py
"""

from __future__ import annotations

import asyncio

from client import HospitalSchedulerEnv


def _pick_easy_action(snapshot: dict) -> tuple[str, dict]:
    patients: dict = snapshot.get("patients", {})
    beds: dict = snapshot.get("beds", {})
    bed_assignments: dict = snapshot.get("bed_assignments", {})

    if not patients or not beds:
        return "help", {}

    used_beds = set(bed_assignments.values())
    patient_id = sorted(patients.keys())[0]
    required_ward = patients[patient_id].get("required_ward")

    for bed_id in sorted(beds.keys()):
        if bed_id in used_beds:
            continue
        if beds[bed_id].get("ward") == required_ward:
            return "assign_bed", {"patient_id": patient_id, "bed_id": bed_id}

    return "help", {}


def _pick_medium_action(snapshot: dict) -> tuple[str, dict]:
    nurses = snapshot.get("nurses", [])
    wards: dict = snapshot.get("wards", {})
    if not nurses or not wards:
        return "help", {}

    nurse_id = sorted(nurses)[0]
    ward_id = sorted(wards.keys())[0]
    return "assign_shift", {"nurse_id": nurse_id, "ward_id": ward_id, "day": 0}


def _pick_hard_action(snapshot: dict) -> tuple[str, dict]:
    casualties: dict = snapshot.get("patients_casualty", {})
    if not casualties:
        return "help", {}

    patient_id = sorted(casualties.keys())[0]
    injury = float(casualties[patient_id].get("injury_score", 0) or 0)
    if injury >= 8:
        label = "red"
    elif injury >= 5:
        label = "yellow"
    else:
        label = "green"
    return "triage", {"patient_id": patient_id, "label": label}


async def main() -> None:
    env = HospitalSchedulerEnv(base_url="http://localhost:8000")
    await env.connect()
    try:
        r = await env.reset()
        print("reset:", r.observation.message)

        r = await env.help()
        print("help:", r.observation.message)

        task = r.observation.task
        snapshot = r.observation.snapshot
        if task == "easy":
            cmd, params = _pick_easy_action(snapshot)
        elif task == "medium":
            cmd, params = _pick_medium_action(snapshot)
        else:
            cmd, params = _pick_hard_action(snapshot)

        r = await env.do(cmd, **params)
        print(f"{cmd}:", r.observation.message, "progress=", r.observation.progress)

        st = await env.state()
        print("state.score=", st.score, "state.step_count=", st.step_count)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
