---
title: Hospital Scheduler Environment Server
emoji: 🏥
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Hospital Bed & Staff Scheduler (OpenEnv)

This environment simulates real hospital operations scheduling:

- Beds: assign patients to ward beds without conflicts
- Staffing: schedule nurse coverage across a week with labor rules
- Mass casualty: triage + reallocate beds + call in off-duty staff

## Tasks

- `easy` — Assign 10 patients to beds across 3 wards (no conflicts, correct ward).
- `medium` — Schedule nurse coverage for 3 wards over 7 days (coverage + labor rules).
- `hard` — Mass casualty event: triage 6 incoming patients, place beds, and ensure ward staffing.

Task selection is done via the `HOSPITAL_TASK` env var (`easy` | `medium` | `hard`).

## Action / Observation

### Action

All actions use a single command-style model:

```json
{
  "command": "assign_bed",
  "parameters": { "patient_id": "P1", "bed_id": "A1" }
}
```

### Observation

Observations provide shaped progress and a structured snapshot:

- `message` (str)
- `task` (str)
- `progress` (float in [0,1])
- `violations` (list[str])
- `snapshot` (dict)

## Run Locally (no Docker)

```bash
./.venv/Scripts/python.exe -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Note: the `/web` UI is disabled by default. To enable it locally:

```powershell
$env:ENABLE_WEB_INTERFACE="true"
$env:HOSPITAL_TASK="easy"  # or: medium | hard
./.venv/Scripts/python.exe -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Client (Python)

There is a typed async client in [client.py](client.py) plus a runnable demo script.

1) Start the server:

```powershell
$env:HOSPITAL_TASK="easy"
./.venv/Scripts/python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

2) Run the demo:

```powershell
./.venv/Scripts/python.exe client_demo.py
```

If you prefer a blocking API for quick scripts, use `HospitalSchedulerSync` from [client.py](client.py).

## Build & Run with Docker

```bash
docker build -t hospital_scheduler-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 -e HOSPITAL_TASK=easy hospital_scheduler-env:latest
```

## Baseline Inference (required by hackathon)

The baseline script is [inference.py](inference.py). It uses the OpenAI client and emits the required `[START]`, `[STEP]`, `[END]` stdout lines.

Required env vars:

- `HF_TOKEN` (or `API_KEY`)
- `API_BASE_URL` (defaults to `https://router.huggingface.co/v1`)
- `MODEL_NAME` (defaults to `Qwen/Qwen2.5-72B-Instruct`)
- `LOCAL_IMAGE_NAME` (optional; defaults to `hospital_scheduler-env:latest`)

## Deploy to Hugging Face Spaces

From this directory (where [openenv.yaml](openenv.yaml) lives):

```bash
openenv push
```
