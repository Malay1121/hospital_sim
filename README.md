---
title: Hospital Scheduler Environment Server
emoji: 🏥
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Hospital Bed & Staff Scheduler (OpenEnv)

A real-world hospital operations scheduling environment where an AI agent replaces the human admin who manually assigns patients to beds, schedules nurse shifts, and responds to mass casualty events.

## Motivation

Healthcare scheduling is severely underrepresented in open RL environments. Most existing environments are games or abstract grids. Hospital scheduling is a domain researchers actually need — it has hard constraints (bed conflicts, labor rules, triage protocols), partial progress signals, conflicting objectives, and real consequences. This environment gives the RL community a clean, deterministic, graded testbed for planning agents in high-stakes logistics.

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `easy` | Easy | Assign 10 patients to beds across 3 wards with no conflicts and correct ward placement | 40 |
| `medium` | Medium | Schedule 8 nurses across 3 wards for 7 days, meeting daily demand while respecting labor rules | 80 |
| `hard` | Hard | Mass casualty event — triage 6 incoming patients, reallocate beds, call in off-duty nurses, staff wards | 120 |

Task selection: set the `HOSPITAL_TASK` env var to `easy`, `medium`, or `hard` before starting the server.

### Randomization & reproducibility

Every episode is randomly generated — patient ward assignments, nurse demand schedules, and casualty injury scores all change each run. This means an agent cannot memorise a fixed scenario; it must genuinely reason from the observation.

Pass a `seed` to `reset()` to reproduce an exact episode:

```python
await env.reset(seed=42)   # always produces the same scenario
await env.reset()          # random seed, different scenario each time
```

The active seed is visible in every observation's `metadata.seed` field and in the `HospitalState`.

---

## Action Space

All actions use a single command-style model:

```json
{ "command": "<command>", "parameters": { "<key>": "<value>" } }
```

### Easy task commands

| Command | Parameters | Description |
|---------|-----------|-------------|
| `assign_bed` | `patient_id`, `bed_id` | Place a patient in a bed |
| `unassign_bed` | `patient_id` | Remove a patient from their bed |
| `finalize` | — | End the episode |

### Medium task commands

| Command | Parameters | Description |
|---------|-----------|-------------|
| `assign_shift` | `nurse_id`, `ward_id`, `day` (0–6) | Schedule a nurse to a ward on a given day |
| `remove_shift` | `nurse_id`, `ward_id`, `day` (0–6) | Remove a shift assignment |
| `finalize` | — | End the episode |

### Hard task commands

| Command | Parameters | Description |
|---------|-----------|-------------|
| `triage` | `patient_id`, `label` (`red`/`yellow`/`green`) | Assign triage label to a casualty |
| `assign_bed` | `patient_id`, `bed_id` | Place a patient in a bed |
| `discharge` | `patient_id` | Discharge a low-acuity existing patient to free a bed |
| `call_in` | `nurse_id` | Move an on-call nurse to available |
| `assign_nurse` | `nurse_id`, `ward_id` | Assign an available nurse to a ward |
| `finalize` | — | End the episode |

---

## Observation Space

Every step returns a `HospitalObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Human-readable result of the last action |
| `task` | `str` | Active task (`easy` / `medium` / `hard`) |
| `progress` | `float [0.02, 0.98]` | Current task score |
| `violations` | `list[str]` | Rule violations from the last action |
| `snapshot` | `dict` | Task-focused state view for the agent |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Dense reward signal (see below) |
| `metadata` | `dict` | Episode ID, step count, last error |

---

## Reward Function

```
reward = (new_score - old_score) - 0.01
```

**Why this design:**

- **Delta score (`new_score - old_score`)** — rewards only *new* progress, not accumulated score. An agent can't sit on a high score and keep collecting reward; it must keep improving to earn positive rewards.
- **Time penalty (`-0.01`)** — charges a small cost on every step. Without it, an agent could take arbitrary useless actions and receive the same total reward as a fast, efficient agent. The penalty forces the agent to solve the task in as few steps as possible.
- **Dense signal** — reward is non-zero at every step (not just at episode end), which gives the agent a learning signal throughout the trajectory rather than only at the final outcome.

Episode ends automatically when `score >= 0.98`, `finalize` is called, or max steps is reached.

### Grader details

**Easy** — `correct_placements / 10` minus 0.02 per conflict or wrong-ward assignment.
A placement is correct only if: the bed exists, the bed's ward matches the patient's required ward, and no other patient is assigned to the same bed.

**Medium** — `filled_slots / total_demand` minus penalties for labor rule violations:
- 0.05 per nurse assigned to multiple wards on the same day (one ward per day rule)
- 0.02 per shift exceeding 5 shifts per week (max weekly hours rule)
- 0.02 per day that extends a streak beyond 3 consecutive working days (rest rule)

**Hard** — weighted combination of three components:
- **40% triage correctness** — ground truth: injury ≥ 8 → red, ≥ 5 → yellow, else green
- **40% bed placement correctness** — red must go to ICU or ED, yellow to ED or GEN, green to GEN only; no bed conflicts
- **20% staffing adequacy** — nurses provided vs nurses required per ward (based on patient workload: red=2.0, yellow=1.5, green=1.0 acuity units; one nurse per 4 units)

---

## Baseline Scores

Scores produced by the deterministic heuristic agent in `inference.py` (no LLM required), tested on seed=42. Because episodes are randomly generated, an LLM agent must reason from the observation — it cannot memorise the answer.

| Task | Score | Steps | Seed |
|------|-------|-------|------|
| `easy` | **0.98** | 10 | 42 |
| `medium` | **0.98** | 28 | 42 |
| `hard` | **0.98** | ~15 | 42 |

---

## Setup & Usage

### Prerequisites

```bash
pip install "openenv-core[core]>=0.2.2" fastapi uvicorn
```

Or with uv (uses the lock file):

```bash
uv sync
```

### Run locally (no Docker)

```powershell
$env:HOSPITAL_TASK="easy"   # easy | medium | hard
./.venv/Scripts/python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Smoke test (no server, no Docker)

```powershell
./.venv/Scripts/python.exe smoke_test.py
```

### Run the client demo

```powershell
./.venv/Scripts/python.exe client_demo.py
```

---

## Docker

### Build

Build from the project root (where `Dockerfile` lives):

```bash
docker build -t hospital-scheduler-env:latest .
```

### Run

```bash
# Easy task
docker run --rm -p 8000:8000 -e HOSPITAL_TASK=easy hospital-scheduler-env:latest

# Medium task
docker run --rm -p 8000:8000 -e HOSPITAL_TASK=medium hospital-scheduler-env:latest

# Hard task
docker run --rm -p 8000:8000 -e HOSPITAL_TASK=hard hospital-scheduler-env:latest
```

### Verify

```bash
curl http://localhost:8000/health
# → {"status":"healthy"}
```

---

## Baseline Inference Script

`inference.py` is the required hackathon baseline. It uses the OpenAI client (falls back to a deterministic heuristic if no API key is set) and emits the required log format:

```
[START] task=easy env=hospital_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=assign_bed(P1,A1) reward=0.09 done=false error=null
...
[END]   success=true steps=10 score=1.00 rewards=0.09,...
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace / API key for LLM calls |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `LOCAL_IMAGE_NAME` | `hospital-scheduler-env:latest` | Docker image name |
| `LOCAL_SERVER_URL` | — | Skip Docker, connect to this running server instead |

### Run with Docker (standard / submission path)

```bash
# Build image first
docker build -t hospital-scheduler-env:latest .

# Run inference (spins up a container per task automatically)
python inference.py
```

### Run without Docker (local testing)

```powershell
# Run all 3 tasks automatically
./.venv/Scripts/python.exe test_inference.py
```

---

## OpenEnv Validation

```bash
# Validate project structure
openenv validate .

# Validate a running server
openenv validate --url http://localhost:8000
```

---

## Deploy to Hugging Face Spaces

### Option A — using openenv push (recommended)

```bash
openenv push . --repo-id your-username/hospital-scheduler
```

This uploads the environment to a HuggingFace Space. The Space is configured via the YAML frontmatter at the top of this README (`sdk: docker`, `app_port: 8000`).

### Option B — using git directly

```bash
# Create a new Space at huggingface.co/spaces (Docker SDK, port 8000)
# Then push this directory to it:
git init
git remote add origin https://huggingface.co/spaces/your-username/hospital-scheduler
git add .
git commit -m "initial commit"
git push origin main
```

HuggingFace Spaces will automatically build the Docker image from the root `Dockerfile` and start the server.

### Required Space secrets (set in Space settings)

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Your HuggingFace token (for the inference script) |
| `HOSPITAL_TASK` | Task to run (`easy` / `medium` / `hard`, default: `easy`) |
