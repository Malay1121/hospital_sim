"""Local inference test runner — no Docker required.

Spins up a server for each task, runs inference against it, then tears it down.

Usage:
    ../.venv/Scripts/python.exe test_inference.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

PYTHON = sys.executable
TASKS = ["easy", "medium", "hard"]
BASE_PORT = 8100


def _free_port(port: int) -> int:
    """Return port if free, else next free port."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
        port += 1


def _wait_for_server(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                return True
        time.sleep(0.3)
    return False


async def _run_task_against_server(task: str, port: int) -> None:
    sys.path.insert(0, os.path.dirname(__file__))
    from inference import _run_task  # noqa: PLC0415
    await _run_task(task, server_url=f"http://localhost:{port}")


def main() -> None:
    results = {}

    for task in TASKS:
        port = _free_port(BASE_PORT + TASKS.index(task) * 10)
        print(f"\n{'='*50}", flush=True)
        print(f"Starting server for task={task} on port={port}", flush=True)
        print(f"{'='*50}", flush=True)

        env = {**os.environ, "HOSPITAL_TASK": task}
        proc = subprocess.Popen(
            [
                PYTHON, "-m", "uvicorn", "server.app:app",
                "--host", "0.0.0.0",
                "--port", str(port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            if not _wait_for_server(port):
                print(f"ERROR: server for {task} did not start in time", flush=True)
                results[task] = "FAILED (server timeout)"
                continue

            asyncio.run(_run_task_against_server(task, port))
            results[task] = "OK"

        except Exception as exc:
            print(f"ERROR running {task}: {exc}", flush=True)
            results[task] = f"FAILED ({exc})"

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    for task, status in results.items():
        print(f"  {task}: {status}", flush=True)


if __name__ == "__main__":
    main()
