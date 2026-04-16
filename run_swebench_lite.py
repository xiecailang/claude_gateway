#!/usr/bin/env python3
"""
SWE-bench Lite runner using Claude Code through the gateway.

For each task:
1. Clone repo at base commit
2. Write CLAUDE.md with task instructions
3. Run Claude Code (pointing to gateway) to solve it
4. Collect the patch

Usage:
    python3 run_swebench_lite.py [--task-id ID] [--start 0] [--num 1]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

WORK_DIR = Path("/tmp/swebench_lite_run")
GATEWAY_DIR = Path("/home/nvidia/my_project/gateway")
TASK_LIST_PATH = Path("/home/nvidia/my_project/pinchbench-test/swebench_results/task_list.json")
MAX_DURATION = 3600  # 1 hour per task


def log(msg="", **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True, **kwargs)


def load_tasks() -> list[dict]:
    """Load tasks from local JSONL file."""
    tasks = []
    with open(TASK_LIST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def clone_and_setup(task: dict) -> str | None:
    """Clone repo at base commit. Returns work dir or None on failure."""
    instance_id = task["instance_id"]
    repo = task["repo"]
    base_commit = task["base_commit"]
    task_dir = WORK_DIR / instance_id

    if task_dir.exists():
        # Already cloned, check if git is set up
        if (task_dir / ".git").exists():
            log(f"  Using existing clone at {task_dir}")
            return str(task_dir)
        shutil.rmtree(str(task_dir))

    task_dir.mkdir(parents=True, exist_ok=True)

    # Use git clone via SSH (HTTPS port 443 is blocked)
    try:
        log(f"  Cloning {repo}@{base_commit[:8]}...")
        ssh_url = f"git@github.com:{repo}.git"
        subprocess.run(
            ["git", "clone", ssh_url, str(task_dir)],
            check=True, timeout=600
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(task_dir), check=True, timeout=60
        )

        log(f"  Repo ready at {task_dir}")
        return str(task_dir)

    except Exception as e:
        log(f"  ERROR cloning {instance_id}: {e}")
        return None


def write_claude_md(task_dir: str, task: dict):
    """Write CLAUDE.md with task instructions."""
    instance_id = task["instance_id"]
    pr_desc = task.get("problem_statement", "")
    hints = task.get("hints_text", "")

    content = f"""# SWE-bench Task: {instance_id}

## Problem Statement
{pr_desc}

{f'## Hints\n{hints}' if hints else ''}

## Instructions
1. Read the problem statement above carefully.
2. Explore the codebase in the current directory to understand the structure.
3. Find the relevant files and code related to the issue.
4. Implement a fix that resolves the problem.
5. Test your fix if possible.
6. When done, create a patch:
   ```
   git diff > /tmp/{instance_id}.patch
   ```
7. Print DONE_PATCH_SAVED at the end.
"""

    claud_md = Path(task_dir) / "CLAUDE.md"
    claud_md.write_text(content)


def run_agent(task_dir: str, task: dict) -> dict:
    """Run the lightweight agent for a single task."""
    instance_id = task["instance_id"]
    log_dir = GATEWAY_DIR / "logs" / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set task_id file for the gateway to route logs
    task_id_file = GATEWAY_DIR / ".current_task_id"
    task_id_file.write_text(instance_id)

    # Task prompt
    prompt = f"""Solve this SWE-bench task.

Instance: {instance_id}

Problem:
{task.get("problem_statement", "")}

{f"Hints: {task.get('hints_text', '')}" if task.get('hints_text') else ''}

Work in the current directory. Explore the codebase, find the issue, fix it, and create a patch."""

    start_time = time.time()

    # Set HF endpoint for any potential downloads
    env = os.environ.copy()
    env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        proc = subprocess.run(
            [sys.executable, str(GATEWAY_DIR / "run_agent.py"),
             "--task-dir", task_dir,
             "--prompt", prompt],
            cwd=task_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=MAX_DURATION
        )

        duration = time.time() - start_time
        result = {
            "instance_id": instance_id,
            "status": "completed" if proc.returncode == 0 else f"failed_{proc.returncode}",
            "duration_seconds": round(duration, 1),
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-1000:],
            "exit_code": proc.returncode,
        }

        # Try to collect patch
        patch_path = f"/tmp/agent_patch_{os.path.basename(task_dir)}.patch"
        if os.path.exists(patch_path):
            with open(patch_path) as pf:
                patch_content = pf.read()
            result["patch"] = patch_content
            result["patch_size"] = len(patch_content)

        # Also check for agent_result.json
        agent_result_path = Path(task_dir) / "agent_result.json"
        if agent_result_path.exists():
            try:
                with open(agent_result_path) as f:
                    agent_result = json.loads(f.read())
                if "patch" in agent_result:
                    result["patch"] = agent_result["patch"]
                    result["patch_size"] = len(agent_result["patch"])
            except Exception:
                pass

        # Copy gateway logs to task dir
        gw_requests = log_dir / "requests.log"
        gw_debug = log_dir / "gateway.log"
        if gw_requests.exists():
            shutil.copy(str(gw_requests), str(Path(task_dir) / "requests.log"))
        if gw_debug.exists():
            shutil.copy(str(gw_debug), str(Path(task_dir) / "gateway.log"))

        # Clear task_id file
        if task_id_file.exists():
            task_id_file.unlink()

        return result

    except subprocess.TimeoutExpired:
        return {
            "instance_id": instance_id,
            "status": "timeout",
            "duration_seconds": MAX_DURATION,
        }


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite via Claude Code + Gateway")
    parser.add_argument("--task-id", help="Run specific task by instance_id")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--num", type=int, default=1, help="Number of tasks")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks
    all_tasks = load_tasks()
    log(f"Loaded {len(all_tasks)} tasks from {TASK_LIST_PATH}")

    if args.task_id:
        tasks = [t for t in all_tasks if t["instance_id"] == args.task_id]
        if not tasks:
            log(f"Task {args.task_id} not found!")
            sys.exit(1)
    else:
        tasks = all_tasks[args.start:args.start + args.num]

    log(f"Running {len(tasks)} task(s)")
    log(f"Gateway: http://127.0.0.1:9000")
    log("")

    results = []
    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        log(f"{'='*60}")
        log(f"[{i+1}/{len(tasks)}] {instance_id}")
        log(f"  Repo: {task['repo']}")
        log(f"  PR: {task.get('problem_statement', '')[:120]}...")

        # Clone and setup
        task_dir = clone_and_setup(task)
        if not task_dir:
            results.append({
                "instance_id": instance_id,
                "status": "clone_failed",
                "duration_seconds": 0,
            })
            continue

        # Write CLAUDE.md
        write_claude_md(task_dir, task)

        # Run agent through gateway
        result = run_agent(task_dir, task)
        results.append(result)

        log(f"  Status: {result['status']}")
        log(f"  Duration: {result.get('duration_seconds', 0):.1f}s")
        if result.get("patch_size"):
            log(f"  Patch: {result['patch_size']} bytes")
        log("")

        # Save progress
        results_file = GATEWAY_DIR / "logs" / "swebench_results.json"
        results_file.write_text(json.dumps(results, indent=2))

    # Summary
    log("=" * 60)
    log("RESULTS SUMMARY")
    for r in results:
        dur = r.get("duration_seconds", 0)
        patch = f"patch={r.get('patch_size', 0)}B" if r.get("patch_size") else "no patch"
        log(f"  {r['instance_id']}: {r['status']} ({dur:.0f}s, {patch})")


if __name__ == "__main__":
    main()
